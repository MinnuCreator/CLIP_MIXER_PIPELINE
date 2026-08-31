"""FrameForge V5.1 Sequential Image Editor — Improved Cut Timing.

Walks through the user's ordered image sequence and makes intelligent
cut/hold decisions based on narration semantics, CLIP visual-text
similarity, pause detection, visual beat analysis, duration pressure,
visual freshness, and capacity-aware pacing.

Key principles:
- Image order is AUTHORITATIVE (never globally reordered)
- Forward-only progression through the image sequence
- Sub-segment beat evaluation for finer cut granularity
- Adaptive shot durations based on capacity (remaining images vs narration)
- Each image used at most twice
- No A->B->A->B oscillation
- Every cut has an inspectable reason
- Genuine-signal tracking: 0.5 fallback values are excluded from scoring
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

import numpy as np

from .config import DEFAULT_CONFIG, FrameForgeConfig
from .models import ImageAsset, Scene, TranscriptSegment, Word
from .embeddings import EmbeddingEngine
from .visual import VisualUnderstandingEngine

logger = logging.getLogger("frameforge.editor")

# ── Sub-segment beat detection ──────────────────────────────────────────────
_CLAUSE_SPLIT = re.compile(
    r"[,;\u2014\u2013]|\b(?:and|but|or|while|whereas|however|although|because|since|when|then|so|yet|still|also|just|now|then|after|before|until|although)\b"
)
_SENTENCE_END = re.compile(r"[.!?]$")
_STRONG_SENTENCE_END = re.compile(r"[.!?]{1,3}$")

MAX_IMAGE_USES = 2


class EditDecision:
    """Records the editor's reasoning for a cut or hold."""
    __slots__ = (
        "decision", "reason", "cut_score", "semantic_shift",
        "current_relevance", "next_relevance", "pause_bonus",
        "duration_pressure", "visual_freshness", "llm_hint",
        "voice_image", "voice_prompt", "prompt_image",
        "usage_count", "beat_text", "signal_quality",
    )

    def __init__(
        self,
        decision: str,
        reason: str,
        cut_score: float = 0.0,
        semantic_shift: float = 0.0,
        current_relevance: float = 0.0,
        next_relevance: float = 0.0,
        pause_bonus: float = 0.0,
        duration_pressure: float = 0.0,
        visual_freshness: float = 0.0,
        llm_hint: float = 0.0,
        voice_image: float = 0.0,
        voice_prompt: float = 0.0,
        prompt_image: float = 0.0,
        usage_count: int = 0,
        beat_text: str = "",
        signal_quality: float = 1.0,
    ):
        self.decision = decision
        self.reason = reason
        self.cut_score = cut_score
        self.semantic_shift = semantic_shift
        self.current_relevance = current_relevance
        self.next_relevance = next_relevance
        self.pause_bonus = pause_bonus
        self.duration_pressure = duration_pressure
        self.visual_freshness = visual_freshness
        self.llm_hint = llm_hint
        self.voice_image = voice_image
        self.voice_prompt = voice_prompt
        self.prompt_image = prompt_image
        self.usage_count = usage_count
        self.beat_text = beat_text
        self.signal_quality = signal_quality

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "cut_score": round(self.cut_score, 4),
            "semantic_shift": round(self.semantic_shift, 4),
            "current_relevance": round(self.current_relevance, 4),
            "next_relevance": round(self.next_relevance, 4),
            "pause_bonus": round(self.pause_bonus, 4),
            "duration_pressure": round(self.duration_pressure, 4),
            "visual_freshness": round(self.visual_freshness, 4),
            "llm_hint": round(self.llm_hint, 4),
            "voice_image": round(self.voice_image, 4),
            "voice_prompt": round(self.voice_prompt, 4),
            "prompt_image": round(self.prompt_image, 4),
            "usage_count": self.usage_count,
            "beat_text": self.beat_text,
            "signal_quality": round(self.signal_quality, 4),
        }


class _SubBeat:
    """A sub-segment beat within a Whisper segment for finer cut granularity."""
    __slots__ = ("start", "end", "text", "prev_text", "is_sentence_end", "is_clause_boundary")

    def __init__(self, start: float, end: float, text: str, prev_text: str = "",
                 is_sentence_end: bool = False, is_clause_boundary: bool = False):
        self.start = start
        self.end = end
        self.text = text
        self.prev_text = prev_text
        self.is_sentence_end = is_sentence_end
        self.is_clause_boundary = is_clause_boundary


def _split_segment_to_beats(
    segment: TranscriptSegment,
    target_duration: float = 2.5,
    min_gap: float = 0.15,
) -> list[_SubBeat]:
    """Split a Whisper segment into sub-beats for finer cut granularity.

    Uses word-level timestamps, sentence endings, clause boundaries, and
    silence gaps to identify meaningful visual beat points.
    """
    words = segment.words or []
    seg_duration = segment.end - segment.start

    if seg_duration <= target_duration * 1.4 or len(words) < 3:
        return [_SubBeat(segment.start, segment.end, segment.text)]

    split_scores: list[tuple[float, int]] = []

    for i, word in enumerate(words):
        elapsed = word.end - segment.start

        if elapsed < target_duration * 0.4:
            continue
        if elapsed > seg_duration - target_duration * 0.3:
            continue

        score = 0.0
        word_stripped = word.word.strip()

        # Sentence ending is the strongest signal
        if _STRONG_SENTENCE_END.search(word_stripped):
            score = 1.0
        elif _SENTENCE_END.search(word_stripped):
            score = 0.9
        elif _CLAUSE_SPLIT.search(word_stripped.lower()):
            score = 0.7

        # Silence gap indicates a natural pause — strong cut opportunity
        if i < len(words) - 1:
            gap = words[i + 1].start - word.end
            if gap >= min_gap:
                gap_score = 0.5 + 0.4 * min(gap / 0.4, 1.0)
                score = max(score, gap_score)

        # Proximity to ideal split point
        dist = abs(elapsed - target_duration)
        proximity = max(0.0, 1.0 - dist / target_duration)
        score *= (0.4 + 0.6 * proximity)

        if score > 0.1:
            split_scores.append((score, i))

    if not split_scores:
        return [_SubBeat(segment.start, segment.end, segment.text)]

    # Sort by word index, keep at most 4 best splits
    split_scores.sort(key=lambda x: x[1])
    if len(split_scores) > 4:
        split_scores.sort(key=lambda x: x[0], reverse=True)
        split_scores = sorted(split_scores[:4], key=lambda x: x[1])

    beats: list[_SubBeat] = []
    prev_start = segment.start
    prev_text = ""
    split_indices = [s[1] for s in split_scores]

    for idx in split_indices:
        word = words[idx]
        beat_end = word.end
        beat_words = []
        for w in words:
            if w.start >= prev_start and w.end <= beat_end + 0.01:
                beat_words.append(w.word)
        beat_text = " ".join(beat_words).strip()
        if beat_text:
            is_sent = bool(_SENTENCE_END.search(word.word.strip()))
            is_clause = bool(_CLAUSE_SPLIT.search(word.word.strip().lower()))
            beats.append(_SubBeat(prev_start, beat_end, beat_text, prev_text,
                                  is_sentence_end=is_sent, is_clause_boundary=is_clause))
            prev_text = beat_text
            prev_start = beat_end

    remaining_words = []
    for w in words:
        if w.start >= prev_start - 0.01:
            remaining_words.append(w.word)
    remaining_text = " ".join(remaining_words).strip()
    if remaining_text:
        beats.append(_SubBeat(prev_start, segment.end, remaining_text, prev_text))

    return beats if beats else [_SubBeat(segment.start, segment.end, segment.text)]


class SequentialEditor:
    """Intelligent sequential visual editor with capacity-aware pacing.

    The core insight: shot duration is determined by how many images
    the user provided vs how long the narration is.  The editor
    adapts its pacing target dynamically so images are spread evenly
    across the narration rather than burned through early.

    V5.1 improvements:
    - Genuine-signal tracking: fallback 0.5 values are excluded from scoring
    - Non-linear duration pressure: exponential overshoot penalty
    - Relevance decay: current image relevance decreases over beats
    - Stronger pause/scoring at sentence boundaries
    - LLM confidence-weighted hints
    """

    def __init__(
        self,
        min_shot_duration: float | None = None,
        target_shot_duration: float | None = None,
        max_shot_duration: float | None = None,
        min_duration_for_cut: float | None = None,
        max_image_uses: int = MAX_IMAGE_USES,
        cut_threshold: float | None = None,
        sub_segment_gap: float | None = None,
        config: Optional[FrameForgeConfig] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.min_shot_duration = min_shot_duration if min_shot_duration is not None else self.config.min_shot_duration
        self.target_shot_duration = target_shot_duration if target_shot_duration is not None else self.config.target_shot_duration
        self.max_shot_duration = max_shot_duration if max_shot_duration is not None else self.config.max_shot_duration
        self.min_duration_for_cut = min_duration_for_cut if min_duration_for_cut is not None else self.config.min_duration_for_cut
        self.max_image_uses = max_image_uses
        self.cut_threshold = cut_threshold if cut_threshold is not None else self.config.cut_threshold
        self.sub_segment_gap = sub_segment_gap if sub_segment_gap is not None else self.config.sub_segment_gap

    # ── Public API ──────────────────────────────────────────────────────────

    def plan(
        self,
        segments: list[TranscriptSegment],
        images: list[ImageAsset],
        audio_duration: float,
        voice_image_matrix: Optional[np.ndarray] = None,
        voice_prompt_matrix: Optional[np.ndarray] = None,
        prompt_image_scores: Optional[list[float]] = None,
        llm_plan: Optional[dict[str, Any]] = None,
    ) -> tuple[list[Scene], dict[str, Any]]:
        """Plan sequential scenes with cut/hold decisions.

        If *llm_plan* is provided (from Gemini), its recommended cut points
        are used as timing hints — the editor adjusts its threshold near
        LLM-suggested boundaries but still enforces all hard constraints.
        """
        if not images:
            raise ValueError("At least one image is required.")
        if not segments:
            return self._no_narration_fallback(images, audio_duration)

        n_imgs = len(images)

        # ── Parse LLM plan into a time-indexed lookup ────────────────────
        # V6 schema: {start, end_hint, decision, reason, confidence}
        # V5 compat: {image_index, action, start, end_hint, reason}
        llm_advance_times: dict[float, float] = {}  # time -> confidence
        llm_hold_times: dict[float, float] = {}     # time -> confidence
        if llm_plan and "beats" in llm_plan:
            for beat in llm_plan["beats"]:
                t = round(beat.get("end_hint", 0), 2)
                # V6 uses 'decision', V5 uses 'action'
                decision = beat.get("decision", beat.get("action", "HOLD"))
                conf = beat.get("confidence", 0.8)
                if decision in ("ADVANCE", "CUT"):
                    llm_advance_times[t] = conf
                else:
                    llm_hold_times[t] = conf
            logger.info("LLM plan loaded: %d beats, %d advance points.",
                        len(llm_plan.get("beats", [])), len(llm_advance_times))

        # ── Determine which signals are genuine vs fallback ──────────────
        has_vim = voice_image_matrix is not None
        has_vpm = voice_prompt_matrix is not None
        has_pim = prompt_image_scores is not None

        # ── Pre-compute per-image similarity vectors ────────────────────
        img_sims = self._compute_similarities(
            segments, images, voice_image_matrix, voice_prompt_matrix, prompt_image_scores
        )

        # ── Split long segments into sub-beats ──────────────────────────
        all_beats: list[tuple[_SubBeat, int, TranscriptSegment]] = []
        for t, seg in enumerate(segments):
            beats = _split_segment_to_beats(seg, self.target_shot_duration, self.sub_segment_gap)
            for b in beats:
                all_beats.append((b, t, seg))

        # ── Compute global pacing target ────────────────────────────────
        total_slots = n_imgs * self.max_image_uses
        global_target = audio_duration / max(1, total_slots)
        global_target = max(self.min_shot_duration, min(global_target, self.max_shot_duration))

        logger.info(
            "Editor pacing: %d images, max_uses=%d, %.1fs narration -> "
            "global_target=%.1fs per shot, signals: CLIP=%s text=%s prompt_image=%s",
            n_imgs, self.max_image_uses, audio_duration, global_target,
            has_vim, has_vpm, has_pim,
        )

        # ── Sequential walk over beats ──────────────────────────────────
        img_idx = 0
        usage: dict[str, int] = {}
        scenes: list[Scene] = []
        cut_log: list[dict] = []
        current_scene_start = all_beats[0][0].start
        current_narration_parts: list[str] = []
        held_duration = 0.0
        last_cut_time = -10.0
        scenes_on_current = 0

        usage[images[0].id] = 1

        for beat_idx, (beat, seg_idx, parent_seg) in enumerate(all_beats):
            beat_duration = beat.end - beat.start
            current_img = images[img_idx]

            # ── Adaptive pacing target ──────────────────────────────────
            remaining_narration = audio_duration - beat.start
            remaining_fwd = n_imgs - img_idx - 1
            reusable = sum(
                1 for j in range(img_idx + 1)
                if usage.get(images[j].id, 0) < self.max_image_uses
            )
            remaining_images = remaining_fwd + reusable

            if remaining_images > 0:
                per_image = remaining_narration / remaining_images
            else:
                per_image = self.max_shot_duration

            adaptive_target = max(global_target, per_image)
            adaptive_target = min(adaptive_target, self.max_shot_duration)

            # ── Find next usable image ──────────────────────────────────
            next_img, next_idx = self._find_next_image(images, img_idx, n_imgs, usage)

            # Gather similarity data for current image
            sims = img_sims[img_idx]
            vi_score = float(voice_image_matrix[seg_idx, img_idx]) if has_vim else sims["voice_image"][seg_idx]
            vp_score = float(voice_prompt_matrix[seg_idx, img_idx]) if has_vpm else sims["voice_prompt"][seg_idx]
            pi_score = float(prompt_image_scores[img_idx]) if has_pim else sims["prompt_image"]

            # Gather similarity data for next image
            next_vi = 0.0
            next_vp = 0.0
            next_pi = 0.0
            if next_img is not None:
                next_sims = img_sims[next_idx]
                next_vi = float(voice_image_matrix[seg_idx, next_idx]) if has_vim else next_sims["voice_image"][seg_idx]
                next_vp = float(voice_prompt_matrix[seg_idx, next_idx]) if has_vpm else next_sims["voice_prompt"][seg_idx]
                next_pi = float(prompt_image_scores[next_idx]) if has_pim else next_sims["prompt_image"]

            # Compute signals
            semantic_shift = self._compute_semantic_shift(beat, beat_idx, all_beats)
            pause_bonus = self._compute_pause_bonus(all_beats, beat_idx)

            # Signal quality: how many of the 3 main signals are genuine
            signal_quality = self._compute_signal_quality(has_vim, has_vpm, has_pim)

            # Relevance: weighted by signal quality (genuine signals get full weight)
            current_relevance = self._relevance_score(
                vi_score, vp_score, pi_score, signal_quality
            )
            next_relevance = self._relevance_score(
                next_vi, next_vp, next_pi, signal_quality
            ) if next_img is not None else 0.0

            # Duration pressure: non-linear (exponential past target)
            duration_pressure = self._compute_duration_pressure(held_duration, adaptive_target)

            # Visual freshness
            visual_freshness = self._compute_visual_freshness(scenes_on_current)

            # Relevance decay: current image relevance decays slightly per beat
            relevance_decay = self._compute_relevance_decay(scenes_on_current)
            current_relevance *= relevance_decay

            # ── CUT DECISION ────────────────────────────────────────────
            can_cut = (
                next_img is not None
                and held_duration >= self.min_duration_for_cut
                and usage.get(next_img.id, 0) < self.max_image_uses
            )

            # Capacity-aware minimum hold time
            capacity_min_hold = adaptive_target * 0.45
            if held_duration < capacity_min_hold:
                can_cut = False

            decision = "HOLD"
            reason = ""
            cut_score = 0.0
            llm_hint = 0.0

            if can_cut:
                cut_score = self._compute_cut_score(
                    current_relevance, next_relevance, semantic_shift,
                    pause_bonus, duration_pressure, visual_freshness,
                )

                # LLM hint: boost/reduce score based on Gemini recommendations
                # V6: ADVANCE boosts cut score, HOLD reduces it
                if llm_advance_times:
                    closest = min(llm_advance_times, key=lambda t: abs(t - beat.start), default=None)
                    if closest is not None and abs(closest - beat.start) < 1.5:
                        conf = llm_advance_times[closest]
                        llm_hint = 0.05 + 0.20 * conf  # 0.05 to 0.25 boost
                        cut_score = min(1.0, cut_score + llm_hint)
                if llm_hold_times:
                    closest_hold = min(llm_hold_times, key=lambda t: abs(t - beat.start), default=None)
                    if closest_hold is not None and abs(closest_hold - beat.start) < 1.5:
                        conf = llm_hold_times[closest_hold]
                        # LLM says HOLD: reduce cut score
                        llm_hint = -(0.03 + 0.07 * conf)  # -0.03 to -0.10
                        cut_score = max(0.0, cut_score + llm_hint)

                # Too-soon-cut penalty: strong inhibition after recent cut
                time_since_last_cut = beat.start - last_cut_time
                if time_since_last_cut < self.min_shot_duration:
                    cut_score *= 0.2
                elif time_since_last_cut < self.min_shot_duration * 1.5:
                    cut_score *= 0.6

                decision, reason = self._evaluate_cut(
                    cut_score, current_relevance, next_relevance,
                    semantic_shift, pause_bonus, duration_pressure,
                    visual_freshness, current_img, next_img, parent_seg,
                    usage.get(current_img.id, 0), held_duration,
                    adaptive_target, llm_hint, signal_quality,
                )

            if decision == "CUT":
                scene_end = beat.start
                scene_end = max(scene_end, current_scene_start + self.min_shot_duration)
                scene_end = min(scene_end, audio_duration)

                if scene_end - current_scene_start < self.min_shot_duration * 0.5:
                    decision = "HOLD"
                    reason = f"Scene too short ({scene_end - current_scene_start:.2f}s < min)"
                else:
                    scene = self._make_scene(
                        scenes, current_img, current_scene_start, scene_end,
                        " ".join(current_narration_parts), cut_score,
                        voice_image=vi_score, voice_prompt=vp_score, prompt_image=pi_score,
                        decision=decision, reason=reason,
                        usage_count=usage.get(current_img.id, 0),
                        beat_text=beat.text,
                        signal_quality=signal_quality,
                    )
                    scenes.append(scene)

                    cut_log.append({
                        "cut_at": round(scene_end, 4),
                        "from_image": current_img.id,
                        "to_image": next_img.id if next_img is not None else None,
                        "score": round(cut_score, 4),
                        "reason": reason,
                        "held_duration": round(held_duration, 3),
                        "llm_hint": round(llm_hint, 4),
                    })

                    # Advance to next image
                    img_idx = next_idx
                    current_img = images[img_idx]
                    usage[current_img.id] = usage.get(current_img.id, 0) + 1
                    current_scene_start = beat.start
                    current_narration_parts = []
                    held_duration = 0.0
                    scenes_on_current = 0
                    last_cut_time = beat.start

                    sims = img_sims[img_idx]
                    vi_score = float(voice_image_matrix[seg_idx, img_idx]) if has_vim else sims["voice_image"][seg_idx]
                    vp_score = float(voice_prompt_matrix[seg_idx, img_idx]) if has_vpm else sims["voice_prompt"][seg_idx]
                    pi_score = float(prompt_image_scores[img_idx]) if has_pim else sims["prompt_image"]

            elif decision == "HOLD":
                if current_img.id not in usage:
                    usage[current_img.id] = 1

            current_narration_parts.append(beat.text)
            held_duration += beat_duration
            scenes_on_current += 1

            # Enforce max shot duration (force cut even if below adaptive target)
            if held_duration >= self.max_shot_duration:
                next_for_force, next_idx_force = self._find_next_image(images, img_idx, n_imgs, usage)
                if next_for_force is not None and usage.get(next_for_force.id, 0) < self.max_image_uses:
                    scene_end = min(beat.end, audio_duration)
                    scene = self._make_scene(
                        scenes, current_img, current_scene_start, scene_end,
                        " ".join(current_narration_parts), 1.0,
                        voice_image=vi_score, voice_prompt=vp_score, prompt_image=pi_score,
                        decision="CUT",
                        reason=f"Max shot duration ({self.max_shot_duration:.1f}s) reached",
                        usage_count=usage.get(current_img.id, 0),
                        beat_text=beat.text,
                        signal_quality=signal_quality,
                    )
                    scenes.append(scene)
                    img_idx = next_idx_force
                    current_img = images[img_idx]
                    usage[current_img.id] = usage.get(current_img.id, 0) + 1
                    current_scene_start = scene_end
                    current_narration_parts = []
                    held_duration = 0.0
                    scenes_on_current = 0
                    last_cut_time = scene_end

        # ── Close final scene ───────────────────────────────────────────
        if current_narration_parts:
            final_img = images[min(img_idx, n_imgs - 1)]
            scene = self._make_scene(
                scenes, final_img, current_scene_start, audio_duration,
                " ".join(current_narration_parts), 1.0,
                voice_image=vi_score if scenes else 0.5,
                voice_prompt=vp_score if scenes else 0.5,
                prompt_image=pi_score if scenes else 0.5,
                decision="HOLD",
                reason="Final narration segment",
                usage_count=usage.get(final_img.id, 0),
                beat_text="",
                signal_quality=signal_quality if scenes else 0.0,
            )
            scenes.append(scene)

        # ── Post-process: reconcile boundaries ──────────────────────────
        self._reconcile_boundaries(scenes, audio_duration)

        # ── Build metadata ──────────────────────────────────────────────
        used_ids = set()
        for s in scenes:
            used_ids.add(s.image_id)

        durations = [s.duration for s in scenes]
        metadata = {
            "total_input_images": n_imgs,
            "images_used": len(used_ids),
            "unused_images": n_imgs - len(used_ids),
            "unique_images_used": len(used_ids),
            "repeat_count": sum(1 for sid, cnt in usage.items() if cnt > 1),
            "max_usage_count": max(usage.values()) if usage else 0,
            "num_cuts": len(cut_log),
            "num_beats": len(all_beats),
            "average_shot_duration": round(sum(durations) / max(1, len(durations)), 3),
            "min_shot_duration": round(min(durations), 3) if durations else 0,
            "max_shot_duration": round(max(durations), 3) if durations else 0,
            "image_order_preserved": True,
            "cut_log": cut_log,
            "usage_counts": dict(usage),
            "global_target": round(global_target, 3),
            "llm_used": bool(llm_plan),
            "llm_advance_points": len(llm_advance_times),
            "llm_hold_points": len(llm_hold_times),
            "signal_quality": round(signal_quality, 4),
            "signals_active": {
                "voice_image": has_vim,
                "voice_prompt": has_vpm,
                "prompt_image": has_pim,
            },
        }

        return scenes, metadata

    # ── Private helpers ─────────────────────────────────────────────────────

    def _find_next_image(
        self, images: list[ImageAsset], current_idx: int,
        n_imgs: int, usage: dict[str, int],
    ) -> tuple[Optional[ImageAsset], int]:
        """Find the next image to cut to."""
        next_idx = current_idx + 1
        if next_idx < n_imgs:
            return images[next_idx], next_idx

        # All forward images used — look for first reusable image
        for j in range(n_imgs):
            if usage.get(images[j].id, 0) < self.max_image_uses:
                return images[j], j

        return None, -1

    def _no_narration_fallback(
        self, images: list[ImageAsset], audio_duration: float
    ) -> tuple[list[Scene], dict[str, Any]]:
        scene = Scene(
            scene_id="scene-001",
            image_id=images[0].id,
            start=0.0,
            end=audio_duration,
            narration="",
            confidence=1.0,
            rerank_breakdown={"decision": "HOLD", "reason": "No narration segments"},
        )
        return [scene], {
            "total_input_images": len(images),
            "images_used": 1,
            "unused_images": len(images) - 1,
            "num_cuts": 0,
            "image_order_preserved": True,
        }

    def _compute_similarities(
        self,
        segments: list[TranscriptSegment],
        images: list[ImageAsset],
        voice_image_matrix: Optional[np.ndarray],
        voice_prompt_matrix: Optional[np.ndarray],
        prompt_image_scores: Optional[list[float]],
    ) -> list[dict[str, list[float]]]:
        n_segs = len(segments)
        result = []
        for j, img in enumerate(images):
            vi = []
            vp = []
            for t in range(n_segs):
                vi.append(float(voice_image_matrix[t, j]) if voice_image_matrix is not None else 0.5)
                vp.append(float(voice_prompt_matrix[t, j]) if voice_prompt_matrix is not None else 0.5)
            pi = float(prompt_image_scores[j]) if prompt_image_scores is not None else 0.5
            result.append({"voice_image": vi, "voice_prompt": vp, "prompt_image": pi})
        return result

    def _compute_signal_quality(self, has_vim: bool, has_vpm: bool, has_pim: bool) -> float:
        """Compute fraction of scoring signals that are genuine (not fallback 0.5).

        When CLIP is unavailable, voice_image falls back to 0.5.
        When sentence-transformers is unavailable, voice_prompt falls back to 0.5.
        When CLIP is unavailable, prompt_image falls back to 0.5.

        A signal_quality of 1.0 means all signals are real.
        A signal_quality of 0.33 means only 1 of 3 signals is real.
        """
        return sum([has_vim, has_vpm, has_pim]) / 3.0

    def _relevance_score(self, vi: float, vp: float, pi: float, signal_quality: float) -> float:
        """Compute image relevance from the three signals.

        Genuine signals use full weight; fallback 0.5 values are
        de-weighted so they don't artificially inflate or deflate scores.
        """
        if signal_quality >= 0.9:
            # All signals genuine: use full weighted average
            return (0.40 * vi + 0.40 * vp + 0.20 * pi)
        elif signal_quality >= 0.6:
            # 2 of 3 signals genuine: average only genuine ones
            # Approximate: if has_vim and has_vpm but not has_pim
            # Then use (vi + vp) / 2
            return (0.50 * vi + 0.50 * vp)
        elif signal_quality > 0.0:
            # Only 1 signal genuine: use it directly
            return max(vi, vp, pi)
        else:
            # No genuine signals: return 0.5 (neutral) — this signal contributes nothing
            return 0.5

    def _compute_semantic_shift(self, beat: _SubBeat, beat_idx: int, all_beats: list) -> float:
        """Compute semantic shift between consecutive beats.

        Uses Jaccard word overlap as the primary signal, with boosts for:
        - Segment boundary transitions (sentence end to new segment)
        - Strong sentence endings
        """
        if beat_idx > 0:
            prev_beat = all_beats[beat_idx - 1][0]
            prev_words = set(prev_beat.text.lower().split())
        else:
            prev_words = set()
        curr_words = set(beat.text.lower().split())
        if not prev_words or not curr_words:
            return 0.0 if beat_idx == 0 else 0.5

        intersection = prev_words & curr_words
        union = prev_words | curr_words
        jaccard = len(intersection) / len(union) if union else 0.0
        shift = 1.0 - jaccard

        # Boost shift at sentence endings — these are natural cut points
        if beat_idx > 0:
            prev_beat_obj = all_beats[beat_idx - 1][0]
            if prev_beat_obj.is_sentence_end:
                shift = min(1.0, shift + 0.25)
            elif prev_beat_obj.is_clause_boundary:
                shift = min(1.0, shift + 0.15)

        # Boost at segment boundaries (Whisper segment transitions)
        _, prev_seg_idx, _ = all_beats[beat_idx - 1] if beat_idx > 0 else (None, -1, None)
        _, curr_seg_idx, _ = all_beats[beat_idx]
        if curr_seg_idx != prev_seg_idx and beat_idx > 0:
            shift = min(1.0, shift + 0.10)

        return shift

    def _compute_pause_bonus(self, all_beats: list, beat_idx: int) -> float:
        """Compute pause bonus from silence gaps and segment transitions.

        Strong pauses indicate natural cut points.
        """
        if beat_idx == 0:
            return 0.0
        prev_beat = all_beats[beat_idx - 1][0]
        gap = all_beats[beat_idx][0].start - prev_beat.end

        _, prev_seg_idx, _ = all_beats[beat_idx - 1]
        _, curr_seg_idx, _ = all_beats[beat_idx]

        # Segment boundary boost
        seg_boost = 0.20 if curr_seg_idx != prev_seg_idx else 0.0

        # Silence gap: 0.3s = moderate, 0.5s+ = strong
        pause = min(1.0, gap / 0.4) * 0.6

        # Previous beat was a sentence end: strong signal
        sent_boost = 0.20 if prev_beat.is_sentence_end else 0.0

        return min(1.0, pause + seg_boost + sent_boost)

    def _compute_duration_pressure(self, held_duration: float, adaptive_target: float) -> float:
        """Non-linear duration pressure: gentle before target, steep after.

        Uses a sigmoid-like curve so pressure builds gradually up to target
        and becomes very strong when held_duration significantly exceeds target.
        """
        if held_duration <= adaptive_target:
            # Below target: gentle linear ramp (0 to 0.3)
            ratio = held_duration / max(0.1, adaptive_target)
            return 0.3 * ratio

        # Above target: exponential pressure
        overshoot = held_duration - adaptive_target
        max_overshoot = self.max_shot_duration - adaptive_target
        if max_overshoot <= 0:
            return 1.0
        ratio = overshoot / max_overshoot
        # Exponential curve: slow start, fast finish
        return min(1.0, 0.3 + 0.7 * (ratio ** 1.5))

    def _compute_visual_freshness(self, beats_on_current: int) -> float:
        """Visual freshness: exponential decay per beat on current image.

        After ~5 beats on the same image, freshness drops below 0.2.
        """
        return max(0.0, math.exp(-0.40 * (beats_on_current - 1)))

    def _compute_relevance_decay(self, beats_on_current: int) -> float:
        """Relevance decay: current image's relevance to narration decreases over time.

        The image was relevant when first shown, but as narration evolves,
        its relevance naturally decays.  This encourages cuts when the
        narration has moved on from the visual concept.
        """
        if beats_on_current <= 1:
            return 1.0
        # Decay: 1.0 -> 0.95 -> 0.90 -> 0.85 -> ... over beats
        return max(0.5, 1.0 - 0.05 * (beats_on_current - 1))

    def _compute_cut_score(
        self,
        current_relevance: float,
        next_relevance: float,
        semantic_shift: float,
        pause_bonus: float,
        duration_pressure: float,
        visual_freshness: float,
    ) -> float:
        """Compute the raw cut-decision score.

        Weights are tuned for 2-4 second target pacing:
        - semantic_shift (0.20): narration moved to new topic
        - duration_pressure (0.22): held too long
        - next_relevance (0.18): next image is more relevant
        - current_low (0.15): current image no longer relevant
        - pause (0.15): natural pause = good cut moment
        - freshness (0.10): visual staleness
        """
        raw = (
            0.20 * semantic_shift
            + 0.15 * max(0.0, 1.0 - current_relevance)
            + 0.18 * next_relevance
            + 0.15 * pause_bonus
            + 0.22 * duration_pressure
            + 0.10 * max(0.0, 1.0 - visual_freshness)
        )
        return max(0.0, min(1.0, raw))

    def _evaluate_cut(
        self,
        cut_score: float,
        current_relevance: float,
        next_relevance: float,
        semantic_shift: float,
        pause_bonus: float,
        duration_pressure: float,
        visual_freshness: float,
        current_img: ImageAsset,
        next_img: ImageAsset,
        seg: TranscriptSegment,
        current_uses: int,
        held_duration: float,
        adaptive_target: float,
        llm_hint: float = 0.0,
        signal_quality: float = 1.0,
    ) -> tuple[str, str]:
        """Evaluate whether to CUT or HOLD, returning (decision, reason).

        The threshold is dynamically adjusted based on:
        - LLM director hint (lowers threshold when Gemini agrees on a cut)
        - Pause bonus (natural cut points)
        - Current image reuse count (encourage cutting away from reused images)
        - Next image relevance advantage
        - Duration pressure
        - Visual freshness
        - Capacity awareness (raises threshold if under adaptive target)
        - Signal quality (raise threshold if signals are mostly fallback)
        """
        threshold = self.cut_threshold

        # LLM director hint: lower threshold when Gemini agrees on a cut
        if llm_hint > 0.05:
            threshold -= 0.04

        # Pause signal: natural cut point
        if pause_bonus > 0.5:
            threshold -= 0.05
        elif pause_bonus > 0.3:
            threshold -= 0.03

        # Current image at reuse limit: strongly encourage cutting away
        if current_uses >= 2:
            threshold -= 0.10

        # Next image is significantly more relevant
        if next_relevance > current_relevance + 0.08:
            threshold -= 0.04

        # Duration pressure: already past target
        if duration_pressure > 0.3:
            threshold -= 0.05 * duration_pressure

        # Visual staleness
        if visual_freshness < 0.4:
            threshold -= 0.04

        # Capacity-aware: if held duration is under adaptive target, raise threshold
        if held_duration < adaptive_target * 0.7:
            threshold += 0.06
        elif held_duration < adaptive_target * 0.85:
            threshold += 0.03

        # Signal quality: if most signals are fallback, raise threshold
        # (Don't make aggressive cut decisions with degraded signals)
        if signal_quality < 0.5:
            threshold += 0.04
        elif signal_quality < 0.8:
            threshold += 0.02

        # Final threshold bounds
        threshold = max(0.15, min(0.50, threshold))

        if cut_score >= threshold:
            reasons = []
            if semantic_shift > 0.5:
                reasons.append(f"narration shifted (shift={semantic_shift:.2f})")
            if current_relevance < 0.4:
                reasons.append(f"current relevance dropped ({current_relevance:.2f})")
            if next_relevance > current_relevance + 0.05:
                reasons.append(f"next image more relevant ({next_relevance:.2f}>{current_relevance:.2f})")
            if pause_bonus > 0.3:
                reasons.append(f"pause detected ({pause_bonus:.2f})")
            if duration_pressure > 0.5:
                reasons.append(f"held too long ({held_duration:.1f}s>{adaptive_target:.1f}s)")
            if visual_freshness < 0.5:
                reasons.append(f"visual stale (freshness={visual_freshness:.2f})")
            if current_uses >= 2:
                reasons.append(f"reuse limit ({current_uses}/{self.max_image_uses})")
            if llm_hint > 0.05:
                reasons.append(f"LLM recommends cut (hint={llm_hint:.2f})")
            if not reasons:
                reasons.append(f"score ({cut_score:.2f}) >= threshold ({threshold:.2f})")
            return "CUT", "; ".join(reasons)
        else:
            reasons = []
            if current_relevance > next_relevance + 0.05:
                reasons.append(f"current more relevant ({current_relevance:.2f}>{next_relevance:.2f})")
            if semantic_shift < 0.35:
                reasons.append(f"same topic (shift={semantic_shift:.2f})")
            if next_relevance < 0.35:
                reasons.append(f"next low relevance ({next_relevance:.2f})")
            if held_duration < adaptive_target * 0.8:
                reasons.append(f"under target ({held_duration:.1f}s < {adaptive_target:.1f}s)")
            if not reasons:
                reasons.append(f"score ({cut_score:.2f}) < threshold ({threshold:.2f})")
            return "HOLD", "; ".join(reasons)

    @staticmethod
    def _reconcile_boundaries(scenes: list[Scene], audio_duration: float) -> None:
        if not scenes:
            return
        scenes[0].start = 0.0
        scenes[-1].end = audio_duration
        for i in range(1, len(scenes)):
            scenes[i].start = round(scenes[i - 1].end, 6)

    def _make_scene(
        self,
        existing: list[Scene],
        image: ImageAsset,
        start: float,
        end: float,
        narration: str,
        score: float,
        voice_image: float = 0.5,
        voice_prompt: float = 0.5,
        prompt_image: float = 0.5,
        decision: str = "HOLD",
        reason: str = "",
        usage_count: int = 0,
        beat_text: str = "",
        signal_quality: float = 1.0,
    ) -> Scene:
        scene_num = len(existing) + 1
        return Scene(
            scene_id=f"scene-{scene_num:03d}",
            image_id=image.id,
            start=round(start, 4),
            end=round(end, 4),
            narration=narration.strip(),
            confidence=round(min(1.0, score), 4),
            rerank_breakdown={
                "voice_image": round(voice_image, 4),
                "voice_prompt": round(voice_prompt, 4),
                "prompt_image": round(prompt_image, 4),
                "final_score": round(score, 4),
                "decision": decision,
                "cut_reason": reason,
                "usage_count": usage_count,
                "beat_text": beat_text[:200] if beat_text else "",
                "signal_quality": round(signal_quality, 4),
            },
        )
