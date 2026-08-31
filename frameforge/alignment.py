"""FrameForge V3 Three-Way Multimodal Alignment & Global Sequence Optimization Engine."""
from __future__ import annotations

import logging
import math
import re
from typing import Optional

from .config import DEFAULT_CONFIG, FrameForgeConfig
from .embeddings import EmbeddingEngine, VectorIndex, _tokenize
from .models import ImageAsset, Scene, TranscriptSegment
from .visual import VisualUnderstandingEngine

logger = logging.getLogger("frameforge.alignment")


class LocalSimilarity:
    """Backward-compatible similarity wrapper leveraging singleton EmbeddingEngine."""

    def __init__(self, config: Optional[FrameForgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.engine = EmbeddingEngine(self.config)
        self.method = "fallback" if self.engine.is_fallback else "semantic"

    def score_matrix(self, narration: list[str], prompts: list[str]) -> list[list[float]]:
        if not narration or not prompts:
            return []

        if not self.engine.is_fallback:
            try:
                import numpy as np
                narration_vecs = self.engine.encode_texts(narration, batch_size=self.config.embedding_batch_size)
                prompt_vecs = self.engine.encode_texts(prompts, batch_size=self.config.embedding_batch_size)
                if narration_vecs is not None and prompt_vecs is not None:
                    matrix = np.dot(narration_vecs, prompt_vecs.T)
                    return matrix.tolist()
            except Exception as exc:
                logger.warning("Vector score_matrix failed: %s. Using lexical fallback.", exc)

        return [[self.engine.lexical_similarity(a, b) for b in prompts] for a in narration]


class SceneAlignmentEngine:
    def __init__(
        self,
        minimum_scene_duration: float | None = None,
        preferred_min_duration: float | None = None,
        maximum_scene_duration: float | None = None,
        config: Optional[FrameForgeConfig] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.minimum_scene_duration = (
            minimum_scene_duration if minimum_scene_duration is not None else self.config.min_scene_duration
        )
        self.preferred_min_duration = (
            preferred_min_duration if preferred_min_duration is not None else self.config.preferred_scene_duration
        )
        self.maximum_scene_duration = (
            maximum_scene_duration if maximum_scene_duration is not None else self.config.max_scene_duration
        )
        self.visual_engine = VisualUnderstandingEngine(self.config)
        self.text_engine = EmbeddingEngine(self.config)

    def model_status(self) -> dict[str, Any]:
        """Return diagnostics about the active models and scoring signals."""
        from typing import Any
        vis = self.visual_engine.model_status() if hasattr(self.visual_engine, "model_status") else {}
        txt_fallback = self.text_engine.is_fallback if hasattr(self.text_engine, "is_fallback") else True
        signals = []
        if vis.get("has_clip"):
            signals.append("voice_image")
        if not txt_fallback:
            signals.append("voice_prompt")
        signals.append("prompt_image")
        signals.append("entity")
        signals.append("quality")
        return {
            "visual_engine": vis,
            "text_engine_fallback": txt_fallback,
            "active_scoring_signals": signals,
            "alignment_method": (
                "three_way_multimodal" if vis.get("has_clip")
                else ("semantic" if not txt_fallback else "fallback")
            ),
        }

    def extract_salient_tokens(self, text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text)
        salient = set()
        for w in words:
            if len(w) > 3 or (w[0].isupper() and len(w) > 1):
                salient.add(w.lower())
        return salient

    def align(
        self,
        segments: list[TranscriptSegment],
        images: list[ImageAsset],
        audio_duration: float,
    ) -> tuple[list[Scene], str]:
        if not images:
            raise ValueError("At least one image is required.")
        if len(images) == 1:
            full_text = " ".join(s.text for s in segments)
            single_method = "three_way_multimodal" if self.visual_engine.has_clip else ("semantic" if not self.text_engine.is_fallback else "fallback")
            return [
                Scene(
                    scene_id="scene-001",
                    image_id=images[0].id,
                    start=0.0,
                    end=audio_duration,
                    narration=full_text,
                    confidence=1.0,
                    rerank_breakdown={
                        "voice_image": 0.5,
                        "voice_prompt": 0.5,
                        "prompt_image": 0.5,
                        "quality": float(images[0].quality_score),
                        "final_score": 1.0,
                    },
                )
            ], single_method

        # Ensure all images have fallback prompts/tags if missing
        from .image_processor import ImagePreprocessor
        for img in images:
            if not img.prompt.strip():
                img.prompt = ImagePreprocessor.extract_fallback_prompt(img.filename, img.metadata.get("visual_tags", []))
                img.prompt_source = "auto_generated"

        prompts = [img.prompt for img in images]
        narration_texts = [s.text for s in segments]
        num_segments = len(segments)
        num_images = len(images)
        top_k = min(self.config.top_k_candidates, num_images)

        logger.info("Starting Three-Way Multimodal Alignment (%d segments, %d images, top_k=%d)...", num_segments, num_images, top_k)

        # -------------------------------------------------------------
        # Stage 1: Multimodal 3-Way Signal Calculation
        # -------------------------------------------------------------
        # 1. Voice <-> Image similarity (CLIP Joint Space)
        voice_image_matrix = None
        if self.visual_engine.has_clip:
            try:
                import numpy as np
                img_vis_vecs = self.visual_engine.encode_images(images)
                narr_vis_vecs = self.visual_engine.encode_text(narration_texts)
                if img_vis_vecs is not None and narr_vis_vecs is not None:
                    raw_vi = np.dot(narr_vis_vecs, img_vis_vecs.T)
                    voice_image_matrix = (np.clip(raw_vi, -1.0, 1.0) + 1.0) / 2.0
            except Exception as exc:
                logger.warning("Visual CLIP similarity calculation failed: %s", exc)

        # 2. Voice <-> Image Prompt similarity (SentenceTransformer / Lexical)
        voice_prompt_matrix = None
        if not self.text_engine.is_fallback:
            try:
                import numpy as np
                prompt_vecs = self.text_engine.encode_texts(prompts, batch_size=self.config.embedding_batch_size)
                narr_vecs = self.text_engine.encode_texts(narration_texts, batch_size=self.config.embedding_batch_size)
                if prompt_vecs is not None and narr_vecs is not None:
                    raw_vp = np.dot(narr_vecs, prompt_vecs.T)
                    voice_prompt_matrix = (np.clip(raw_vp, -1.0, 1.0) + 1.0) / 2.0
            except Exception as exc:
                logger.warning("Text embedding similarity calculation failed: %s", exc)

        # 3. Prompt <-> Image Consistency
        prompt_image_consistency = self.visual_engine.compute_prompt_image_consistency(images)

        # Build candidate Top-K lattice for each segment
        lattice: list[list[dict]] = []
        for t, seg in enumerate(segments):
            salient_narr = self.extract_salient_tokens(seg.text)
            candidates_scores = []

            for j, img in enumerate(images):
                # 1. Voice <-> Image score
                s_vi = float(voice_image_matrix[t, j]) if voice_image_matrix is not None else 0.5

                # 2. Voice <-> Prompt score
                if voice_prompt_matrix is not None:
                    s_vp = float(voice_prompt_matrix[t, j])
                else:
                    lex = self.text_engine.lexical_similarity(seg.text, img.prompt)
                    s_vp = max(0.0, min(1.0, lex))

                # 3. Prompt <-> Image score
                s_pi = float(prompt_image_consistency[j])

                # 4. Entity overlap score
                s_ent = 0.0
                if salient_narr:
                    img_tokens = self.extract_salient_tokens(img.prompt + " " + " ".join(img.metadata.get("visual_tags", [])))
                    if img_tokens:
                        s_ent = min(1.0, len(salient_narr & img_tokens) / max(1, len(salient_narr)))

                # 5. Image quality score
                s_qual = float(img.quality_score)

                # Normalized Three-Way Linear Combination
                if voice_image_matrix is not None:
                    node_score = (
                        self.config.w_voice_image * s_vi
                        + self.config.w_voice_prompt * s_vp
                        + self.config.w_prompt_image * s_pi
                        + self.config.w_entity * s_ent
                        + self.config.w_quality * (s_qual - 0.5)
                    )
                else:
                    combined_w = self.config.w_voice_image + self.config.w_voice_prompt
                    node_score = (
                        combined_w * s_vp
                        + self.config.w_prompt_image * s_pi
                        + self.config.w_entity * s_ent
                        + self.config.w_quality * (s_qual - 0.5)
                    )

                node_score = max(0.01, min(1.0, node_score))

                candidates_scores.append(
                    {
                        "img_idx": j,
                        "node_score": node_score,
                        "voice_image": round(s_vi, 4),
                        "voice_prompt": round(s_vp, 4),
                        "prompt_image": round(s_pi, 4),
                        "entity_sim": round(s_ent, 4),
                        "quality": round(s_qual, 4),
                    }
                )

            # Keep Top-K candidates for this segment
            candidates_scores.sort(key=lambda x: x["node_score"], reverse=True)
            lattice.append(candidates_scores[:top_k])

            # Diagnostic logging for top candidate
            top_cand = candidates_scores[0]
            logger.info(
                "SEGMENT %d [%.2fs - %.2fs]: '%s' -> Top Candidate: %s (Score: %.4f | VI: %.4f | VP: %.4f | PI: %.4f)",
                t + 1,
                seg.start,
                seg.end,
                seg.text[:50],
                images[top_cand["img_idx"]].filename,
                top_cand["node_score"],
                top_cand["voice_image"],
                top_cand["voice_prompt"],
                top_cand["prompt_image"],
            )

        # -------------------------------------------------------------
        # Stage 2: Global Sequence Optimization (Viterbi Lattice Search)
        # -------------------------------------------------------------
        dp: list[list[float]] = [[0.0] * len(lattice[t]) for t in range(num_segments)]
        backtrack: list[list[int]] = [[-1] * len(lattice[t]) for t in range(num_segments)]

        # Initialize t = 0
        for k, cand in enumerate(lattice[0]):
            dp[0][k] = cand["node_score"]

        # Forward Viterbi DP with continuity rewards, recency penalties,
        # duplicate penalties, and topic-shift awareness.
        # recent_history[t][k] = set of image indices used in the last
        # ``recent_history_window`` segments along the partial path ending
        # at lattice state (t, k).
        recent_history: list[list[set[int]]] = [
            [set() for _ in lattice[t]] for t in range(num_segments)
        ]

        for t in range(1, num_segments):
            for k, curr_cand in enumerate(lattice[t]):
                curr_img_idx = curr_cand["img_idx"]
                curr_img = images[curr_img_idx]
                best_prev_val = -1e9
                best_prev_idx = -1

                # Detect topic shift: low word-overlap between consecutive segments
                is_topic_shift = False
                if t > 0:
                    prev_tokens = set(segments[t - 1].text.lower().split())
                    curr_tokens = set(segments[t].text.lower().split())
                    if prev_tokens and curr_tokens:
                        overlap = len(prev_tokens & curr_tokens) / max(
                            len(prev_tokens), len(curr_tokens)
                        )
                        is_topic_shift = overlap < self.config.dp_topic_shift_threshold

                for p, prev_cand in enumerate(lattice[t - 1]):
                    prev_img_idx = prev_cand["img_idx"]
                    trans_reward = 0.0

                    # 1. Continuity reward: same image as previous segment
                    if curr_img_idx == prev_img_idx:
                        if is_topic_shift:
                            # Topic shifted — same image is undesirable
                            trans_reward -= self.config.dp_transition_weight
                        else:
                            trans_reward += self.config.continuity_boost
                    else:
                        # 2. Duplicate penalty: switching to a near-duplicate image
                        if curr_img.is_duplicate and curr_img.duplicate_of == images[prev_img_idx].id:
                            trans_reward -= self.config.duplicate_penalty

                        # 3. Recency penalty: penalize images used within the
                        #    recent history window along this partial path
                        if curr_img_idx in recent_history[t - 1][p]:
                            trans_reward -= self.config.recent_use_penalty

                    candidate_val = dp[t - 1][p] + curr_cand["node_score"] + trans_reward
                    if candidate_val > best_prev_val:
                        best_prev_val = candidate_val
                        best_prev_idx = p

                dp[t][k] = best_prev_val
                backtrack[t][k] = best_prev_idx

                # Propagate recent usage set along the best incoming path
                if best_prev_idx >= 0:
                    new_history = set(recent_history[t - 1][best_prev_idx])
                    new_history.add(curr_img_idx)
                    # Trim to window size — drop oldest entries when full
                    if len(new_history) > self.config.recent_history_window:
                        # Remove the image that entered earliest (smallest segment index)
                        # We approximate by dropping one arbitrary entry when overfull
                        # (exact tracking would require a deque, but this is sufficient
                        # for penalty purposes)
                        new_history.pop()
                    recent_history[t][k] = new_history
                else:
                    recent_history[t][k] = {curr_img_idx}

        # Backtrack optimal global path
        best_end_k = max(range(len(lattice[-1])), key=lambda k: dp[-1][k])
        optimal_path: list[int] = [0] * num_segments
        curr_k = best_end_k

        for t in range(num_segments - 1, -1, -1):
            optimal_path[t] = lattice[t][curr_k]["img_idx"]
            curr_k = backtrack[t][curr_k]

        # -------------------------------------------------------------
        # Stage 3: Segment-Preserving Scene Construction
        # -------------------------------------------------------------
        # Each narration segment maps to its own scene on the timeline,
        # preserving temporal scene cuts and narration boundaries.
        scenes: list[Scene] = []
        for t, img_idx in enumerate(optimal_path):
            cand_info = next((c for c in lattice[t] if c["img_idx"] == img_idx), lattice[t][0])
            score = cand_info["node_score"]
            breakdown = {
                "voice_image": cand_info["voice_image"],
                "voice_prompt": cand_info["voice_prompt"],
                "prompt_image": cand_info["prompt_image"],
                "entity_sim": cand_info["entity_sim"],
                "quality": cand_info["quality"],
                "final_score": round(score, 4),
            }

            seg = segments[t]
            start_t = 0.0 if t == 0 else seg.start
            end_t = audio_duration if t == num_segments - 1 else seg.end

            scene = Scene(
                scene_id=f"scene-{t + 1:03d}",
                image_id=images[img_idx].id,
                start=start_t,
                end=end_t,
                narration=seg.text,
                confidence=round(score, 4),
                rerank_breakdown=breakdown,
            )
            scenes.append(scene)
            logger.info(
                "SCENE CREATED: %s [%.2fs - %.2fs (%.2fs)] -> Asset: %s | Text: '%s'",
                scene.scene_id,
                scene.start,
                scene.end,
                scene.duration,
                images[img_idx].filename,
                seg.text[:40],
            )

        # Reconcile boundary micro-gaps and enforce audio coverage
        reconciled = self._reconcile_boundaries(scenes, audio_duration)
        merged = self._merge_short(reconciled)

        logger.info("Timeline generated: %d distinct scenes from %d narration segments.", len(merged), num_segments)
        alignment_method = "three_way_multimodal" if self.visual_engine.has_clip else ("semantic" if not self.text_engine.is_fallback else "fallback")
        return merged, alignment_method

    @staticmethod
    def _reconcile_boundaries(scenes: list[Scene], audio_duration: float) -> list[Scene]:
        if not scenes:
            return scenes
        epsilon = 1e-6
        for scene in scenes:
            scene.start = max(0.0, min(float(scene.start), audio_duration))
            scene.end = max(0.0, min(float(scene.end), audio_duration))
        scenes[0].start = 0.0
        scenes[-1].end = audio_duration

        index = 0
        while index < len(scenes) - 1:
            left, right = scenes[index], scenes[index + 1]
            lower = left.start + epsilon
            upper = right.end - epsilon
            preferred = right.start
            if upper <= lower:
                logger.info("SCENE MERGED: %s and %s merged due to zero/negative duration span.", left.scene_id, right.scene_id)
                left.end = max(left.end, right.end)
                left.narration = (left.narration + " " + right.narration).strip()
                left.confidence = round((left.confidence + right.confidence) / 2, 4)
                scenes.pop(index + 1)
                continue
            if preferred < lower or preferred > upper:
                preferred = (left.end + right.start) / 2
            boundary = max(lower, min(preferred, upper))
            left.end = boundary
            right.start = boundary
            index += 1

        scenes[-1].end = audio_duration
        return scenes

    def sequential_align(
        self,
        segments: list[TranscriptSegment],
        images: list[ImageAsset],
        audio_duration: float,
    ) -> tuple[list[Scene], str, dict]:
        """Sequential image-order-preserving alignment.

        Computes the same CLIP multimodal signals as align() but uses
        SequentialEditor for cut/hold decisions instead of Viterbi global
        reordering.  Image order is AUTHORITATIVE.

        Returns: (scenes, method_name, editor_metadata)
        """
        if not images:
            raise ValueError("At least one image is required.")
        if len(images) == 1:
            full_text = " ".join(s.text for s in segments)
            single_method = "three_way_multimodal" if self.visual_engine.has_clip else (
                "semantic" if not self.text_engine.is_fallback else "fallback"
            )
            scene = Scene(
                scene_id="scene-001",
                image_id=images[0].id,
                start=0.0,
                end=audio_duration,
                narration=full_text,
                confidence=1.0,
                rerank_breakdown={
                    "voice_image": 0.5, "voice_prompt": 0.5, "prompt_image": 0.5,
                    "quality": float(images[0].quality_score), "final_score": 1.0,
                    "decision": "HOLD", "cut_reason": "Single image",
                },
            )
            return [scene], single_method, {"images_used": 1, "num_cuts": 0}

        # Ensure fallback prompts
        from .image_processor import ImagePreprocessor
        for img in images:
            if not img.prompt.strip():
                img.prompt = ImagePreprocessor.extract_fallback_prompt(
                    img.filename, img.metadata.get("visual_tags", [])
                )
                img.prompt_source = "auto_generated"

        prompts = [img.prompt for img in images]
        narration_texts = [s.text for s in segments]
        num_images = len(images)

        logger.info(
            "Starting Sequential Alignment (%d segments, %d images)...",
            len(segments), num_images,
        )

        # --- Compute the same multimodal signals as global alignment ---
        import numpy as np

        voice_image_matrix = None
        if self.visual_engine.has_clip:
            try:
                img_vecs = self.visual_engine.encode_images(images)
                narr_vecs = self.visual_engine.encode_text(narration_texts)
                if img_vecs is not None and narr_vecs is not None:
                    raw_vi = np.dot(narr_vecs, img_vecs.T)
                    voice_image_matrix = (np.clip(raw_vi, -1.0, 1.0) + 1.0) / 2.0
            except Exception as exc:
                logger.warning("Visual CLIP similarity failed in sequential_align: %s", exc)

        voice_prompt_matrix = None
        if not self.text_engine.is_fallback:
            try:
                prompt_vecs = self.text_engine.encode_texts(prompts, batch_size=self.config.embedding_batch_size)
                narr_vecs = self.text_engine.encode_texts(narration_texts, batch_size=self.config.embedding_batch_size)
                if prompt_vecs is not None and narr_vecs is not None:
                    raw_vp = np.dot(narr_vecs, prompt_vecs.T)
                    voice_prompt_matrix = (np.clip(raw_vp, -1.0, 1.0) + 1.0) / 2.0
            except Exception as exc:
                logger.warning("Text embedding similarity failed in sequential_align: %s", exc)

        prompt_image_scores = self.visual_engine.compute_prompt_image_consistency(images)

        # --- V5: Gemini story/beat planning (optional) ---
        llm_plan = None
        llm_info = {}
        if self.config.llm_enabled:
            try:
                from .llm import GeminiBeatPlanner
                planner = GeminiBeatPlanner(self.config)
                if planner.is_available:
                    llm_plan = planner.plan(segments, images, audio_duration)
                    if llm_plan:
                        llm_info = {
                            "model": llm_plan.get("model", "unknown"),
                            "latency_ms": llm_plan.get("latency_ms", 0),
                            "from_cache": llm_plan.get("from_cache", False),
                            "num_beats": len(llm_plan.get("beats", [])),
                        }
                        logger.info("Gemini plan: %d beats, %.0fms, cache=%s",
                                    llm_info["num_beats"], llm_info["latency_ms"],
                                    llm_info["from_cache"])
            except Exception as exc:
                logger.warning("Gemini planning failed, falling back to deterministic: %s", exc)

        # --- Delegate to SequentialEditor ---
        from .editor import SequentialEditor
        editor = SequentialEditor(config=self.config)
        scenes, metadata = editor.plan(
            segments, images, audio_duration,
            voice_image_matrix=voice_image_matrix,
            voice_prompt_matrix=voice_prompt_matrix,
            prompt_image_scores=prompt_image_scores,
            llm_plan=llm_plan,
        )

        method = (
            "sequential_multimodal" if self.visual_engine.has_clip
            else ("sequential_semantic" if not self.text_engine.is_fallback
                  else "sequential_fallback")
        )
        if llm_info:
            method += "+llm"
            metadata["llm_info"] = llm_info
        logger.info(
            "Sequential timeline generated: %d scenes from %d segments (%d cuts, %d images used, llm=%s).",
            len(scenes), len(segments), metadata.get("num_cuts", 0),
            metadata.get("images_used", 0), bool(llm_info),
        )
        return scenes, method, metadata

    def _merge_short(self, scenes: list[Scene]) -> list[Scene]:
        index = 0
        while index < len(scenes):
            if len(scenes) > 1 and scenes[index].duration < self.minimum_scene_duration:
                target = index - 1 if index > 0 else 1
                left, right = sorted((index, target))
                keeper, removed = scenes[left], scenes[right]
                logger.info(
                    "SCENE MERGED: %s (%.2fs) merged into %s because duration < %.2fs threshold.",
                    removed.scene_id,
                    removed.duration,
                    keeper.scene_id,
                    self.minimum_scene_duration,
                )
                keeper.start, keeper.end = min(keeper.start, removed.start), max(keeper.end, removed.end)
                keeper.narration = (keeper.narration + " " + removed.narration).strip()
                scenes.pop(right)
                index = max(0, left - 1)
            else:
                index += 1

        for n, scene in enumerate(scenes, 1):
            scene.scene_id = f"scene-{n:03d}"
        return scenes
