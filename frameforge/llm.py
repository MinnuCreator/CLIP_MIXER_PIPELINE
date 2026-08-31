"""FrameForge V6 Gemini 2.5 Flash Story & Visual-Beat Planner.

Gemini acts as the high-level STORY / VISUAL-BEAT DIRECTOR.  It receives
a LOCAL SEQUENTIAL WINDOW of narration + nearby image prompts, and outputs
structured beat recommendations with timing, decision, reason, and confidence.

The SequentialEditor then validates and enforces the plan, never allowing
it to reorder images or violate hard constraints.

V6 changes from V5:
- Windowed context: only sends current + next 1-3 prompts, not all 700
- New schema: beats contain decision/hold timing instead of image_index/action
- Confidence-weighted hints
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_CONFIG, FrameForgeConfig
from .models import ImageAsset, TranscriptSegment

logger = logging.getLogger("frameforge.llm")

# ── Structured output schema ────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a professional video editor and story analyst.  Given a voiceover \
transcript with timestamps and a local window of the upcoming ordered image \
prompts, you determine the best moments to advance to the next visual.

IMPORTANT RULES:
1. You are a TIMING ADVISOR, not an image selector.
2. You NEVER choose which image to use — the images are in a fixed, \
authoritative order.  You only decide WHEN to advance.
3. Each beat represents one visual moment — either HOLD the current image \
or ADVANCE to the next image in sequence.
4. The "decision" field is either "HOLD" (keep current image) or "ADVANCE" \
(move to next image in the ordered sequence).
5. Place cuts at natural sentence/clause boundaries, pauses, or semantic \
transitions in the narration.
6. Aim for 2-4 second shots when the narration supports it, but adapt to \
the actual content.  Allow longer holds when the narration clearly \
continues the same visual idea.
7. Output MUST be valid JSON matching the requested schema.
8. Start time of first beat must be 0.0.
9. End_hint values must increase monotonically and never exceed the total \
narration duration.

OUTPUT FORMAT — a JSON object with a "beats" array:
{
  "beats": [
    {
      "start": 0.0,
      "end_hint": 2.4,
      "decision": "HOLD",
      "reason": "Narration continues describing the current visual",
      "confidence": 0.92
    },
    {
      "start": 2.4,
      "end_hint": 4.8,
      "decision": "ADVANCE",
      "reason": "Narration changes to the next visual concept",
      "confidence": 0.95
    }
  ]
}

Each beat object has:
- start: when this visual moment begins (seconds)
- end_hint: suggested end of this visual moment (seconds)
- decision: "HOLD" (keep showing current image) or "ADVANCE" (cut to next)
- reason: one short sentence explaining the decision
- confidence: 0.0 to 1.0, how confident you are in this decision

Place cuts at meaningful transitions — topic changes, sentence endings, \
pauses, clause boundaries.  Do not cut unnecessarily.  \
Allow the narration to breathe.
"""


class GeminiBeatPlanner:
    """Calls Gemini 2.5 Flash to produce structured beat recommendations.

    V6: Uses WINDOWED context — only sends current + next few image prompts,
    not the entire image library.  This keeps token usage reasonable even
    with 700 images.

    If the API key is missing or the call fails, returns ``None`` so the
    system can fall back to the deterministic SequentialEditor.
    """

    def __init__(self, config: Optional[FrameForgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.api_key = self.config.llm_api_key
        self.model = self.config.llm_model
        self.cache_dir = Path(self.config.cache_dir) / "v6" / "llm_plans"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.model)

    def _make_cache_key(
        self,
        audio_hash: str,
        image_ids: list[str],
        prompts: list[str],
    ) -> str:
        raw = (
            f"{audio_hash}_"
            f"{'|'.join(image_ids)}_"
            f"{'|'.join(prompts)}_"
            f"{self.model}_"
            f"{self.config.config_version}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _audio_hash(self, segments: list[TranscriptSegment]) -> str:
        text = "".join(s.text for s in segments)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # ── Public API ──────────────────────────────────────────────────────────

    def plan(
        self,
        segments: list[TranscriptSegment],
        images: list[ImageAsset],
        audio_duration: float,
    ) -> Optional[dict[str, Any]]:
        """Request a beat plan from Gemini. Returns validated dict or None.

        Returns:
            {
                "beats": [{"start": float, "end_hint": float, "decision": str,
                           "reason": str, "confidence": float}, ...],
                "model": str,
                "latency_ms": float,
                "from_cache": bool,
            }
        """
        if not self.is_available:
            logger.info("Gemini not configured — skipping LLM planning.")
            return None

        # Build user prompt with windowed context
        user_prompt = self._build_user_prompt(segments, images, audio_duration)

        # Check cache
        audio_hash = self._audio_hash(segments)
        image_ids = [img.id for img in images]
        prompts = [img.prompt for img in images]
        cache_key = self._make_cache_key(audio_hash, image_ids, prompts)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                cached["from_cache"] = True
                logger.info("Gemini plan loaded from cache.")
                return cached
            except Exception as exc:
                logger.warning("Failed to read cached Gemini plan: %s", exc)

        # Call Gemini
        import time
        t0 = time.monotonic()
        raw_text = self._call_gemini(user_prompt)
        latency_ms = (time.monotonic() - t0) * 1000

        if raw_text is None:
            return None

        # Parse and validate
        result = self._parse_and_validate(raw_text, audio_duration)
        if result is None:
            return None

        result["model"] = self.model
        result["latency_ms"] = round(latency_ms, 1)
        result["from_cache"] = False

        # Persist cache
        try:
            cache_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to cache Gemini plan: %s", exc)

        return result

    # ── Prompt construction (windowed context) ──────────────────────────────

    def _build_user_prompt(
        self,
        segments: list[TranscriptSegment],
        images: list[ImageAsset],
        audio_duration: float,
    ) -> str:
        """Build prompt using windowed context.

        Instead of sending ALL 700 images, send only:
        - Full narration transcript
        - Current + next 3 image prompts (the local sequential window)
        - Total image count for context

        This keeps Gemini fast and inexpensive.
        """
        lines: list[str] = []
        lines.append(f"Narration duration: {audio_duration:.2f} seconds")
        lines.append(f"Total images in storyboard: {len(images)}")
        lines.append("")

        # Full narration transcript
        lines.append("VOICEOVER TRANSCRIPT (with timestamps):")
        for i, seg in enumerate(segments, 1):
            words_hint = ""
            if seg.words:
                words_hint = f"  [words: {seg.words[0].start:.2f}-{seg.words[-1].end:.2f}]"
            lines.append(f"  [{seg.start:.2f}s - {seg.end:.2f}s] Segment {i}: {seg.text}{words_hint}")
        lines.append("")

        # Windowed image prompts: current + next 3
        lines.append("IMAGE STORYBOARD SEQUENCE (order is AUTHORITATIVE):")
        lines.append(f"  (Showing first 5 images of {len(images)} total)")
        n_show = min(5, len(images))
        for i in range(n_show):
            img = images[i]
            prompt_text = img.prompt[:150] if img.prompt else "(no prompt)"
            lines.append(f"  Image {i + 1} (current → next): {prompt_text}")
        if len(images) > n_show:
            lines.append(f"  ... {len(images) - n_show} more images follow in sequence")
        lines.append("")

        lines.append(
            f"TASK: For the FULL narration, produce a JSON object with a \"beats\" array.  "
            f"For each moment decide HOLD (keep current image) or ADVANCE (cut to next image).  "
            f"Never reorder.  Start at 0.0, end at {audio_duration:.2f}."
        )
        return "\n".join(lines)

    # ── Gemini API call via AICredits (OpenAI-compatible) ────────────────────

    def _call_gemini(self, user_prompt: str) -> Optional[str]:
        """Call Gemini via AICredits OpenAI-compatible endpoint.

        Uses:
          base_url from config (default: https://api.aicredits.in/v1)
          model    = google/gemini-2.5-flash
          auth     = Bearer AICREDITS_API_KEY
        """
        base_url = self.config.llm_base_url

        for attempt in range(self.config.llm_max_retries + 1):
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=8192,
                    temperature=0.3,
                )
                text = response.choices[0].message.content
                if text:
                    return text.strip()
            except ImportError:
                logger.warning(
                    "openai package not installed.  "
                    "Install with: pip install openai"
                )
                return None
            except Exception as exc:
                logger.warning(
                    "AICredits Gemini call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.llm_max_retries + 1,
                    exc,
                )
                if attempt < self.config.llm_max_retries:
                    import time
                    time.sleep(1.0 * (attempt + 1))

        return None

    # ── Parsing & validation ────────────────────────────────────────────────

    def _parse_and_validate(
        self,
        raw_text: str,
        audio_duration: float,
    ) -> Optional[dict[str, Any]]:
        """Parse raw LLM text into a validated beat plan, or None.

        V6 schema: beats contain start/end_hint/decision/reason/confidence.
        No image_index or action fields — the editor handles image selection.
        """
        # Strip markdown code blocks
        raw_text = re.sub(r"```json\s*", "", raw_text)
        raw_text = re.sub(r"```\s*", "", raw_text)
        raw_text = raw_text.strip()

        # Extract JSON — may be a top-level object {"beats": [...]} or just [...]
        # Try object first
        obj_match = re.search(r"\{[\s\S]*\}", raw_text)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
                if isinstance(parsed, dict) and "beats" in parsed:
                    beats = parsed["beats"]
                else:
                    beats = [parsed]
            except json.JSONDecodeError:
                beats = None
        else:
            beats = None

        # Try array
        if beats is None:
            arr_match = re.search(r"\[[\s\S]*\]", raw_text)
            if arr_match:
                try:
                    beats = json.loads(arr_match.group())
                except json.JSONDecodeError as exc:
                    logger.warning("Gemini response JSON parse error: %s", exc)
                    return None

        if not isinstance(beats, list) or len(beats) == 0:
            logger.warning("Gemini returned empty or invalid beats.")
            return None

        validated: list[dict[str, Any]] = []

        for i, beat in enumerate(beats):
            if not isinstance(beat, dict):
                continue

            # Extract fields — V6 schema (flexible field names)
            # Gemini may use: time/start/start_time, action/decision, end_hint/end_time
            start = float(beat.get("start", beat.get("start_time", beat.get("time", 0))))
            end_hint = float(beat.get("end_hint", beat.get("end_time", start + 2.0)))
            decision = str(beat.get("decision", beat.get("action", "HOLD"))).upper()
            reason = str(beat.get("reason", beat.get("description", "")))[:200]
            confidence = float(beat.get("confidence", 0.8))

            # Validate decision
            if decision not in ("HOLD", "ADVANCE", "CUT"):
                decision = "HOLD"
            # Normalize CUT -> ADVANCE
            if decision == "CUT":
                decision = "ADVANCE"

            # Validate timestamps
            if start < 0:
                start = 0.0
            if end_hint > audio_duration:
                end_hint = audio_duration
            if end_hint <= start:
                end_hint = min(start + 1.0, audio_duration)
            if confidence < 0.0 or confidence > 1.0:
                confidence = 0.8

            validated.append({
                "start": round(start, 4),
                "end_hint": round(end_hint, 4),
                "decision": decision,
                "reason": reason,
                "confidence": round(confidence, 4),
            })

        if not validated:
            return None

        # Ensure first beat starts at 0.0
        validated[0]["start"] = 0.0

        # Ensure timestamps are monotonically increasing
        for i in range(1, len(validated)):
            if validated[i]["start"] < validated[i - 1]["end_hint"] - 0.01:
                validated[i]["start"] = validated[i - 1]["end_hint"]

        # Ensure last beat ends at audio_duration
        validated[-1]["end_hint"] = audio_duration

        return {"beats": validated}
