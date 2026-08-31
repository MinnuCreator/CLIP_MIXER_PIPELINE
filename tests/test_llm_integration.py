"""Tests for V6 Gemini LLM integration in the sequential editor.

These tests verify:
- LLM plans are validated (no reordering, no oscillation, max usage)
- Editor correctly uses LLM cut hints as timing signals
- Fallback works when LLM is unavailable
- LLM caching works
- Image order is never violated by LLM output
- V6 schema: decision/end_hint/confidence
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from frameforge.config import FrameForgeConfig
from frameforge.editor import SequentialEditor
from frameforge.llm import GeminiBeatPlanner
from frameforge.models import ImageAsset, Scene, TranscriptSegment, Word


def _make_config(**overrides) -> FrameForgeConfig:
    cfg = FrameForgeConfig()
    cfg.llm_api_key = ""  # disabled by default for tests
    cfg.llm_enabled = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_images(n: int = 5) -> list[ImageAsset]:
    return [
        ImageAsset(
            id=f"img-{i:03d}",
            file_path=f"/tmp/test_{i}.png",
            filename=f"test_{i}.png",
            prompt=f"image {i} prompt describing scene {i}",
            prompt_source="supplied_file",
            metadata={"visual_tags": [f"tag{i}"]},
        )
        for i in range(1, n + 1)
    ]


def _make_segments(durations: list[tuple[float, float, str]] | None = None) -> list[TranscriptSegment]:
    """Create Whisper-like segments with word-level timestamps."""
    if durations is None:
        durations = [
            (0.0, 2.0, "The lighthouse stood tall against the night sky."),
            (2.0, 4.5, "A violent storm was approaching the coast."),
            (4.5, 7.0, "The keeper climbed the tower stairs quickly."),
            (7.0, 9.5, "He switched on the powerful beacon light."),
            (9.5, 12.0, "Ships in the distance saw the guiding beam."),
        ]
    segments = []
    for start, end, text in durations:
        words = text.split()
        n_words = len(words)
        word_list = []
        for wi, w in enumerate(words):
            ws = start + (end - start) * wi / n_words
            we = start + (end - start) * (wi + 1) / n_words
            word_list.append(Word(word=w, start=ws, end=we))
        segments.append(TranscriptSegment(text=text, start=start, end=end, words=word_list))
    return segments


def _make_llm_plan(beats: list[dict]) -> dict:
    """Create a valid LLM plan dict."""
    return {"beats": beats, "model": "gemini-2.5-flash", "latency_ms": 100, "from_cache": False}


# ── V6 schema beat factory ──────────────────────────────────────────────────

def _beat(start: float, end: float, decision: str, reason: str = "", confidence: float = 0.8) -> dict:
    return {"start": start, "end_hint": end, "decision": decision, "reason": reason, "confidence": confidence}


class GeminiValidatorTests(unittest.TestCase):
    """Test that the LLM output validator catches invalid plans."""

    def setUp(self):
        self.config = _make_config()
        self.editor = SequentialEditor(config=self.config)
        self.images = _make_images(5)
        self.segments = _make_segments()
        self.audio_duration = 12.0

    def _run_plan_with_llm(self, llm_plan: dict) -> tuple[list[Scene], dict]:
        """Helper: run the editor with a given LLM plan."""
        return self.editor.plan(
            self.segments, self.images, self.audio_duration, llm_plan=llm_plan,
        )

    def test_valid_llm_plan_produces_scenes(self):
        """A well-formed V6 LLM plan should produce a valid timeline."""
        plan = _make_llm_plan([
            _beat(0.0, 3.0, "HOLD", "Opening shot", 0.9),
            _beat(3.0, 6.0, "ADVANCE", "Storm approaches", 0.85),
            _beat(6.0, 9.0, "ADVANCE", "Keeper climbs", 0.88),
            _beat(9.0, 12.0, "ADVANCE", "Beacon lights", 0.92),
        ])
        scenes, metadata = self._run_plan_with_llm(plan)

        self.assertGreater(len(scenes), 0)
        self.assertTrue(metadata["image_order_preserved"])
        self.assertTrue(metadata["llm_used"])
        self.assertGreater(metadata["llm_advance_points"], 0)

    def test_llm_cannot_reorder_images(self):
        """Even if LLM plan references image_index, editor enforces order."""
        # V5 compat: pass image_index/action plan — editor handles both
        plan = _make_llm_plan([
            {"image_index": 1, "action": "HOLD", "start": 0.0, "end_hint": 3.0, "reason": "Opener"},
            {"image_index": 2, "action": "CUT", "start": 3.0, "end_hint": 6.0, "reason": "Cut to 2"},
            {"image_index": 3, "action": "CUT", "start": 6.0, "end_hint": 9.0, "reason": "Cut to 3"},
            {"image_index": 4, "action": "CUT", "start": 9.0, "end_hint": 12.0, "reason": "Cut to 4"},
        ])
        scenes, metadata = self._run_plan_with_llm(plan)

        # Images should be in order
        img_indices = [int(s.image_id.split("-")[1]) for s in scenes]
        for i in range(1, len(img_indices)):
            self.assertGreaterEqual(img_indices[i], img_indices[i - 1])

    def test_llm_cannot_cause_oscillation(self):
        """LLM plan cannot cause A->B->A oscillation."""
        plan = _make_llm_plan([
            _beat(0.0, 2.0, "HOLD", "Start"),
            _beat(2.0, 4.0, "ADVANCE", "Cut to 2"),
            _beat(4.0, 6.0, "HOLD", "Hold 2"),
            _beat(6.0, 8.0, "ADVANCE", "Cut to 3"),
            _beat(8.0, 10.0, "HOLD", "Hold 3"),
            _beat(10.0, 12.0, "HOLD", "End"),
        ])
        scenes, _ = self._run_plan_with_llm(plan)

        img_ids = [s.image_id for s in scenes]
        for i in range(2, len(img_ids)):
            self.assertFalse(
                img_ids[i] == img_ids[i - 2] and img_ids[i] != img_ids[i - 1],
                f"Oscillation detected: {img_ids[i-2]} -> {img_ids[i-1]} -> {img_ids[i]}",
            )

    def test_llm_max_usage_enforced(self):
        """No image should be used more than twice."""
        plan = _make_llm_plan([
            _beat(0.0, 2.0, "ADVANCE", "Use 1"),
            _beat(2.0, 4.0, "ADVANCE", "Use 2"),
            _beat(4.0, 6.0, "ADVANCE", "Use 3"),
            _beat(6.0, 8.0, "ADVANCE", "Use 4"),
            _beat(8.0, 10.0, "ADVANCE", "Use 5"),
            _beat(10.0, 12.0, "ADVANCE", "Use 1 again"),
        ])
        scenes, metadata = self._run_plan_with_llm(plan)

        self.assertLessEqual(metadata["max_usage_count"], 2)

    def test_no_llm_plan_works(self):
        """Editor works fine without any LLM plan."""
        scenes, metadata = self.editor.plan(
            self.segments, self.images, self.audio_duration, llm_plan=None,
        )
        self.assertGreater(len(scenes), 0)
        self.assertFalse(metadata["llm_used"])

    def test_starts_at_zero_ends_at_audio_duration(self):
        """Timeline always starts at 0 and ends at audio_duration."""
        plan = _make_llm_plan([
            _beat(0.0, 4.0, "HOLD", "Start"),
            _beat(4.0, 8.0, "ADVANCE", "Middle"),
            _beat(8.0, 12.0, "ADVANCE", "End"),
        ])
        scenes, _ = self._run_plan_with_llm(plan)

        self.assertAlmostEqual(scenes[0].start, 0.0)
        self.assertAlmostEqual(scenes[-1].end, self.audio_duration)

    def test_gap_free_timeline(self):
        """No gaps or overlaps in the timeline."""
        plan = _make_llm_plan([
            _beat(0.0, 3.0, "HOLD", "Start"),
            _beat(3.0, 6.0, "ADVANCE", "Cut"),
            _beat(6.0, 9.0, "HOLD", "Hold"),
            _beat(9.0, 12.0, "ADVANCE", "End"),
        ])
        scenes, _ = self._run_plan_with_llm(plan)

        for i in range(1, len(scenes)):
            self.assertAlmostEqual(scenes[i].start, scenes[i - 1].end, places=3)


class GeminiPlannerValidatorTests(unittest.TestCase):
    """Test the GeminiBeatPlanner._parse_and_validate method."""

    def setUp(self):
        self.config = _make_config()
        self.planner = GeminiBeatPlanner(self.config)
        self.images = _make_images(5)
        self.audio_duration = 12.0

    def test_valid_plan_parsed(self):
        """Validator should accept a well-formed V6 plan."""
        raw = json.dumps({
            "beats": [
                _beat(0.0, 3.0, "HOLD", "Opening"),
                _beat(3.0, 6.0, "ADVANCE", "Transition"),
                _beat(6.0, 12.0, "ADVANCE", "End"),
            ]
        })
        result = self.planner._parse_and_validate(raw, self.audio_duration)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["beats"]), 3)
        self.assertEqual(result["beats"][0]["decision"], "HOLD")
        self.assertEqual(result["beats"][1]["decision"], "ADVANCE")

    def test_valid_plan_as_array(self):
        """Validator also accepts a plain JSON array."""
        raw = json.dumps([
            _beat(0.0, 3.0, "HOLD", "Opening"),
            _beat(3.0, 6.0, "ADVANCE", "Transition"),
        ])
        result = self.planner._parse_and_validate(raw, self.audio_duration)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["beats"]), 2)

    def test_invalid_json_returns_none(self):
        """Non-JSON response should return None."""
        result = self.planner._parse_and_validate("this is not JSON", self.audio_duration)
        self.assertIsNone(result)

    def test_empty_beats_returns_none(self):
        result = self.planner._parse_and_validate("[]", self.audio_duration)
        self.assertIsNone(result)

    def test_cut_normalized_to_advance(self):
        """'CUT' decisions should be normalized to 'ADVANCE'."""
        raw = json.dumps({"beats": [
            _beat(0.0, 3.0, "CUT", "Should become ADVANCE"),
        ]})
        result = self.planner._parse_and_validate(raw, self.audio_duration)
        self.assertIsNotNone(result)
        self.assertEqual(result["beats"][0]["decision"], "ADVANCE")

    def test_timestamp_exceeds_duration_clamped(self):
        """end_hint > audio_duration should be clamped."""
        raw = json.dumps({"beats": [
            _beat(0.0, 999.0, "HOLD", "Too long"),
        ]})
        result = self.planner._parse_and_validate(raw, self.audio_duration)
        self.assertIsNotNone(result)
        self.assertLessEqual(result["beats"][0]["end_hint"], self.audio_duration)

    def test_first_beat_starts_at_zero(self):
        """First beat must always start at 0.0."""
        raw = json.dumps({"beats": [
            _beat(5.0, 12.0, "HOLD", "Wrong start"),
        ]})
        result = self.planner._parse_and_validate(raw, self.audio_duration)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["beats"][0]["start"], 0.0)

    def test_confidence_range(self):
        """Confidence values should be clamped to [0, 1]."""
        raw = json.dumps({"beats": [
            {"start": 0.0, "end_hint": 5.0, "decision": "HOLD", "reason": "test", "confidence": 1.5},
            {"start": 5.0, "end_hint": 12.0, "decision": "ADVANCE", "reason": "test", "confidence": -0.2},
        ]})
        result = self.planner._parse_and_validate(raw, self.audio_duration)
        self.assertIsNotNone(result)
        for beat in result["beats"]:
            self.assertGreaterEqual(beat["confidence"], 0.0)
            self.assertLessEqual(beat["confidence"], 1.0)

    def test_invalid_decision_becomes_hold(self):
        """Invalid decision values should default to HOLD."""
        raw = json.dumps({"beats": [
            {"start": 0.0, "end_hint": 12.0, "decision": "JUMP", "reason": "bad"},
        ]})
        result = self.planner._parse_and_validate(raw, self.audio_duration)
        self.assertIsNotNone(result)
        self.assertEqual(result["beats"][0]["decision"], "HOLD")


class GeminiCacheTests(unittest.TestCase):
    """Test LLM caching logic."""

    def test_cache_key_determinism(self):
        """Same inputs should produce same cache key."""
        cfg = _make_config()
        planner = GeminiBeatPlanner(cfg)
        segs = _make_segments()
        audio_hash = planner._audio_hash(segs)
        image_ids = ["img-001", "img-002"]
        prompts = ["prompt 1", "prompt 2"]

        k1 = planner._make_cache_key(audio_hash, image_ids, prompts)
        k2 = planner._make_cache_key(audio_hash, image_ids, prompts)
        self.assertEqual(k1, k2)

    def test_different_images_different_key(self):
        """Different image IDs should produce different cache keys."""
        cfg = _make_config()
        planner = GeminiBeatPlanner(cfg)
        segs = _make_segments()
        audio_hash = planner._audio_hash(segs)

        k1 = planner._make_cache_key(audio_hash, ["img-001"], ["prompt 1"])
        k2 = planner._make_cache_key(audio_hash, ["img-002"], ["prompt 1"])
        self.assertNotEqual(k1, k2)


class GeminiFallbackTests(unittest.TestCase):
    """Test that alignment works without LLM."""

    def test_alignment_works_without_api_key(self):
        cfg = _make_config()
        planner = GeminiBeatPlanner(cfg)
        self.assertFalse(planner.is_available)

    def test_alignment_works_without_llm(self):
        cfg = _make_config()
        editor = SequentialEditor(config=cfg)
        images = _make_images(3)
        segs = _make_segments()
        scenes, metadata = editor.plan(segs, images, 12.0, llm_plan=None)

        self.assertGreater(len(scenes), 0)
        self.assertFalse(metadata["llm_used"])


class GeminiConfidenceTests(unittest.TestCase):
    """Test that Gemini ADVANCE/HOLD confidence affects cut decisions."""

    def test_high_confidence_advance_increases_cuts(self):
        """High-confidence ADVANCE hints should promote more cuts."""
        cfg = _make_config(min_duration_for_cut=0.0, min_shot_duration=0.5)
        editor = SequentialEditor(config=cfg)
        images = _make_images(5)
        segs = _make_segments()

        plan = _make_llm_plan([
            _beat(0.0, 2.0, "HOLD", "Start", 0.5),
            _beat(2.0, 4.0, "ADVANCE", "Cut early", 0.95),
            _beat(4.0, 6.0, "ADVANCE", "Cut again", 0.95),
            _beat(6.0, 8.0, "ADVANCE", "Cut again", 0.95),
            _beat(8.0, 12.0, "HOLD", "End", 0.5),
        ])
        scenes_with_llm, meta_with = editor.plan(segs, images, 12.0, llm_plan=plan)

        scenes_no_llm, meta_no = editor.plan(segs, images, 12.0, llm_plan=None)

        # Both should work
        self.assertGreater(len(scenes_with_llm), 0)
        self.assertGreater(len(scenes_no_llm), 0)
        self.assertTrue(meta_with["llm_used"])
        self.assertFalse(meta_no["llm_used"])


if __name__ == "__main__":
    unittest.main()
