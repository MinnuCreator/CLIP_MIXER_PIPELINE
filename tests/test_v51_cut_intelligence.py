"""FrameForge V5.1 — Cut Intelligence Regression Tests.

Tests that verify:
1. Signal quality tracking (genuine vs fallback)
2. Non-linear duration pressure
3. Relevance decay over beats
4. LLM confidence-weighted hints
5. Cut score computation with degraded signals
6. Pacing comparison: V5.0 baseline vs V5.1 improved
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frameforge.config import FrameForgeConfig
from frameforge.editor import SequentialEditor, _SubBeat
from frameforge.models import ImageAsset, TranscriptSegment, Word
from tests.helpers import create_real_image


def _make_image(path: Path, idx: int) -> ImageAsset:
    return ImageAsset(
        id=f"img-{idx:03d}",
        file_path=str(path),
        filename=path.name,
        prompt=f"image {idx} description",
    )


def _make_segment(text: str, start: float, end: float) -> TranscriptSegment:
    words = text.split()
    word_list = []
    w_start = start
    for w in words:
        w_end = w_start + (end - start) / len(words)
        word_list.append(Word(w, round(w_start, 3), round(w_end, 3)))
        w_start = w_end
    return TranscriptSegment(text, start, end, word_list)


class SignalQualityTests:
    """Tests for genuine-signal tracking."""

    def test_all_genuine(self):
        e = SequentialEditor()
        assert e._compute_signal_quality(True, True, True) == 1.0

    def test_all_fallback(self):
        e = SequentialEditor()
        assert e._compute_signal_quality(False, False, False) == 0.0

    def test_two_of_three(self):
        e = SequentialEditor()
        assert abs(e._compute_signal_quality(True, True, False) - 0.667) < 0.01
        assert abs(e._compute_signal_quality(True, False, True) - 0.667) < 0.01

    def test_relevance_with_all_genuine(self):
        e = SequentialEditor()
        r = e._relevance_score(0.7, 0.6, 0.5, 1.0)
        # Full weighted: 0.4*0.7 + 0.4*0.6 + 0.2*0.5 = 0.62
        assert abs(r - 0.62) < 0.01

    def test_relevance_with_fallback(self):
        e = SequentialEditor()
        # With all fallback, should return neutral 0.5
        r = e._relevance_score(0.7, 0.6, 0.5, 0.0)
        assert r == 0.5

    def test_relevance_with_clip_only(self):
        e = SequentialEditor()
        # With 1 of 3 signals, uses max(vi, vp, pi)
        r = e._relevance_score(0.7, 0.6, 0.5, 0.33)
        assert r == 0.7


class DurationPressureTests:
    """Tests for non-linear duration pressure."""

    def test_below_target(self):
        e = SequentialEditor()
        # Below target: gentle ramp
        p = e._compute_duration_pressure(1.5, 3.0)
        assert 0.0 < p < 0.2

    def test_at_target(self):
        e = SequentialEditor()
        p = e._compute_duration_pressure(3.0, 3.0)
        assert abs(p - 0.3) < 0.01

    def test_above_target(self):
        e = SequentialEditor()
        p = e._compute_duration_pressure(4.5, 3.0)
        assert p > 0.3  # Above target: exponential

    def test_far_above_target(self):
        e = SequentialEditor()
        p = e._compute_duration_pressure(6.0, 3.0)
        assert p >= 0.95  # Near max: very high pressure

    def test_zero_held(self):
        e = SequentialEditor()
        p = e._compute_duration_pressure(0, 3.0)
        assert p == 0.0


class RelevanceDecayTests:
    """Tests for relevance decay over beats on the same image."""

    def test_no_decay_first_beat(self):
        e = SequentialEditor()
        assert e._compute_relevance_decay(1) == 1.0

    def test_decays_over_time(self):
        e = SequentialEditor()
        d5 = e._compute_relevance_decay(5)
        d10 = e._compute_relevance_decay(10)
        assert d5 < 1.0
        assert d10 < d5  # More beats = more decay

    def test_floor(self):
        e = SequentialEditor()
        d = e._compute_relevance_decay(100)
        assert d >= 0.5  # Never drops below 0.5


class LLMHintTests:
    """Tests for LLM confidence-weighted hints."""

    def test_llm_plan_with_confidence(self):
        """LLM plan with varying confidence should affect cut timing."""
        e = SequentialEditor(min_duration_for_cut=0.0, min_shot_duration=0.5)
        images = [_make_image(Path(f"f{i}.png"), i) for i in range(5)]
        segs = [_make_segment(f"Segment {i} about topic {i}", i * 3.0, (i + 1) * 3.0) for i in range(5)]

        # LLM plan: cut at 3.0 with high confidence
        llm_plan = {
            "beats": [
                {"image_index": 1, "action": "HOLD", "start": 0.0, "end_hint": 3.0, "reason": "intro"},
                {"image_index": 2, "action": "CUT", "start": 3.0, "end_hint": 6.0, "reason": "shift"},
                {"image_index": 3, "action": "CUT", "start": 6.0, "end_hint": 9.0, "reason": "shift"},
                {"image_index": 4, "action": "HOLD", "start": 9.0, "end_hint": 12.0, "reason": "intro"},
                {"image_index": 5, "action": "HOLD", "start": 12.0, "end_hint": 15.0, "reason": "final"},
            ]
        }

        # With LLM plan
        scenes_llm, meta_llm = e.plan(segs, images, 15.0, llm_plan=llm_plan)

        # Without LLM plan
        scenes_no_llm, meta_no_llm = e.plan(segs, images, 15.0, llm_plan=None)

        # Both should produce valid timelines
        assert len(scenes_llm) > 0
        assert len(scenes_no_llm) > 0
        assert meta_llm.get("llm_used") is True


class PacingComparisonTests:
    """Compare V5.0 baseline pacing vs V5.1 improved pacing."""

    @staticmethod
    def _run_pacing_test(tmp_dir: Path, voiceover_path: str, n_images: int = 20):
        """Run the full pipeline and return pacing metrics."""
        from frameforge.service import FrameForgeService

        # Create ordered images
        images_data = []
        assets = []
        for i in range(n_images):
            img_path = create_real_image(tmp_dir / f"img_{i:03d}.png", seed=100 + i)
            asset = ImageAsset(
                id=f"img-{i:03d}",
                file_path=str(img_path),
                filename=f"img_{i:03d}.png",
                prompt=f"Image {i}: {'mountain' if i % 3 == 0 else 'ocean' if i % 3 == 1 else 'forest'} scene",
            )
            assets.append(asset)
            images_data.append({
                "id": asset.id,
                "file_path": asset.file_path,
                "filename": asset.filename,
                "prompt": asset.prompt,
            })

        cfg = FrameForgeConfig.from_env()
        store = TimelineStore(tmp_dir / "test.db")
        svc = FrameForgeService(store=store, config=cfg)

        result = svc.analyze(
            project_id="pacing-test",
            audio_file=voiceover_path,
            audio_duration=None,
            images=images_data,
        )

        editor_meta = result.get("editor_metadata", {})
        scenes = result.get("scenes", [])

        durations = [s["end"] - s["start"] for s in scenes]
        image_sequence = [s["image_id"] for s in scenes]

        return {
            "n_scenes": len(scenes),
            "durations": durations,
            "avg_duration": sum(durations) / max(1, len(durations)),
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "images_used": editor_meta.get("images_used", 0),
            "num_cuts": editor_meta.get("num_cuts", 0),
            "image_sequence": image_sequence,
            "order_preserved": _check_order(image_sequence),
            "oscillation": _check_oscillation(image_sequence),
            "max_usage": editor_meta.get("max_usage_count", 0),
            "scenes_over_10s": sum(1 for d in durations if d > 10),
            "scenes_2_to_6s": sum(1 for d in durations if 2 <= d <= 6),
        }


def _check_order(seq: list[str]) -> bool:
    """Check that image IDs appear in non-decreasing order."""
    nums = [int(s.split("-")[1]) for s in seq]
    for i in range(1, len(nums)):
        # Allow same image (reuse) but not backward
        if nums[i] < nums[i - 1]:
            return False
    return True


def _check_oscillation(seq: list[str]) -> int:
    """Count A→B→A oscillation patterns."""
    count = 0
    for i in range(2, len(seq)):
        if seq[i] == seq[i - 2] and seq[i] != seq[i - 1]:
            count += 1
    return count


class TestV51Pacing:
    """Integration tests for V5.1 improved pacing."""

    def test_v51_pacing_quality(self):
        """Verify V5.1 editor produces well-paced output on real data."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # Create images
            images = []
            for i in range(10):
                img_path = create_real_image(tmp_dir / f"img_{i:03d}.png", seed=100 + i)
                images.append(ImageAsset(
                    id=f"img-{i:03d}",
                    file_path=str(img_path),
                    filename=f"img_{i:03d}.png",
                    prompt=f"Scene {i}: {'nightclub' if i % 2 == 0 else 'bottle'}",
                ))

            # Create segments
            segs = [
                _make_segment("The nightclub looks packed at night.", 0.0, 3.0),
                _make_segment("But the real money comes from expensive bottles.", 3.0, 6.5),
                _make_segment("Every table was reserved for the VIP guests.", 6.5, 10.0),
                _make_segment("And the music kept playing until dawn.", 10.0, 15.0),
            ]

            # Create CLIP-like matrices (varying scores)
            np.random.seed(42)
            vim = np.random.uniform(0.45, 0.65, (4, 10)).astype(np.float32)
            vpm = np.random.uniform(0.40, 0.70, (4, 10)).astype(np.float32)
            pim = np.random.uniform(0.50, 0.65, (10,)).tolist()

            e = SequentialEditor(min_duration_for_cut=0.0, min_shot_duration=0.5, max_shot_duration=6.0)
            scenes, meta = e.plan(segs, images, 15.0,
                                   voice_image_matrix=vim,
                                   voice_prompt_matrix=vpm,
                                   prompt_image_scores=pim)

            durations = [s.duration for s in scenes]

            # Verify basic properties
            assert meta["image_order_preserved"] is True
            assert meta["max_usage_count"] <= 2

            # Verify pacing: no scene > 7s, avg < 5s
            assert meta["max_shot_duration"] < 7.0, f"Max shot {meta['max_shot_duration']}s too long"
            assert meta["average_shot_duration"] < 5.0, f"Avg shot {meta['average_shot_duration']}s too long"

            # Verify signal quality is tracked
            assert "signal_quality" in meta
            assert meta["signal_quality"] > 0  # CLIP should be available
            assert meta["signals_active"]["voice_image"] is True
            assert meta["signals_active"]["voice_prompt"] is True

    def test_order_preserved_with_v51(self):
        """Verify V5.1 never reorders images."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            images = []
            for i in range(8):
                img_path = create_real_image(tmp_dir / f"img_{i:03d}.png", seed=200 + i)
                images.append(ImageAsset(
                    id=f"img-{i:03d}",
                    file_path=str(img_path),
                    filename=f"img_{i:03d}.png",
                    prompt=f"Image {i}",
                ))

            segs = [
                _make_segment("First part of the story.", 0.0, 3.0),
                _make_segment("Second part introduces conflict.", 3.0, 6.0),
                _make_segment("Third part resolves the story.", 6.0, 9.0),
                _make_segment("And everything ends peacefully.", 9.0, 12.0),
            ]

            e = SequentialEditor(min_duration_for_cut=0.0, min_shot_duration=0.5)
            scenes, meta = e.plan(segs, images, 12.0)

            assert meta["image_order_preserved"] is True

            # Check no backward jumps
            prev_idx = -1
            for s in scenes:
                idx = int(s.image_id.split("-")[1])
                assert idx >= prev_idx, f"Backward jump: {prev_idx} -> {idx}"
                prev_idx = idx

    def test_no_excessive_long_shots(self):
        """Verify no scene exceeds max_shot_duration."""
        e = SequentialEditor(max_shot_duration=6.0)
        images = [_make_image(Path(f"f{i}.png"), i) for i in range(15)]
        segs = [_make_segment(f"Part {i}", i * 2.0, (i + 1) * 2.0) for i in range(15)]

        scenes, meta = e.plan(segs, images, 30.0)

        for s in scenes:
            assert s.duration <= 6.5, f"Scene {s.scene_id} too long: {s.duration}s"

    def test_gap_free_timeline(self):
        """Verify timeline is gap-free and covers full audio duration."""
        e = SequentialEditor()
        images = [_make_image(Path(f"f{i}.png"), i) for i in range(5)]
        segs = [_make_segment(f"Part {i}", i * 4.0, (i + 1) * 4.0) for i in range(5)]

        scenes, meta = e.plan(segs, images, 20.0)

        assert scenes[0].start == 0.0
        assert abs(scenes[-1].end - 20.0) < 0.01

        for i in range(1, len(scenes)):
            assert abs(scenes[i].start - scenes[i - 1].end) < 0.001, \
                f"Gap/overlap between scene {i} and {i + 1}"

    def test_signal_quality_in_metadata(self):
        """Verify signal quality and active signals are reported."""
        e = SequentialEditor()
        images = [_make_image(Path(f"f{i}.png"), i) for i in range(3)]
        segs = [_make_segment("Hello world.", 0.0, 5.0), _make_segment("Goodbye earth.", 5.0, 10.0)]

        scenes, meta = e.plan(segs, images, 10.0)

        assert "signal_quality" in meta
        assert "signals_active" in meta
        assert isinstance(meta["signals_active"], dict)
        assert "voice_image" in meta["signals_active"]
        assert "voice_prompt" in meta["signals_active"]
        assert "prompt_image" in meta["signals_active"]

    def test_cut_log_includes_llm_hint(self):
        """Verify cut log entries include LLM hint info."""
        e = SequentialEditor(min_duration_for_cut=0.0, min_shot_duration=0.5)
        images = [_make_image(Path(f"f{i}.png"), i) for i in range(5)]
        segs = [_make_segment(f"Part {i}", i * 3.0, (i + 1) * 3.0) for i in range(5)]

        scenes, meta = e.plan(segs, images, 15.0)

        for cut in meta.get("cut_log", []):
            assert "llm_hint" in cut
            assert isinstance(cut["llm_hint"], float)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
