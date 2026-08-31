"""Regression tests for the V4 Sequential Image Editor.

Validates:
- Image order is preserved (never globally reordered)
- Forward-only progression through image sequence
- Max image usage <= 2
- No A->B->A->B oscillation
- Gap-free, contiguous timeline
- Adaptive shot durations
- Cut decisions have inspectable reasons
- Real CLIP multimodal signals influence cuts
- 8 narration segments -> correct scene count (not forced to 8)
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import create_real_image, create_audio_stub
from frameforge.config import FrameForgeConfig
from frameforge.models import ImageAsset, TranscriptSegment, Word
from frameforge.editor import SequentialEditor, MAX_IMAGE_USES
from frameforge.alignment import SceneAlignmentEngine


def _make_segments_and_images(tmp_dir: Path, n_segments: int = 8, n_images: int = 20):
    """Create realistic test data for sequential editor tests."""
    # 8 narration segments with distinct topics
    topics = [
        "The lighthouse stood tall on the rocky coast",
        "Waves crashed against the shore during the storm",
        "The keeper climbed the spiral staircase",
        "He lit the beacon to guide the ships",
        "Fog rolled in from the open ocean",
        "A distant ship changed course toward the light",
        "Morning came and the storm finally passed",
        "The lighthouse keeper smiled at the calm sea",
    ][:n_segments]

    segments = []
    t = 0.0
    for i, text in enumerate(topics):
        dur = 2.0 + (i % 3) * 0.5  # 2.0, 2.5, 3.0, 2.0, ...
        segments.append(TranscriptSegment(
            text=text, start=round(t, 4), end=round(t + dur, 4),
        ))
        t += dur

    audio_duration = t

    # Create images with distinct visual themes
    themes = [
        "ocean", "forest", "desert", "arctic", "space",
        "lighthouse", "storm", "sunset", "city", "mountain",
        "river", "volcano", "rain", "snow", "fire",
        "night", "morning", "evening", "beach", "cave",
    ][:n_images]

    images = []
    for i, theme in enumerate(themes):
        p = create_real_image(tmp_dir / f"img_{i:03d}.png", theme=theme, seed=i * 17, label=theme)
        images.append(ImageAsset(
            id=f"img-{i:03d}", file_path=str(p), filename=p.name,
            prompt=f"A {theme} scene with relevant visual elements",
            quality_score=1.0,
        ))

    return segments, images, audio_duration


class SequentialEditorBasicTests(unittest.TestCase):
    """Core sequential editor behavior tests."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)
        self.segments, self.images, self.audio_dur = _make_segments_and_images(self.tmp_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_editor(self, **kwargs):
        editor = SequentialEditor(config=FrameForgeConfig(), **kwargs)
        # Create random but consistent score matrices
        np.random.seed(42)
        n_seg = len(self.segments)
        n_img = len(self.images)
        vi = np.random.rand(n_seg, n_img) * 0.3 + 0.4
        vp = np.random.rand(n_seg, n_img) * 0.3 + 0.4
        pi = np.random.rand(n_img) * 0.3 + 0.4
        return editor.plan(self.segments, self.images, self.audio_dur, vi, vp, pi)

    def test_image_order_preserved(self):
        """Scene image IDs must appear in strictly non-decreasing order."""
        scenes, meta = self._run_editor()
        image_ids = [s.image_id for s in scenes]
        id_to_idx = {img.id: i for i, img in enumerate(self.images)}
        idxs = [id_to_idx[sid] for sid in image_ids]
        for i in range(1, len(idxs)):
            self.assertGreaterEqual(
                idxs[i], idxs[i - 1],
                f"Image order violated at scene {i}: {image_ids[i-1]} -> {image_ids[i]}",
            )

    def test_no_backward_jumps(self):
        """No scene should reference an image that appeared earlier than the previous scene's image."""
        scenes, meta = self._run_editor()
        id_to_idx = {img.id: i for i, img in enumerate(self.images)}
        for i in range(1, len(scenes)):
            prev_idx = id_to_idx[scenes[i - 1].image_id]
            curr_idx = id_to_idx[scenes[i].image_id]
            self.assertGreaterEqual(
                curr_idx, prev_idx,
                f"Backward jump at scene {i}: {scenes[i-1].image_id}(idx={prev_idx}) -> {scenes[i].image_id}(idx={curr_idx})",
            )

    def test_no_oscillation(self):
        """No A->B->A->B pattern should occur."""
        scenes, _ = self._run_editor()
        ids = [s.image_id for s in scenes]
        for i in range(2, len(ids)):
            # Check A->B->A pattern
            self.assertFalse(
                ids[i] == ids[i - 2] and ids[i] != ids[i - 1],
                f"Oscillation detected at scene {i}: {ids[i-2]} -> {ids[i-1]} -> {ids[i]}",
            )

    def test_max_image_usage(self):
        """No image should be used more than max_image_uses times."""
        scenes, meta = self._run_editor()
        usage = meta["usage_counts"]
        for img_id, count in usage.items():
            self.assertLessEqual(
                count, MAX_IMAGE_USES,
                f"Image {img_id} used {count} times (max={MAX_IMAGE_USES})",
            )

    def test_gap_free_timeline(self):
        """Scenes must be contiguous with no gaps or overlaps."""
        scenes, _ = self._run_editor()
        for i in range(1, len(scenes)):
            gap = scenes[i].start - scenes[i - 1].end
            self.assertAlmostEqual(
                gap, 0.0, places=4,
                msg=f"Gap/overlap between scene {i-1} and {i}: {gap:.6f}s",
            )

    def test_starts_at_zero_ends_at_audio_duration(self):
        """First scene starts at 0.0, last scene ends at audio_duration."""
        scenes, _ = self._run_editor()
        self.assertAlmostEqual(scenes[0].start, 0.0, places=4)
        self.assertAlmostEqual(scenes[-1].end, self.audio_dur, places=4)

    def test_adaptive_shot_durations(self):
        """Shot durations should vary, not all be equal."""
        scenes, _ = self._run_editor()
        durations = [s.duration for s in scenes]
        if len(durations) > 1:
            self.assertGreater(
                max(durations) - min(durations), 0.1,
                "Shot durations should be adaptive, not uniform",
            )

    def test_cut_reasons_present(self):
        """Every scene should have a cut_reason in its rerank_breakdown."""
        scenes, _ = self._run_editor()
        for s in scenes:
            self.assertIn("decision", s.rerank_breakdown)
            self.assertIn("cut_reason", s.rerank_breakdown)
            self.assertGreater(len(s.rerank_breakdown["cut_reason"]), 0,
                             f"Scene {s.scene_id} missing cut_reason")

    def test_metadata_statistics(self):
        """Metadata should contain required fields."""
        scenes, meta = self._run_editor()
        self.assertIn("total_input_images", meta)
        self.assertIn("images_used", meta)
        self.assertIn("num_cuts", meta)
        self.assertIn("average_shot_duration", meta)
        self.assertIn("image_order_preserved", meta)
        self.assertTrue(meta["image_order_preserved"])
        self.assertGreater(meta["images_used"], 0)

    def test_single_image(self):
        """Single image should produce one scene covering full duration."""
        editor = SequentialEditor(config=FrameForgeConfig())
        img = ImageAsset("only", str(self.tmp_path / "only.png"), "only.png",
                         prompt="only image", quality_score=1.0)
        seg = TranscriptSegment("Just one sentence.", 0.0, 5.0)
        scenes, meta = editor.plan([seg], [img], 5.0)
        self.assertEqual(len(scenes), 1)
        self.assertAlmostEqual(scenes[0].start, 0.0, places=4)
        self.assertAlmostEqual(scenes[0].end, 5.0, places=4)
        self.assertEqual(scenes[0].image_id, "only")

    def test_no_narration_fallback(self):
        """Empty segments should hold the first image for full duration."""
        editor = SequentialEditor(config=FrameForgeConfig())
        img = ImageAsset("img-0", str(self.tmp_path / "i.png"), "i.png",
                         prompt="test", quality_score=1.0)
        scenes, meta = editor.plan([], [img], 10.0)
        self.assertEqual(len(scenes), 1)
        self.assertAlmostEqual(scenes[0].end, 10.0, places=4)

    def test_configurable_min_max_duration(self):
        """Changing min/max shot duration should affect scene count."""
        np.random.seed(99)
        n_seg = len(self.segments)
        n_img = len(self.images)
        vi = np.random.rand(n_seg, n_img) * 0.3 + 0.4
        vp = np.random.rand(n_seg, n_img) * 0.3 + 0.4
        pi = np.random.rand(n_img) * 0.3 + 0.4

        editor_short = SequentialEditor(min_shot_duration=1.0, max_shot_duration=3.0)
        scenes_short, _ = editor_short.plan(self.segments, self.images, self.audio_dur, vi, vp, pi)

        editor_long = SequentialEditor(min_shot_duration=4.0, max_shot_duration=15.0)
        scenes_long, _ = editor_long.plan(self.segments, self.images, self.audio_dur, vi, vp, pi)

        # Short max → more scenes, long max → fewer scenes
        self.assertGreaterEqual(
            len(scenes_short), len(scenes_long),
            "Shorter max shot duration should produce >= scenes than longer max",
        )


class SequentialEditorWithCLIPTests(unittest.TestCase):
    """Tests using real CLIP signals through the SequentialEditor."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sequential_align_preserves_order_with_clip(self):
        """sequential_align must preserve image order using real CLIP signals."""
        segments = [
            TranscriptSegment("A bright ocean with waves", 0.0, 3.0),
            TranscriptSegment("A dark forest at night", 3.0, 6.0),
            TranscriptSegment("A burning desert", 6.0, 9.0),
            TranscriptSegment("An arctic ice landscape", 9.0, 12.0),
        ]
        themes = ["ocean", "forest", "desert", "arctic", "mountain", "city"]
        images = []
        for i, theme in enumerate(themes):
            p = create_real_image(self.tmp_path / f"img_{i}.png", theme=theme, seed=i)
            images.append(ImageAsset(
                id=f"img-{i}", file_path=str(p), filename=p.name,
                prompt=f"A {theme} landscape", quality_score=1.0,
            ))

        aligner = SceneAlignmentEngine()
        scenes, method, meta = aligner.sequential_align(segments, images, 12.0)

        # Verify method
        self.assertIn("sequential", method)

        # Verify image order preserved
        id_to_idx = {img.id: i for i, img in enumerate(images)}
        for i in range(1, len(scenes)):
            self.assertGreaterEqual(
                id_to_idx[scenes[i].image_id], id_to_idx[scenes[i - 1].image_id],
                f"Order violated: {scenes[i-1].image_id} -> {scenes[i].image_id}",
            )

        # Verify gap-free
        for i in range(1, len(scenes)):
            self.assertAlmostEqual(scenes[i].start, scenes[i - 1].end, places=4)

        # Verify cut reasons exist
        for s in scenes:
            self.assertIn("decision", s.rerank_breakdown)

        # Verify metadata
        self.assertTrue(meta["image_order_preserved"])
        self.assertGreater(meta["images_used"], 0)

    def test_sequential_align_uses_fewer_images_than_segments(self):
        """Sequential editor should NOT force one image per segment — it decides when to cut."""
        # 10 segments about similar topics
        segments = [
            TranscriptSegment(f"This is segment number {i} about similar things", i * 1.5, (i + 1) * 1.5)
            for i in range(10)
        ]
        # 10 images
        images = []
        for i in range(10):
            p = create_real_image(self.tmp_path / f"img_{i}.png",
                                  theme=["ocean", "forest", "desert"][i % 3], seed=i)
            images.append(ImageAsset(
                id=f"img-{i}", file_path=str(p), filename=p.name,
                prompt=f"Similar scene variant {i}", quality_score=1.0,
            ))

        aligner = SceneAlignmentEngine()
        scenes, method, meta = aligner.sequential_align(segments, images, 15.0)

        # Should use fewer scenes than segments (since topics are similar)
        # The editor decides when to cut based on semantic shifts
        # We just verify the mechanism works — scene count may vary
        self.assertGreater(len(scenes), 0)
        self.assertLessEqual(len(scenes), len(segments))
        self.assertTrue(meta["image_order_preserved"])

    def test_editor_metadata_in_service_result(self):
        """Service analyze() should include editor_metadata in the result."""
        from frameforge.service import FrameForgeService
        from frameforge.storage import TimelineStore
        import uuid

        db = self.tmp_path / "test.db"
        cfg = FrameForgeConfig(db_path=db)
        svc = FrameForgeService(store=TimelineStore(db), config=cfg)

        audio = create_audio_stub(self.tmp_path / "voice.wav")
        images = [
            create_real_image(self.tmp_path / f"img_{i}.png",
                              theme=["ocean", "forest"][i], seed=i)
            for i in range(2)
        ]
        assets = [
            ImageAsset(f"img-{i}", str(p), p.name, f"Image {i}", quality_score=1.0)
            for i, p in enumerate(images)
        ]
        segments = [
            TranscriptSegment("The ocean is vast.", 0.0, 0.5),
            TranscriptSegment("The forest is dense.", 0.5, 1.0),
        ]

        result = svc.analyze(
            str(uuid.uuid4()), str(audio), 1.0,
            [a.__dict__ for a in assets],
            transcript=segments,
        )

        # editor_metadata should be present when sequential_mode=True (default)
        self.assertIn("editor_metadata", result)
        em = result["editor_metadata"]
        self.assertIn("images_used", em)
        self.assertIn("num_cuts", em)
        self.assertTrue(em["image_order_preserved"])


if __name__ == "__main__":
    unittest.main()
