"""Regression tests for audio-path persistence across analyze -> render cycle.

Bug: Timeline.as_dict() stripped the audio path to a bare filename, causing
render() to fail with 'Audio file is missing' after the timeline was saved to
SQLite and reconstituted during rendering.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import create_real_image, create_audio_stub

from frameforge.config import FrameForgeConfig
from frameforge.models import ImageAsset, Scene, Timeline
from frameforge.renderer import FFmpegRenderer
from frameforge.service import FrameForgeService
from frameforge.storage import TimelineStore
from frameforge.validation import validate_timeline, TimelineValidationError


class AudioPersistenceTests(unittest.TestCase):
    """Verify audio path survives the full analyze -> store -> load -> render cycle."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)
        self.db_path = self.tmp_path / "test.db"

        # Create real audio
        self.audio = create_audio_stub(self.tmp_path / "test_audio.wav")

        # Create real images with distinct themes
        self.img_dir = self.tmp_path / "images"
        self.img_dir.mkdir()
        self.images_paths = []
        themes = ["ocean", "forest", "desert", "arctic", "space"]
        for i, theme in enumerate(themes):
            p = create_real_image(self.img_dir / f"img_{i}.png", theme=theme, seed=i, label=theme)
            self.images_paths.append(p)

        self.assets = [
            ImageAsset(f"img-{i}", str(p), p.name, f"Description of {p.stem}")
            for i, p in enumerate(self.images_paths)
        ]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_service(self) -> FrameForgeService:
        cfg = FrameForgeConfig(db_path=self.db_path)
        store = TimelineStore(self.db_path)
        return FrameForgeService(store=store, config=cfg)

    def test_timeline_dict_preserves_full_audio_path(self):
        """Timeline.as_dict() must store the full absolute path, not just the filename."""
        full_path = str(self.audio)
        tl = Timeline(full_path, 1.0, [], "test")
        d = tl.as_dict()
        self.assertEqual(d["audio"]["file"], full_path,
                         "as_dict() must preserve the full audio path")

    def test_analyze_saves_full_audio_path_to_db(self):
        """After analyze(), the DB audio_file column must contain the full path."""
        svc = self._make_service()
        project_id = "test-proj-1"

        # Short transcript segments
        from frameforge.models import TranscriptSegment, Word
        segments = [
            TranscriptSegment("Hello world.", 0, 1.0, [Word("Hello", 0, 0.5), Word("world.", 0.5, 1.0)]),
            TranscriptSegment("Second sentence.", 1.0, 2.0, [Word("Second", 1.0, 1.5), Word("sentence.", 1.5, 2.0)]),
        ]

        result = svc.analyze(
            project_id,
            str(self.audio),
            2.0,
            [a.__dict__ for a in self.assets],
            transcript=segments,
        )

        # Check DB column
        job = svc.store.get_latest(project_id)
        self.assertIsNotNone(job, "Timeline was saved to DB")
        self.assertEqual(job["audio_file"], str(self.audio),
                         "DB audio_file column must contain the full absolute path")

        # Check JSON timeline
        tl_json = job["timeline"]
        self.assertEqual(tl_json["audio"]["file"], str(self.audio),
                         "Timeline JSON audio.file must contain the full absolute path")

    def test_render_loads_audio_from_db_after_analyze(self):
        """After analyze, render must find the audio file using the persisted path."""
        svc = self._make_service()
        project_id = "test-proj-2"

        from frameforge.models import TranscriptSegment, Word
        segments = [
            TranscriptSegment("Hello world.", 0, 1.0, [Word("Hello", 0, 0.5), Word("world.", 0.5, 1.0)]),
            TranscriptSegment("Second sentence.", 1.0, 2.0, [Word("Second", 1.0, 1.5), Word("sentence.", 1.5, 2.0)]),
        ]

        svc.analyze(
            project_id,
            str(self.audio),
            2.0,
            [a.__dict__ for a in self.assets],
            transcript=segments,
        )

        # Create a NEW service (simulating server restart)
        svc2 = self._make_service()
        job = svc2.store.get_latest(project_id)
        self.assertIsNotNone(job, "New service can read the saved timeline")

        # Reconstruct timeline the same way render() does
        saved = job["timeline"]
        db_audio_path = Path(job["audio_file"]) if job.get("audio_file") else None
        json_audio_ref = saved["audio"]["file"]

        # The render path resolution logic
        if db_audio_path and db_audio_path.is_file():
            audio_path = str(db_audio_path)
        elif Path(json_audio_ref).is_file():
            audio_path = json_audio_ref
        else:
            audio_path = None

        self.assertIsNotNone(audio_path, "Audio file must be resolvable after store/load cycle")
        self.assertTrue(Path(audio_path).is_file(), f"Resolved audio path must exist: {audio_path}")

    def test_render_with_bare_filename_falls_back_to_uploads_dir(self):
        """If timeline JSON contains a bare filename, render() falls back to uploads dir."""
        svc = self._make_service()
        project_id = "test-proj-3"

        # Manually save a timeline with just a bare filename (simulating old bug)
        bare_filename = self.audio.name  # e.g., "test_audio.wav"
        timeline_dict = {
            "audio": {"file": bare_filename, "duration": 1.0},
            "alignment_method": "test",
            "scenes": [
                {
                    "scene_id": "s1",
                    "image_id": "img-0",
                    "start": 0.0,
                    "end": 1.0,
                    "duration": 1.0,
                    "narration": "test",
                    "confidence": 0.5,
                    "rerank_breakdown": {},
                }
            ],
            "images": [a.__dict__ for a in self.assets],
            "raw_transcript": [],
        }

        svc.store.save(
            "job-1",
            project_id,
            str(self.audio),  # Full path in DB column
            1.0,
            timeline_dict,
            "ready",
            "test",
        )

        # Now simulate what render() does - it should find the audio via DB column
        job = svc.store.get_latest(project_id)
        self.assertEqual(job["audio_file"], str(self.audio))
        self.assertTrue(Path(job["audio_file"]).is_file())

    def test_render_end_to_end_with_validation(self):
        """Full end-to-end: analyze -> store -> load -> validate_timeline with verify_files=True."""
        svc = self._make_service()
        project_id = "test-proj-4"

        from frameforge.models import TranscriptSegment, Word
        segments = [
            TranscriptSegment("Hello world.", 0, 1.0, [Word("Hello", 0, 0.5), Word("world.", 0.5, 1.0)]),
            TranscriptSegment("Second sentence.", 1.0, 2.0, [Word("Second", 1.0, 1.5), Word("sentence.", 1.5, 2.0)]),
        ]

        svc.analyze(
            project_id,
            str(self.audio),
            2.0,
            [a.__dict__ for a in self.assets],
            transcript=segments,
        )

        # Load from DB and rebuild timeline (same as render())
        job = svc.store.get_latest(project_id)
        saved = job["timeline"]

        # Resolve audio path using the same logic as fixed render()
        db_audio_path = Path(job["audio_file"]) if job.get("audio_file") else None
        json_audio_ref = saved["audio"]["file"]
        audio_path = json_audio_ref
        if db_audio_path and db_audio_path.is_file():
            audio_path = str(db_audio_path)

        timeline = Timeline(
            audio_path,
            float(saved["audio"]["duration"]),
            [
                Scene(**{k: v for k, v in scene.items() if k != "duration"})
                for scene in saved["scenes"]
            ],
            saved.get("alignment_method", "manual"),
        )

        # THIS must NOT raise TimelineValidationError("Audio file is missing.")
        validate_timeline(timeline, self.assets, verify_files=True)

    def test_update_preserves_full_audio_path(self):
        """update() must not overwrite DB audio_file with a bare filename."""
        svc = self._make_service()
        project_id = "test-proj-5"

        from frameforge.models import TranscriptSegment, Word
        segments = [
            TranscriptSegment("Hello.", 0, 1.0, [Word("Hello", 0, 1.0)]),
        ]

        svc.analyze(
            project_id,
            str(self.audio),
            1.0,
            [a.__dict__ for a in self.assets],
            transcript=segments,
        )

        # Simulate a timeline update with bare filename (from client)
        job = svc.store.get_latest(project_id)
        tl = job["timeline"]
        tl["audio"]["file"] = self.audio.name  # Bare filename!

        svc.update(project_id, tl)

        # Verify the DB still has the full path
        job2 = svc.store.get_latest(project_id)
        self.assertEqual(job2["audio_file"], str(self.audio),
                         "update() must preserve the full audio path in the DB")


class RegressionEndToEndRender(unittest.TestCase):
    """Simulate full analyze -> server restart -> render pipeline."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp)
        self.db_path = self.tmp_path / "e2e.db"
        self.output_dir = self.tmp_path / "renders"
        self.output_dir.mkdir()

        self.audio = create_audio_stub(self.tmp_path / "voiceover.wav")
        # audio_stub is 1 second; record that for the test
        self.audio_duration = 1.0

        self.img_dir = self.tmp_path / "images"
        self.img_dir.mkdir()
        themes = ["ocean", "forest", "desert", "arctic", "space", "fire", "sunset", "city"]
        self.images_paths = []
        for i, theme in enumerate(themes):
            p = create_real_image(self.img_dir / f"img_{i}.png", theme=theme, seed=i)
            self.images_paths.append(p)

        self.assets = [
            ImageAsset(f"img-{i}", str(p), p.name, f"Image of {theme}")
            for i, (p, theme) in enumerate(zip(self.images_paths, themes))
        ]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_analyze_render_cycle(self):
        """Full cycle: analyze with Service1, render with Service2 (simulating restart)."""
        from frameforge.models import TranscriptSegment, Word

        segments = [
            TranscriptSegment("The ocean waves crash.", 0, 0.5, [Word("The", 0, 0.1), Word("ocean", 0.1, 0.3), Word("waves", 0.3, 0.4), Word("crash.", 0.4, 0.5)]),
            TranscriptSegment("A forest stretches.", 0.5, 1.0, [Word("A", 0.5, 0.6), Word("forest", 0.6, 0.8), Word("stretches.", 0.8, 1.0)]),
        ]

        # Service 1: Analyze
        cfg1 = FrameForgeConfig(db_path=self.db_path)
        store1 = TimelineStore(self.db_path)
        svc1 = FrameForgeService(store=store1, config=cfg1)
        project_id = "e2e-test"

        svc1.analyze(
            project_id,
            str(self.audio),
            self.audio_duration,
            [a.__dict__ for a in self.assets],
            transcript=segments,
        )
        store1.close()

        # Service 2: Render (simulating server restart - completely new connections)
        cfg2 = FrameForgeConfig(db_path=self.db_path)
        store2 = TimelineStore(self.db_path)
        svc2 = FrameForgeService(store=store2, config=cfg2)

        output = str(self.output_dir / "output.mp4")

        # This must NOT fail with "Audio file is missing."
        result = svc2.render(
            project_id,
            output,
            preset="landscape",
            render_profile="fast",
        )

        self.assertIn("output", result)
        self.assertTrue(Path(result["output"]).is_file(), "MP4 must exist")

        metadata = result["metadata"]
        self.assertEqual(metadata["video_codec"], "h264")
        self.assertEqual(metadata["audio_codec"], "aac")
        self.assertAlmostEqual(metadata["duration"], self.audio_duration, delta=0.5)

        store2.close()


if __name__ == "__main__":
    unittest.main()
