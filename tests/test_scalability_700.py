import tempfile
import time
import unittest
from pathlib import Path
from frameforge.alignment import SceneAlignmentEngine
from frameforge.config import FrameForgeConfig
from frameforge.image_processor import ImagePreprocessor
from frameforge.models import ImageAsset, Scene, Timeline, TranscriptSegment, Word
from frameforge.renderer import FFmpegRenderer
from frameforge.service import FrameForgeService
from frameforge.validation import validate_timeline
from tests.helpers import create_real_images_batch, ALL_THEME_NAMES


class Scalability700BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name) / "cache"
        self.config = FrameForgeConfig(cache_dir=self.cache_dir, max_images=700, image_workers=4)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_assets(self, count: int = 700) -> list[ImageAsset]:
        img_dir = Path(self.temp_dir.name) / "images"
        paths = create_real_images_batch(img_dir, count, prefix="img", size=(128, 128))
        themes = ["lighthouse on ocean cliff", "rescue ship at sea", "stormy waves night", "keeper in tower", "golden sunrise coast"]
        assets = []
        for i, fp in enumerate(paths):
            theme = themes[i % len(themes)]
            assets.append(
                ImageAsset(
                    id=f"image-{i+1}",
                    file_path=str(fp),
                    filename=fp.name,
                    prompt=f"A cinematic illustration of {theme} variation {i}",
                )
            )
        return assets

    def _create_synthetic_segments(self, num_segments: int = 60, total_duration: float = 300.0) -> list[TranscriptSegment]:
        segments = []
        seg_dur = total_duration / num_segments
        topics = ["The lonely lighthouse stood on the rugged Atlantic coast.", "Storm clouds gathered rapidly over the ocean.", "Waves battered the vessel in the darkness.", "The keeper illuminated the bright beacon.", "Dawn broke as the rescue vessel approached the shore."]
        for i in range(num_segments):
            start = round(i * seg_dur, 2)
            end = round((i + 1) * seg_dur, 2)
            text = topics[i % len(topics)]
            words = [Word(w, round(start + j * 0.4, 2), round(start + (j + 1) * 0.4, 2)) for j, w in enumerate(text.split())]
            segments.append(TranscriptSegment(text, start, end, words))
        return segments

    def test_700_images_alignment_and_caching(self):
        assets = self._create_synthetic_assets(700)
        segments = self._create_synthetic_segments(60, 300.0)
        audio_duration = 300.0

        engine = SceneAlignmentEngine(config=self.config)

        # Run 1: Cold Run
        t0 = time.perf_counter()
        scenes_1, method = engine.align(segments, assets, audio_duration)
        cold_time = time.perf_counter() - t0

        self.assertGreater(len(scenes_1), 0)
        self.assertEqual(scenes_1[0].start, 0.0)
        self.assertEqual(scenes_1[-1].end, 300.0)

        # Validate gap-free continuous timeline
        timeline_1 = Timeline("dummy_audio.mp3", audio_duration, scenes_1, method, segments)
        validated = validate_timeline(timeline_1, assets, verify_files=False)
        self.assertEqual(validated.scenes[-1].end, 300.0)

        # Run 2: Warm Run (Cache Reuse)
        t1 = time.perf_counter()
        scenes_2, _ = engine.align(segments, assets, audio_duration)
        warm_time = time.perf_counter() - t1

        self.assertEqual(len(scenes_1), len(scenes_2))
        print(f"\n[700-Image Benchmark] Cold: {cold_time:.3f}s | Warm (Cached): {warm_time:.3f}s")

    def test_fault_tolerance_with_missing_prompts_and_duplicates(self):
        assets = self._create_synthetic_assets(50)
        # Clear prompts on some assets to verify automatic fallback
        assets[0].prompt = ""
        assets[5].prompt = "   "
        # Set duplicate
        assets[2].is_duplicate = True
        assets[2].duplicate_of = assets[1].id

        segments = self._create_synthetic_segments(10, 50.0)
        engine = SceneAlignmentEngine(config=self.config)
        scenes, _ = engine.align(segments, assets, 50.0)

        self.assertGreater(len(scenes), 0)
        self.assertEqual(scenes[0].start, 0.0)
        self.assertEqual(scenes[-1].end, 50.0)

    def test_scalable_concat_demuxer_command_generation(self):
        assets = self._create_synthetic_assets(100)
        segments = self._create_synthetic_segments(100, 500.0)
        scenes = [
            Scene(f"scene-{i:03d}", f"image-{i+1}", i * 5.0, (i + 1) * 5.0, f"narration {i}", 0.8)
            for i in range(100)
        ]
        timeline = Timeline("voice.mp3", 500.0, scenes, "semantic")
        renderer = FFmpegRenderer("ffmpeg", config=self.config)
        
        # Verify direct command can be generated without errors
        cmd = renderer.command(timeline, assets, "output.mp4")
        self.assertIn("-shortest", cmd)
        self.assertIn("concat=n=100:v=1:a=0[v]", cmd[cmd.index("-filter_complex") + 1])


if __name__ == "__main__":
    unittest.main()
