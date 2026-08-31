"""Multi-Scale Benchmarks across 100, 300, and 700 Images / Scenes."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from frameforge.alignment import SceneAlignmentEngine
from frameforge.config import FrameForgeConfig
from frameforge.image_processor import ImagePreprocessor
from frameforge.models import ImageAsset, Scene, Timeline, TranscriptSegment, Word
from frameforge.renderer import FFmpegRenderer
from frameforge.validation import validate_timeline
from tests.helpers import create_real_images_batch, ALL_THEME_NAMES


class MultiScaleBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name) / "cache"
        self.config = FrameForgeConfig(cache_dir=self.cache_dir, max_images=700, image_workers=4)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _generate_synthetic_project(self, image_count: int, segment_count: int, total_duration: float):
        themes = [
            "lighthouse on coastal cliff",
            "cargo ship crossing dark ocean",
            "violent storm crashing rocks",
            "keeper in tower beacon",
            "rescue ship approaching at sunrise",
            "aerial coast panorama",
            "sailors on deck ocean",
            "misty morning horizon",
        ]
        img_dir = Path(self.temp_dir.name) / f"images_{image_count}"
        paths = create_real_images_batch(img_dir, image_count, prefix=f"img{image_count}", size=(128, 128))
        assets = []
        for i, fp in enumerate(paths):
            theme = themes[i % len(themes)]
            # Deliberately leave some prompts blank to test visual attribute fallback
            prompt = f"Cinematic photography of {theme} {i}" if (i % 3 != 0) else ""
            assets.append(ImageAsset(f"img-{i+1}", str(fp), fp.name, prompt))

        segments = []
        seg_dur = total_duration / segment_count
        narrations = [
            "For more than a century the lonely lighthouse stood beside the Atlantic ocean.",
            "Dark storm clouds rapidly gathered across the northern horizon.",
            "Violent waves battered the small vessel struggling in the rough sea.",
            "Inside the tower the keeper urgently activated the bright emergency beacon.",
            "At dawn the rescue ship arrived safely along the calm coastline.",
        ]
        for s in range(segment_count):
            st = round(s * seg_dur, 2)
            en = round((s + 1) * seg_dur, 2)
            text = narrations[s % len(narrations)]
            words = [Word(w, round(st + j * 0.3, 2), round(st + (j + 1) * 0.3, 2)) for j, w in enumerate(text.split())]
            segments.append(TranscriptSegment(text, st, en, words))

        return assets, segments

    def _benchmark_scale(self, image_count: int, segment_count: int, duration: float) -> dict:
        assets, segments = self._generate_synthetic_project(image_count, segment_count, duration)
        preprocessor = ImagePreprocessor(self.config)
        alignment_engine = SceneAlignmentEngine(config=self.config)

        # 1. Cold Run Preprocessing & Alignment
        t0 = time.perf_counter()
        prep_assets = preprocessor.preprocess_batch(assets)
        scenes_cold, method_cold = alignment_engine.align(segments, prep_assets, duration)
        cold_time = time.perf_counter() - t0

        timeline_cold = Timeline("audio.mp3", duration, scenes_cold, method_cold, segments)
        validate_timeline(timeline_cold, prep_assets, verify_files=False)

        # 2. Warm Run (Full Cache Reuse)
        t1 = time.perf_counter()
        prep_assets_warm = preprocessor.preprocess_batch(assets)
        scenes_warm, method_warm = alignment_engine.align(segments, prep_assets_warm, duration)
        warm_time = time.perf_counter() - t1

        timeline_warm = Timeline("audio.mp3", duration, scenes_warm, method_warm, segments)
        validate_timeline(timeline_warm, prep_assets_warm, verify_files=False)

        # 3. Renderer Concat Demuxer Script Stress Test
        renderer = FFmpegRenderer("ffmpeg", config=self.config)
        scenes_stress = [
            Scene(f"sc-{i:04d}", f"img-{(i % image_count) + 1}", i * (duration / segment_count), (i + 1) * (duration / segment_count), f"narr {i}", 0.8)
            for i in range(segment_count)
        ]
        stress_timeline = Timeline("audio.mp3", duration, scenes_stress, "fast", segments)
        cmd = renderer.command(stress_timeline, prep_assets_warm, "out.mp4")
        self.assertIn("-shortest", cmd)

        speedup = (cold_time / max(0.001, warm_time))

        return {
            "image_count": image_count,
            "scene_count": len(scenes_cold),
            "cold_time": cold_time,
            "warm_time": warm_time,
            "speedup": speedup,
        }

    def test_multiscale_100_300_700_benchmarks(self):
        results = []
        for img_cnt, seg_cnt, dur in [(100, 25, 120.0), (300, 50, 240.0), (700, 80, 360.0)]:
            res = self._benchmark_scale(img_cnt, seg_cnt, dur)
            results.append(res)
            print(f"\n[Scale {res['image_count']} Images / {res['scene_count']} Scenes] Cold: {res['cold_time']:.3f}s | Warm: {res['warm_time']:.3f}s | Speedup: {res['speedup']:.1f}x")

        self.assertEqual(len(results), 3)
        # Verify warm time is strictly faster than cold time across all scales
        for r in results:
            self.assertLess(r["warm_time"], r["cold_time"])


if __name__ == "__main__":
    unittest.main()
