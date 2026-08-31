"""Unit tests for FrameForge V3 Three-Way Multimodal Alignment & Disambiguation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frameforge.alignment import SceneAlignmentEngine
from frameforge.config import FrameForgeConfig
from frameforge.image_processor import ImagePreprocessor
from frameforge.models import ImageAsset, TranscriptSegment, Word
from tests.helpers import create_real_image


class ThreeWayAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = FrameForgeConfig(cache_dir=Path(self.temp_dir.name) / "cache")
        self.engine = SceneAlignmentEngine(config=self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_three_way_prompt_disambiguation(self):
        """Tests that a specific image prompt successfully disambiguates between visually similar candidate images."""
        # Candidate images with distinct visual themes
        td = Path(self.temp_dir.name)
        a_path = create_real_image(td / "a.png", theme="storm", seed=201, label="lighthouse storm")
        b_path = create_real_image(td / "b.png", theme="lighthouse", seed=202, label="lighthouse keeper")
        c_path = create_real_image(td / "c.png", theme="sunset", seed=203, label="sunrise ocean")
        assets = [
            ImageAsset("img-A", str(a_path), "a.jpg", "lighthouse on a cliff during a violent ocean storm"),
            ImageAsset("img-B", str(b_path), "b.jpg", "lighthouse keeper lighting an old lantern inside the tower"),
            ImageAsset("img-C", str(c_path), "c.jpg", "peaceful sunrise over the ocean morning"),
        ]

        segment = TranscriptSegment(
            "The lighthouse keeper carefully lit the lantern inside the tower as the storm continued outside.",
            0.0,
            6.0,
            [],
        )

        scenes, method = self.engine.align([segment], assets, 6.0)

        self.assertEqual(len(scenes), 1)
        # Image B must be strongly selected due to voice-prompt and 3-way alignment
        self.assertEqual(scenes[0].image_id, "img-B")

        bd = scenes[0].rerank_breakdown
        self.assertIn("voice_prompt", bd)
        self.assertIn("prompt_image", bd)
        self.assertIn("final_score", bd)
        self.assertGreater(bd["voice_prompt"], 0.4)

    def test_backward_compatibility_unprompted_images(self):
        """Tests that plain images without supplied prompts receive automatic visual tags and align correctly."""
        td = Path(self.temp_dir.name)
        s1 = create_real_image(td / "s1.png", theme="storm", seed=204, label="storm")
        s2 = create_real_image(td / "s2.png", theme="sunset", seed=205, label="sunrise")
        assets = [
            ImageAsset("img-1", str(s1), "IMG_0001.jpg", "", metadata={"visual_tags": ["storm", "ocean", "dark"]}),
            ImageAsset("img-2", str(s2), "IMG_0002.jpg", "", metadata={"visual_tags": ["sunrise", "golden", "coast"]}),
        ]

        segment = TranscriptSegment("A ferocious dark storm battered the rocky coastline.", 0.0, 5.0, [])
        scenes, method = self.engine.align([segment], assets, 5.0)

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].image_id, "img-1")


if __name__ == "__main__":
    unittest.main()
