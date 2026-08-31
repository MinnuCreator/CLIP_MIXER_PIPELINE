import tempfile
import unittest
from pathlib import Path
from frameforge.config import FrameForgeConfig
from frameforge.image_processor import ImagePreprocessor
from frameforge.models import ImageAsset


class ImageProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = FrameForgeConfig(cache_dir=Path(self.temp_dir.name))
        self.processor = ImagePreprocessor(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_fallback_prompt(self):
        self.assertEqual(
            self.processor.extract_fallback_prompt("Historic_lighthouse_on_cliff_202608210249.jpeg"),
            "Historic lighthouse on cliff",
        )
        self.assertEqual(
            self.processor.extract_fallback_prompt("Rescue_ship_approaching_coastline.jpg"),
            "Rescue ship approaching coastline",
        )

    def test_hamming_distance(self):
        self.assertEqual(self.processor.hamming_distance("0000", "0000"), 0)
        self.assertEqual(self.processor.hamming_distance("0001", "0000"), 1)
        self.assertEqual(self.processor.hamming_distance("000f", "0000"), 4)

    def test_duplicate_detection(self):
        a1 = ImageAsset("img1", "dummy1.jpg", "dummy1.jpg", "prompt 1", phash="ffff000000000000")
        a2 = ImageAsset("img2", "dummy2.jpg", "dummy2.jpg", "prompt 2", phash="ffff000000000001")  # distance 1
        a3 = ImageAsset("img3", "dummy3.jpg", "dummy3.jpg", "prompt 3", phash="0000ffffffffffff")  # distance large
        assets = [a1, a2, a3]
        self.processor.detect_duplicates(assets, max_distance=4)
        self.assertFalse(a1.is_duplicate)
        self.assertTrue(a2.is_duplicate)
        self.assertEqual(a2.duplicate_of, "img1")
        self.assertFalse(a3.is_duplicate)


if __name__ == "__main__":
    unittest.main()
