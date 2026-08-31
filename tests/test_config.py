import os
import unittest
from frameforge.config import FrameForgeConfig


class ConfigTests(unittest.TestCase):
    def test_default_config_values(self):
        cfg = FrameForgeConfig()
        self.assertEqual(cfg.max_images, 700)
        self.assertEqual(cfg.top_k_candidates, 20)
        self.assertEqual(cfg.min_scene_duration, 1.5)
        self.assertEqual(cfg.max_scene_duration, 12.0)
        self.assertEqual(cfg.image_workers, 4)

    def test_env_override(self):
        os.environ["MAX_IMAGES"] = "500"
        os.environ["TOP_K"] = "15"
        try:
            cfg = FrameForgeConfig.from_env()
            self.assertEqual(cfg.max_images, 500)
            self.assertEqual(cfg.top_k_candidates, 15)
        finally:
            del os.environ["MAX_IMAGES"]
            del os.environ["TOP_K"]


if __name__ == "__main__":
    unittest.main()
