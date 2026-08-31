"""Unit tests for FrameForge V3 Prompt Parser."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frameforge.models import ImageAsset
from frameforge.prompt_parser import PromptMappingParser, PromptParseError


class PromptParserTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_json_dict(self):
        json_content = '{"IMG_0001.jpg": "stormy ocean with huge waves", "IMG_0002.jpg": "lighthouse on a cliff"}'
        result = PromptMappingParser.parse_json(json_content)
        self.assertEqual(result["IMG_0001.jpg"], "stormy ocean with huge waves")
        self.assertEqual(result["IMG_0002.jpg"], "lighthouse on a cliff")

    def test_parse_json_array_of_objects(self):
        json_content = '[{"filename": "IMG_0001.jpg", "prompt": "stormy ocean"}, {"filename": "IMG_0002.jpg", "prompt": "lighthouse"}]'
        result = PromptMappingParser.parse_json(json_content)
        self.assertEqual(result["IMG_0001.jpg"], "stormy ocean")
        self.assertEqual(result["IMG_0002.jpg"], "lighthouse")

    def test_parse_csv_with_headers(self):
        csv_content = "filename,prompt\nIMG_0001.jpg,stormy ocean with huge waves\nIMG_0002.jpg,lighthouse on a cliff"
        result = PromptMappingParser.parse_csv(csv_content)
        self.assertEqual(result["IMG_0001.jpg"], "stormy ocean with huge waves")
        self.assertEqual(result["IMG_0002.jpg"], "lighthouse on a cliff")

    def test_parse_csv_without_headers(self):
        csv_content = 'IMG_0001.jpg,"stormy ocean with huge waves"\nIMG_0002.jpg,"lighthouse on a cliff"'
        result = PromptMappingParser.parse_csv(csv_content)
        self.assertEqual(result["IMG_0001.jpg"], "stormy ocean with huge waves")
        self.assertEqual(result["IMG_0002.jpg"], "lighthouse on a cliff")

    def test_apply_prompts_to_assets_with_case_and_extension_tolerance(self):
        assets = [
            ImageAsset("img-1", "path/img_0001.jpg", "IMG_0001.jpg", ""),
            ImageAsset("img-2", "path/IMG_0002.png", "IMG_0002.png", ""),
            ImageAsset("img-3", "path/img_0003.jpeg", "img_0003.jpeg", ""),
        ]
        mapping = {
            "img_0001.jpg": "dramatic storm at sea",
            "img_0002": "tall lighthouse tower beacon",  # stem matching without .png
        }

        stats = PromptMappingParser.apply_prompts_to_assets(assets, mapping)
        self.assertEqual(stats["total_images"], 3)
        self.assertEqual(stats["prompts_mapped"], 2)
        self.assertEqual(stats["prompts_missing"], 1)

        self.assertEqual(assets[0].prompt, "dramatic storm at sea")
        self.assertEqual(assets[0].prompt_source, "supplied_file")
        self.assertEqual(assets[1].prompt, "tall lighthouse tower beacon")
        self.assertEqual(assets[1].prompt_source, "supplied_file")
        self.assertEqual(assets[2].prompt_source, "auto_generated")


if __name__ == "__main__":
    unittest.main()
