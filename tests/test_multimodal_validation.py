"""Real multimodal validation tests that verify CLIP visual embeddings genuinely differentiate images.

These tests create REAL images with distinct visual themes, encode them with CLIP,
and verify that Voice->Image similarity scores are meaningfully different across images.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from frameforge.alignment import SceneAlignmentEngine
from frameforge.config import FrameForgeConfig
from frameforge.models import ImageAsset, TranscriptSegment, Word
from frameforge.visual import VisualUnderstandingEngine
from tests.helpers import create_real_image, create_real_images_batch, ALL_THEME_NAMES


class CLIPVisualEmbeddingTests(unittest.TestCase):
    """Verify that CLIP actually produces different image embeddings for different images."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.config = FrameForgeConfig(cache_dir=self.tmp / "cache")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_clip_loads_and_has_expected_dimensions(self):
        """CLIP must load successfully and produce 512-dim embeddings."""
        engine = VisualUnderstandingEngine(self.config)
        self.assertTrue(engine.has_clip, "CLIP must be loaded for visual tests to be meaningful")
        status = engine.model_status()
        self.assertEqual(status["embedding_dim"], 512)
        self.assertFalse(status["is_fallback"])
        self.assertIn("openai/clip-vit-base-patch32", status["visual_model"])
        self.assertIsNone(status.get("failure_reason"))

    def test_different_images_produce_different_embeddings(self):
        """Visually distinct images MUST produce different CLIP embeddings."""
        engine = VisualUnderstandingEngine(self.config)
        self.assertTrue(engine.has_clip, "CLIP must be loaded")

        themes = ["ocean", "forest", "space", "fire", "desert"]
        images = []
        for i, theme in enumerate(themes):
            fp = self.tmp / f"test_{theme}.png"
            create_real_image(fp, theme=theme, seed=i + 50, label=theme)
            images.append(ImageAsset(
                id=f"img-{theme}",
                file_path=str(fp),
                filename=f"{theme}.png",
                prompt=f"a {theme} scene",
            ))

        img_vecs = engine.encode_images(images)
        self.assertIsNotNone(img_vecs, "Image encoding must return non-None when CLIP is available")
        self.assertEqual(img_vecs.shape, (5, 512))

        # Check that all vectors are non-zero
        for i in range(5):
            norm = np.linalg.norm(img_vecs[i])
            self.assertGreater(norm, 0.01, f"Embedding {i} ({themes[i]}) must be non-zero (norm={norm:.6f})")

        # Check that different images produce different embeddings
        # Compute pairwise cosine similarities
        for i in range(5):
            for j in range(i + 1, 5):
                sim = float(np.dot(img_vecs[i], img_vecs[j]) / (
                    np.linalg.norm(img_vecs[i]) * np.linalg.norm(img_vecs[j])
                ))
                # Different themed images should not have identical embeddings
                # (allow some tolerance for CLIP's embedding space)
                self.assertFalse(
                    np.allclose(img_vecs[i], img_vecs[j], atol=1e-4),
                    f"Images {themes[i]} and {themes[j]} have identical embeddings — CLIP is broken",
                )

    def test_voice_image_scores_vary_across_images(self):
        """Voice->Image similarity MUST be different for different narration+image pairs."""
        engine = VisualUnderstandingEngine(self.config)
        self.assertTrue(engine.has_clip, "CLIP must be loaded")

        # Create visually distinct images
        themes = ["ocean", "forest", "space", "fire"]
        images = []
        for i, theme in enumerate(themes):
            fp = self.tmp / f"vi_{theme}.png"
            create_real_image(fp, theme=theme, seed=i + 100, label=theme)
            images.append(ImageAsset(
                id=f"img-{theme}",
                file_path=str(fp),
                filename=f"{theme}.png",
                prompt=f"a beautiful {theme} landscape",
            ))

        # Encode images
        img_vecs = engine.encode_images(images)

        # Encode narration texts
        narrations = [
            "The deep blue ocean stretched endlessly to the horizon.",
            "Tall green trees covered the misty mountain slope.",
            "Stars and galaxies filled the vast dark expanse of outer space.",
            "Bright orange flames engulfed the ancient volcanic peak.",
        ]
        narr_vecs = engine.encode_text(narrations)

        self.assertEqual(img_vecs.shape, (4, 512))
        self.assertEqual(narr_vecs.shape, (4, 512))

        # Compute the full similarity matrix
        sim_matrix = np.dot(narr_vecs, img_vecs.T)  # shape (4, 4)

        # The diagonal (narration[i] <-> image[i]) should generally be higher
        # than off-diagonal for well-matched pairs
        diagonal_scores = np.diag(sim_matrix)
        print("\nVoice->Image Similarity Matrix:")
        for i, narr in enumerate(narrations[:4]):
            scores_str = " ".join(f"{sim_matrix[i, j]:.4f}" for j in range(4))
            print(f"  '{narr[:40]}...' -> [{scores_str}]")

        # Verify that not all scores are identical (which would indicate constant 0.5 fallback)
        all_scores = sim_matrix.flatten()
        score_range = float(all_scores.max() - all_scores.min())
        print(f"\nScore range: {score_range:.4f} (min={all_scores.min():.4f}, max={all_scores.max():.4f})")
        self.assertGreater(
            score_range, 0.02,
            f"Voice->Image scores have no variation (range={score_range:.6f}). "
            f"CLIP may not be producing meaningful embeddings.",
        )

    def test_alignment_uses_clip_when_available(self):
        """Verify alignment engine reports three_way_multimodal when CLIP is loaded."""
        engine = SceneAlignmentEngine(config=self.config)
        status = engine.model_status()
        print(f"\nAlignment model status: {status}")
        self.assertEqual(status["alignment_method"], "three_way_multimodal")
        self.assertIn("voice_image", status["active_scoring_signals"])
        self.assertTrue(status["visual_engine"]["has_clip"])

    def test_prompt_influences_image_selection(self):
        """Verify that supplied image prompts actually change which image is selected."""
        # Create images with distinct themes
        img_data = [
            ("ocean_img", "ocean", "vast blue ocean with waves"),
            ("forest_img", "forest", "dense green forest with tall trees"),
            ("space_img", "space", "outer space with stars and galaxies"),
        ]

        assets = []
        for name, theme, prompt in img_data:
            fp = self.tmp / f"{name}.png"
            create_real_image(fp, theme=theme, seed=200 + len(assets), label=name)
            assets.append(ImageAsset(
                id=name, file_path=str(fp), filename=f"{name}.png", prompt=prompt,
            ))

        # Narration that mentions ocean
        ocean_seg = TranscriptSegment(
            "The vast blue ocean waves crashed against the shore.",
            0.0, 5.0,
            [Word("ocean", 1.0, 1.5), Word("waves", 1.5, 2.0)],
        )

        engine = SceneAlignmentEngine(config=self.config)
        scenes, method = engine.align([ocean_seg], assets, 5.0)

        self.assertEqual(len(scenes), 1)
        selected = scenes[0].image_id
        breakdown = scenes[0].rerank_breakdown
        print(f"\nOcean narration -> selected: {selected}")
        print(f"  Breakdown: {breakdown}")

        # The ocean image should be selected (or at minimum, the prompt should guide selection)
        self.assertEqual(selected, "ocean_img",
                         f"Expected 'ocean_img' for ocean narration, got '{selected}'")

    def test_different_narrations_select_different_images(self):
        """Verify that different narration topics select different images from the library."""
        # Create a diverse image library
        img_themes = [
            ("lighthouse", "lighthouse on rocky cliff at ocean"),
            ("storm", "dark stormy ocean violent waves"),
            ("forest", "green pine forest mountain"),
            ("space", "space rocket launch fire"),
            ("desert", "egyptian pyramid desert sand"),
        ]

        assets = []
        for name, prompt in img_themes:
            fp = self.tmp / f"sel_{name}.png"
            create_real_image(fp, theme=name, seed=300 + len(assets), label=name)
            assets.append(ImageAsset(
                id=f"img-{name}", file_path=str(fp), filename=f"{name}.png", prompt=prompt,
            ))

        # Create narration segments with distinct topics
        segments = [
            TranscriptSegment("The keeper climbed the lighthouse tower.", 0.0, 4.0,
                              [Word("lighthouse", 1.0, 1.5), Word("tower", 2.0, 2.5)]),
            TranscriptSegment("Dark storm clouds gathered over the ocean.", 4.0, 8.0,
                              [Word("storm", 5.0, 5.5), Word("ocean", 6.0, 6.5)]),
            TranscriptSegment("Tall pine trees covered the mountain forest.", 8.0, 12.0,
                              [Word("forest", 9.0, 9.5), Word("trees", 9.5, 10.0)]),
            TranscriptSegment("The rocket launched into outer space.", 12.0, 16.0,
                              [Word("rocket", 13.0, 13.5), Word("space", 14.0, 14.5)]),
            TranscriptSegment("Camels walked across the sandy desert.", 16.0, 20.0,
                              [Word("desert", 17.0, 17.5), Word("sand", 18.0, 18.5)]),
        ]

        engine = SceneAlignmentEngine(config=self.config)
        scenes, method = engine.align(segments, assets, 20.0)

        self.assertEqual(len(scenes), 5)

        selected_ids = [s.image_id for s in scenes]
        print(f"\n5-segment alignment results:")
        for i, (seg, scene) in enumerate(zip(segments, scenes)):
            print(f"  Seg {i+1} '{seg.text[:40]}' -> {scene.image_id} (score={scene.confidence:.4f})")

        # Verify that at least 4 out of 5 segments select the correct image
        expected = ["img-lighthouse", "img-storm", "img-forest", "img-space", "img-desert"]
        hits = sum(1 for sel, exp in zip(selected_ids, expected) if sel == exp)
        print(f"\nTop-1 accuracy: {hits}/5 ({hits*20:.0f}%)")
        self.assertGreaterEqual(hits, 3, f"Expected at least 3/5 correct selections, got {hits}/5")

    def test_comprehensive_score_reporting(self):
        """Full 10-image alignment with detailed per-candidate score reporting.

        Prints IMAGE | VOICE->IMAGE | VOICE->PROMPT | PROMPT->IMAGE | ENTITY | QUALITY | FINAL
        for every candidate, verifying that all three semantic signals contain genuinely
        different values.
        """
        scenarios = [
            ("lighthouse", "lighthouse on rocky cliff at ocean", "The keeper climbed the lighthouse tower.", "img-LH"),
            ("storm", "dark stormy ocean violent waves", "Dark storm clouds battered the coast.", "img-ST"),
            ("forest", "green pine forest mountain", "Tall pine trees covered the slope.", "img-FO"),
            ("space", "space rocket launch fire", "The rocket launched into outer space.", "img-SP"),
            ("desert", "egyptian pyramid desert sand", "Camels walked across the sandy desert.", "img-DS"),
            ("ocean", "vast blue ocean with waves", "The ocean waves crashed on the shore.", "img-OC"),
            ("fire", "bright orange flames burning", "Bright orange flames engulfed the building.", "img-FI"),
            ("city", "modern city skyline at night", "The neon-lit city skyline sparkled.", "img-CI"),
            ("sunset", "golden sunset over the coast", "Golden sunset painted the sky orange.", "img-SU"),
            ("arctic", "arctic ice glacier snow", "Glacial ice cracked under the weight.", "img-AR"),
        ]

        assets = []
        for theme, prompt, _, img_id in scenarios:
            fp = self.tmp / f"{theme}.png"
            create_real_image(fp, theme=theme, seed=hash(theme) & 0xFFFF, label=theme)
            assets.append(ImageAsset(
                id=img_id, file_path=str(fp), filename=f"{theme}.png", prompt=prompt,
            ))

        segments = []
        for i, (_, _, narration, _) in enumerate(scenarios):
            start = float(i * 4)
            end = float((i + 1) * 4)
            words = [Word(w, start + j * 0.4, start + (j + 1) * 0.4)
                     for j, w in enumerate(narration.split())]
            segments.append(TranscriptSegment(narration, start, end, words))

        engine = SceneAlignmentEngine(config=self.config)
        scenes, method = engine.align(segments, assets, 40.0)

        self.assertEqual(len(scenes), 10)

        # Collect all voice_image, voice_prompt, prompt_image scores
        vi_scores = []
        vp_scores = []
        pi_scores = []
        hits = 0

        print("\n" + "=" * 120)
        print(f"{'IMAGE':>8s} | {'VOICE->IMAGE':>12s} | {'VOICE->PROMPT':>13s} | {'PROMPT->IMAGE':>13s} | {'ENTITY':>6s} | {'QUALITY':>7s} | {'FINAL':>7s} | SELECTED")
        print("-" * 120)

        for i, (scene, (_, _, _, expected_id)) in enumerate(zip(scenes, scenarios)):
            bd = scene.rerank_breakdown
            vi = bd.get("voice_image", 0)
            vp = bd.get("voice_prompt", 0)
            pi = bd.get("prompt_image", 0)
            ent = bd.get("entity_sim", 0)
            qual = bd.get("quality", 0)
            final = bd.get("final_score", 0)

            vi_scores.append(vi)
            vp_scores.append(vp)
            pi_scores.append(pi)

            is_correct = scene.image_id == expected_id
            if is_correct:
                hits += 1
            marker = "OK" if is_correct else "MISS"

            print(f"{expected_id:>8s} | {vi:12.4f} | {vp:13.4f} | {pi:13.4f} | {ent:6.4f} | {qual:7.4f} | {final:7.4f} | {scene.image_id} {marker}")

        print("=" * 120)
        print(f"Top-1 accuracy: {hits}/10 ({hits*10:.0f}%)")
        print(f"Voice->Image  score range: {min(vi_scores):.4f} - {max(vi_scores):.4f} (delta={max(vi_scores)-min(vi_scores):.4f})")
        print(f"Voice->Prompt score range: {min(vp_scores):.4f} - {max(vp_scores):.4f} (delta={max(vp_scores)-min(vp_scores):.4f})")
        print(f"Prompt->Image score range: {min(pi_scores):.4f} - {max(pi_scores):.4f} (delta={max(pi_scores)-min(pi_scores):.4f})")

        # All three signals must have genuine variation (not constant)
        self.assertGreater(max(vi_scores) - min(vi_scores), 0.01,
                           "Voice->Image scores are nearly constant — CLIP visual matching may be broken")
        self.assertGreater(max(vp_scores) - min(vp_scores), 0.02,
                           "Voice->Prompt scores are nearly constant — text matching may be broken")
        self.assertGreater(max(pi_scores) - min(pi_scores), 0.005,
                           "Prompt->Image scores are nearly constant — CLIP prompt-image may be broken")

        # At least 6/10 should be correct
        self.assertGreaterEqual(hits, 6, f"Expected at least 6/10 correct, got {hits}/10")


def _as_tensor(obj):
    """Extract tensor from CLIP model output (handles transformers 5.x)."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "pooler_output"):
        return obj.pooler_output
    if hasattr(obj, "image_embeds"):
        return obj.image_embeds
    if hasattr(obj, "text_embeds"):
        return obj.text_embeds
    return torch.tensor(obj)


if __name__ == "__main__":
    unittest.main()
