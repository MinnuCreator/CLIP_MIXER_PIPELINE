"""FrameForge V3 Visual Understanding, Multimodal CLIP & Prompt-Image Consistency Engine.

Uses raw transformers CLIPModel + CLIPProcessor for reliable text/image encoding.
The sentence-transformers wrapper (clip-ViT-B-32) has a broken tokenizer with
transformers>=5.x that produces identical padding-token sequences for all inputs,
making all text embeddings identical. The raw openai/clip-vit-base-patch32 model
loads and encodes correctly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import DEFAULT_CONFIG, FrameForgeConfig
from .models import ImageAsset

logger = logging.getLogger("frameforge.visual")

def _as_tensor(obj: Any) -> "torch.Tensor":
    """Extract a torch.Tensor from various CLIP output types.

    transformers>=5.x returns BaseModelOutputWithPooling instead of a plain
    Tensor from get_image_features / get_text_features. This normalizer
    handles both cases.
    """
    import torch

    if isinstance(obj, torch.Tensor):
        return obj
    # BaseModelOutputWithPooling has pooler_output
    if hasattr(obj, "pooler_output") and obj.pooler_output is not None:
        return obj.pooler_output
    # DataObject with image_embeds or text_embeds
    if hasattr(obj, "image_embeds"):
        return obj.image_embeds
    if hasattr(obj, "text_embeds"):
        return obj.text_embeds
    # Last resort: try torch.tensor()
    return torch.tensor(obj)


# ---------------------------------------------------------------------------
# CLIP singleton state (module-level to survive GC and allow reuse)
# ---------------------------------------------------------------------------
_clip_model: Any = None
_clip_processor: Any = None
_clip_loaded: bool = False
_clip_failure_reason: str = ""
CLIP_EMBEDDING_DIM: int = 512  # openai/clip-vit-base-patch32 produces 512-dim


def _load_clip(config: FrameForgeConfig) -> bool:
    """Load CLIP model and processor. Returns True on success.

    Strategy:
    1. Try local_files_only=True first (fully offline, fast)
    2. If not cached, try downloading once (first run)
    3. On failure, record the reason and return False

    Uses openai/clip-vit-base-patch32 via transformers directly.
    """
    global _clip_model, _clip_processor, _clip_loaded, _clip_failure_reason

    if _clip_loaded:
        return _clip_model is not None

    if config.visual_model in {"none", "", "disabled"}:
        _clip_failure_reason = "visual_model is disabled in config"
        _clip_loaded = True
        return False

    model_name = config.visual_model  # "openai/clip-vit-base-patch32"

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Attempt local-first loading
        try:
            _clip_processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            _clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
            _clip_model.eval()
            _clip_loaded = True
            _clip_failure_reason = ""
            logger.info(
                "CLIP loaded from local cache: %s (device=%s, dim=%d)",
                model_name, device, CLIP_EMBEDDING_DIM,
            )
            return True
        except Exception:
            pass

        # First run: download from HuggingFace Hub
        try:
            _clip_processor = CLIPProcessor.from_pretrained(model_name)
            _clip_model = CLIPModel.from_pretrained(model_name)
            _clip_model.eval()
            _clip_loaded = True
            _clip_failure_reason = ""
            logger.info(
                "CLIP downloaded and loaded: %s (device=%s, dim=%d)",
                model_name, device, CLIP_EMBEDDING_DIM,
            )
            return True
        except Exception as exc2:
            _clip_failure_reason = f"Model load failed: {exc2}"
            logger.warning("CLIP model load failed: %s", exc2)

    except ImportError as exc:
        _clip_failure_reason = f"transformers/torch not installed: {exc}"
        logger.warning("CLIP unavailable: %s", _clip_failure_reason)
    except Exception as exc:
        _clip_failure_reason = f"Unexpected error: {exc}"
        logger.warning("CLIP load error: %s", exc)

    _clip_model = None
    _clip_processor = None
    _clip_loaded = True
    return False


def clip_available() -> bool:
    """Check if CLIP is loaded and available."""
    return _clip_model is not None and _clip_processor is not None


class VisualUnderstandingEngine:
    """Multimodal vision-language understanding engine.

    Uses raw transformers CLIPModel for genuine cross-modal encoding.
    Supports direct visual pixel embeddings, automated visual attribute extraction,
    and prompt-image consistency scoring for 3-way alignment.
    """
    _instance: Optional[VisualUnderstandingEngine] = None

    def __new__(cls, config: Optional[FrameForgeConfig] = None) -> VisualUnderstandingEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine(config or DEFAULT_CONFIG)
        elif config is not None:
            cls._instance.config = config
            cls._instance.cache_dir = Path(config.cache_dir) / "v3" / "visual_embeddings"
            cls._instance.cache_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    def _init_engine(self, config: FrameForgeConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir) / "v3" / "visual_embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _load_clip(self.config)

    # ---- Model status -------------------------------------------------------

    @property
    def has_clip(self) -> bool:
        return clip_available()

    def model_status(self) -> dict[str, Any]:
        """Return diagnostics about the current CLIP model state."""
        device = self.config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return {
            "visual_model": self.config.visual_model,
            "has_clip": self.has_clip,
            "is_fallback": not self.has_clip,
            "clip_loaded": _clip_loaded,
            "device": device,
            "embedding_dim": CLIP_EMBEDDING_DIM if self.has_clip else 0,
            "failure_reason": _clip_failure_reason or None,
            "active_signals": ["voice_image", "prompt_image"] if self.has_clip else [],
        }

    # ---- Cache helpers ------------------------------------------------------

    def make_cache_key(self, asset_hash: str) -> str:
        raw = f"{asset_hash}_visual_{self.config.visual_model_version}_{self.config.config_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ---- Visual attribute extraction (pixel-based, no CLIP needed) ----------

    def extract_visual_attributes(self, file_path: Path) -> dict[str, Any]:
        """Extract visual concepts, colors, and lighting attributes from image pixels."""
        try:
            from PIL import Image, ImageStat

            with Image.open(file_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                thumb = img.resize((64, 64))
                stat = ImageStat.Stat(thumb)
                r, g, b = stat.mean[0], stat.mean[1], stat.mean[2]
                brightness = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0

                tags: list[str] = []
                if brightness < 0.25:
                    tags.extend(["night", "dark", "shadow", "storm"])
                elif brightness > 0.70:
                    tags.extend(["bright", "daylight", "sunny", "clear"])
                elif brightness > 0.45 and r > b * 1.2:
                    tags.extend(["sunrise", "sunset", "golden", "warm"])

                if b > r * 1.25 and b > g * 1.1:
                    tags.extend(["ocean", "sea", "water", "sky", "blue"])
                elif g > r * 1.2 and g > b * 1.1:
                    tags.extend(["forest", "nature", "green", "landscape"])
                elif r > g * 1.3 and r > b * 1.3:
                    tags.extend(["warm", "fire", "beacon", "light", "red"])

                return {
                    "mean_brightness": round(brightness, 4),
                    "dominant_rgb": (int(r), int(g), int(b)),
                    "visual_tags": list(set(tags)),
                }
        except Exception:
            return {"mean_brightness": 0.5, "dominant_rgb": (128, 128, 128), "visual_tags": []}

    # ---- CLIP image encoding ------------------------------------------------

    def encode_images(self, assets: list[ImageAsset], batch_size: int = 32) -> Optional[np.ndarray]:
        """Encode images into the CLIP visual embedding space.

        Returns an (N, 512) float32 normalized numpy array, or None if CLIP is unavailable.
        """
        if not self.has_clip:
            logger.warning("CLIP unavailable — visual image embeddings will not be computed.")
            return None

        embeddings: list[Optional[np.ndarray]] = [None] * len(assets)
        uncached_indices: list[int] = []
        uncached_images: list[Any] = []
        uncached_keys: list[str] = []

        for idx, asset in enumerate(assets):
            key = self.make_cache_key(asset.file_hash or hashlib.sha256(asset.filename.encode()).hexdigest())
            cache_file = self.cache_dir / f"{key}.json"

            if cache_file.is_file():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    embeddings[idx] = np.array(data, dtype=np.float32)
                    continue
                except Exception:
                    pass

            try:
                from PIL import Image
                img = Image.open(asset.file_path).convert("RGB")
                uncached_indices.append(idx)
                uncached_images.append(img)
                uncached_keys.append(key)
            except Exception as exc:
                logger.warning("Could not open image for visual embedding %s: %s", asset.file_path, exc)

        if uncached_images:
            try:
                # Process in batches
                for batch_start in range(0, len(uncached_images), batch_size):
                    batch_end = min(batch_start + batch_size, len(uncached_images))
                    batch_imgs = uncached_images[batch_start:batch_end]
                    batch_keys = uncached_keys[batch_start:batch_end]
                    batch_globals = uncached_indices[batch_start:batch_end]

                    import torch
                    inputs = _clip_processor(images=batch_imgs, return_tensors="pt")
                    with torch.no_grad():
                        img_features = _clip_model.get_image_features(**inputs)
                    # Extract tensor from output (handles both Tensor and DataObject)
                    img_features = _as_tensor(img_features)
                    # Normalize
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    vecs = img_features.cpu().numpy().astype(np.float32)

                    for local_idx, global_idx in enumerate(batch_globals):
                        vec = vecs[local_idx]
                        embeddings[global_idx] = vec
                        k = batch_keys[local_idx]
                        try:
                            (self.cache_dir / f"{k}.json").write_text(json.dumps(vec.tolist()), encoding="utf-8")
                        except Exception as exc:
                            logger.warning("Failed to write visual embedding cache: %s", exc)
            except Exception as exc:
                logger.warning("CLIP batch image encoding failed: %s", exc)

        for idx in range(len(assets)):
            if embeddings[idx] is None:
                embeddings[idx] = np.zeros(CLIP_EMBEDDING_DIM, dtype=np.float32)

        return np.array(embeddings, dtype=np.float32)

    # ---- CLIP text encoding -------------------------------------------------

    def _make_text_cache_key(self, text: str) -> str:
        raw = f"{text.strip()}_clip_text_{self.config.visual_model_version}_{self.config.config_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def encode_text(self, texts: list[str], batch_size: int = 64) -> Optional[np.ndarray]:
        """Encode text into the CLIP shared embedding space.

        Returns an (N, 512) float32 normalized numpy array, or None if CLIP is unavailable.
        """
        if not self.has_clip:
            return None

        embeddings: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check disk cache
        for idx, text in enumerate(texts):
            key = self._make_text_cache_key(text)
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.is_file():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    embeddings[idx] = np.array(data, dtype=np.float32)
                    continue
                except Exception:
                    pass
            uncached_indices.append(idx)
            uncached_texts.append(text)

        # Batch encode any uncached items
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                batch_globals = uncached_indices[batch_start:batch_end]
                try:
                    import torch
                    inputs = _clip_processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        txt_features = _clip_model.get_text_features(**inputs)
                    txt_features = _as_tensor(txt_features)
                    txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
                    vecs = txt_features.cpu().numpy().astype(np.float32)
                    for local_idx, global_idx in enumerate(batch_globals):
                        vec = vecs[local_idx]
                        embeddings[global_idx] = vec
                        # Cache to disk
                        key = self._make_text_cache_key(uncached_texts[batch_start + local_idx])
                        try:
                            (self.cache_dir / f"{key}.json").write_text(json.dumps(vec.tolist()), encoding="utf-8")
                        except Exception as exc:
                            logger.warning("Failed to write CLIP text embedding cache: %s", exc)
                except Exception as exc:
                    logger.warning("CLIP text encoding failed for batch: %s", exc)
                    for local_idx, global_idx in enumerate(batch_globals):
                        embeddings[global_idx] = np.zeros(CLIP_EMBEDDING_DIM, dtype=np.float32)

        # Fill any remaining None entries with zero vectors
        for idx in range(len(texts)):
            if embeddings[idx] is None:
                embeddings[idx] = np.zeros(CLIP_EMBEDDING_DIM, dtype=np.float32)

        return np.array(embeddings, dtype=np.float32)

    # ---- Prompt-Image consistency -------------------------------------------

    def compute_prompt_image_consistency(self, assets: list[ImageAsset], batch_size: int = 64) -> list[float]:
        """Compute semantic consistency between image prompts and actual image pixels (Prompt <-> Image).

        Uses the CLIP shared embedding space to measure whether the prompt text
        describes what the image actually contains.
        """
        if self.has_clip:
            try:
                prompts = [a.prompt for a in assets]
                img_vecs = self.encode_images(assets, batch_size=batch_size)
                prompt_vecs = self.encode_text(prompts, batch_size=batch_size)
                if img_vecs is not None and prompt_vecs is not None:
                    dots = np.sum(img_vecs * prompt_vecs, axis=1)
                    normalized_scores = (np.clip(dots, -1.0, 1.0) + 1.0) / 2.0
                    return [round(float(s), 4) for s in normalized_scores]
            except Exception as exc:
                logger.warning("CLIP prompt-image consistency computation failed: %s", exc)

        # Fallback: visual attribute overlap
        scores: list[float] = []
        for asset in assets:
            vis_tags = set(asset.metadata.get("visual_tags", []))
            prompt_words = set(asset.prompt.lower().split())
            if vis_tags and prompt_words:
                overlap = len(vis_tags & prompt_words) / max(1, len(vis_tags))
                scores.append(round(min(1.0, 0.4 + 0.6 * overlap), 4))
            else:
                scores.append(0.5)
        return scores
