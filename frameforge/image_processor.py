"""FrameForge V2 Image Preprocessor & Normalization Engine with Versioned Caching."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from .config import DEFAULT_CONFIG, FrameForgeConfig
from .models import ImageAsset
from .visual import VisualUnderstandingEngine

logger = logging.getLogger("frameforge.image_processor")


class ImagePreprocessor:
    def __init__(self, config: Optional[FrameForgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.cache_dir = Path(self.config.cache_dir) / "v2"
        self.metadata_cache_dir = self.cache_dir / "metadata"
        self.normalized_cache_dir = self.cache_dir / "normalized"
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_cache_dir.mkdir(parents=True, exist_ok=True)
        self.visual_engine = VisualUnderstandingEngine(self.config)

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute streaming SHA-256 hash without loading full file into RAM."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def make_metadata_cache_key(self, file_hash: str) -> str:
        raw = f"{file_hash}_metadata_{self.config.config_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def make_normalized_cache_key(self, file_hash: str, width: int, height: int, fit: str) -> str:
        raw = f"{file_hash}_norm_{width}x{height}_{fit}_{self.config.config_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_phash_and_metrics(file_path: Path) -> dict:
        """Compute perceptual hash (dHash), dimensions, blur score, and brightness."""
        try:
            from PIL import Image, ImageOps, ImageStat

            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img) or img
                width, height = img.size
                img_format = img.format or "JPEG"

                # Perceptual difference hash (dHash 8x8 -> 64-bit)
                gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                pixels = list(gray.getdata())
                diff = []
                for row in range(8):
                    for col in range(8):
                        left = pixels[row * 9 + col]
                        right = pixels[row * 9 + col + 1]
                        diff.append("1" if left > right else "0")
                phash_hex = f"{int(''.join(diff), 2):016x}"

                # Brightness & contrast metrics
                stat = ImageStat.Stat(gray)
                mean_brightness = stat.mean[0] / 255.0
                contrast = stat.stddev[0] / 128.0

                # Lightweight blur score via thumbnail gradient variation
                thumb = gray.resize((64, 64), Image.Resampling.BOX)
                thumb_pixels = list(thumb.getdata())
                grad_sum = 0.0
                for y in range(63):
                    for x in range(63):
                        gx = abs(thumb_pixels[y * 64 + x + 1] - thumb_pixels[y * 64 + x])
                        gy = abs(thumb_pixels[(y + 1) * 64 + x] - thumb_pixels[y * 64 + x])
                        grad_sum += gx + gy
                blur_score = min(1.0, (grad_sum / (64 * 64)) / 30.0)

                # Overall quality score
                res_factor = min(1.0, (width * height) ** 0.5 / 720.0)
                bright_penalty = 0.0
                if mean_brightness < 0.15:
                    bright_penalty = (0.15 - mean_brightness) * 2.0
                elif mean_brightness > 0.92:
                    bright_penalty = (mean_brightness - 0.92) * 2.0

                quality_score = max(
                    0.1,
                    min(
                        1.0,
                        (0.4 * res_factor + 0.3 * blur_score + 0.3 * min(1.0, contrast))
                        - bright_penalty,
                    ),
                )

                return {
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "phash": phash_hex,
                    "blur_score": round(blur_score, 4),
                    "quality_score": round(quality_score, 4),
                    "mean_brightness": round(mean_brightness, 4),
                    "corrupt": False,
                }
        except Exception as exc:
            logger.warning("PIL image metric extraction failed for %s: %s", file_path, exc)
            return {
                "width": 1080,
                "height": 1920,
                "format": "UNKNOWN",
                "phash": "0000000000000000",
                "blur_score": 0.5,
                "quality_score": 0.5,
                "mean_brightness": 0.5,
                "corrupt": True,
            }

    @staticmethod
    def hamming_distance(hex1: str, hex2: str) -> int:
        """Compute bitwise hamming distance between two 16-char hex pHash strings."""
        try:
            n1 = int(hex1, 16)
            n2 = int(hex2, 16)
            return bin(n1 ^ n2).count("1")
        except Exception:
            return 64

    @staticmethod
    def extract_fallback_prompt(filename: str, visual_tags: list[str] | None = None) -> str:
        """Extract clean descriptive tokens from filename and visual attributes."""
        stem = Path(filename).stem
        cleaned = re.sub(r"[_\-\d]{6,}$", "", stem)
        cleaned = re.sub(r"[_\-]+", " ", cleaned).strip()
        cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
        
        # Check if filename is meaningless (e.g. IMG_001, DSC_002, image1)
        is_generic = bool(re.match(r"^(img|image|dsc|photo|picture|frame|asset)[\s_\-\d]*$", cleaned.lower()))
        
        parts = []
        if not is_generic and len(cleaned) > 2:
            parts.append(cleaned)
            
        if visual_tags:
            parts.extend(visual_tags)
            
        if not parts:
            parts.append(f"scene illustration {stem}")
            
        return " ".join(parts)

    def preprocess_single(self, asset: ImageAsset) -> ImageAsset:
        """Preprocess a single image asset with versioned multi-attribute caching."""
        path = Path(asset.file_path)
        if not path.is_file():
            asset.quality_score = 0.0
            return asset

        file_hash = self.compute_file_hash(path)
        asset.file_hash = file_hash
        cache_key = self.make_metadata_cache_key(file_hash)
        cache_file = self.metadata_cache_dir / f"{cache_key}.json"

        if cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                asset.width = cached.get("width", 1080)
                asset.height = cached.get("height", 1920)
                asset.phash = cached.get("phash", "")
                asset.quality_score = cached.get("quality_score", 1.0)
                asset.blur_score = cached.get("blur_score", 1.0)
                asset.metadata.update(cached.get("metadata", {}))
                if not asset.prompt.strip():
                    asset.prompt = cached.get("prompt", self.extract_fallback_prompt(asset.filename, asset.metadata.get("visual_tags", [])))
                return asset
            except Exception:
                pass

        # Compute metrics & visual concepts
        metrics = self.compute_phash_and_metrics(path)
        vis_attrs = self.visual_engine.extract_visual_attributes(path)
        metrics["metadata"] = vis_attrs
        asset.metadata.update(vis_attrs)

        asset.width = metrics["width"]
        asset.height = metrics["height"]
        asset.phash = metrics["phash"]
        asset.quality_score = metrics["quality_score"]
        asset.blur_score = metrics["blur_score"]

        if not asset.prompt.strip():
            asset.prompt = self.extract_fallback_prompt(asset.filename, vis_attrs.get("visual_tags", []))

        metrics["prompt"] = asset.prompt
        try:
            cache_file.write_text(json.dumps(metrics), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not write image metadata cache: %s", exc)

        return asset

    def preprocess_batch(
        self,
        assets: list[ImageAsset],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[ImageAsset]:
        """Preprocess batch of images in parallel with duplicate detection."""
        total = len(assets)
        processed: list[ImageAsset] = [None] * total  # type: ignore

        with ThreadPoolExecutor(max_workers=self.config.image_workers) as executor:
            future_to_idx = {
                executor.submit(self.preprocess_single, asset): idx
                for idx, asset in enumerate(assets)
            }
            completed_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    processed[idx] = future.result()
                except Exception as exc:
                    logger.error("Error preprocessing asset %s: %s", assets[idx].id, exc)
                    processed[idx] = assets[idx]
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total)

        # Detect near-duplicate images via perceptual hash clustering
        self.detect_duplicates(processed)
        return processed

    def detect_duplicates(self, assets: list[ImageAsset], max_distance: int = 4) -> None:
        """Mark near-duplicate images based on pHash hamming distance."""
        for i, first in enumerate(assets):
            if first.is_duplicate or not first.phash:
                continue
            for j in range(i + 1, len(assets)):
                second = assets[j]
                if second.is_duplicate or not second.phash:
                    continue
                if self.hamming_distance(first.phash, second.phash) <= max_distance:
                    second.is_duplicate = True
                    second.duplicate_of = first.id

    def normalize_image(
        self,
        asset: ImageAsset,
        target_width: int = 1080,
        target_height: int = 1920,
        fit: str = "cover",
    ) -> Path:
        """Create or reuse a pre-scaled render-ready image in cache."""
        source_path = Path(asset.file_path)
        if not source_path.is_file():
            return source_path

        file_hash = asset.file_hash or self.compute_file_hash(source_path)
        cache_key = self.make_normalized_cache_key(file_hash, target_width, target_height, fit)
        cached_name = f"{cache_key}.jpg"
        cached_path = self.normalized_cache_dir / cached_name

        if cached_path.is_file() and cached_path.stat().st_size > 0:
            asset.normalized_path = str(cached_path.resolve())
            return cached_path

        try:
            from PIL import Image, ImageOps

            with Image.open(source_path) as img:
                img = ImageOps.exif_transpose(img) or img
                img = img.convert("RGB")
                src_w, src_h = img.size

                if fit == "cover" or fit == "crop":
                    scale = max(target_width / src_w, target_height / src_h)
                    new_w, new_h = int(src_w * scale), int(src_h * scale)
                    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    left = (new_w - target_width) // 2
                    top = (new_h - target_height) // 2
                    cropped = resized.crop((left, top, left + target_width, top + target_height))
                    cropped.save(cached_path, "JPEG", quality=92, optimize=True)
                elif fit == "contain":
                    scale = min(target_width / src_w, target_height / src_h)
                    new_w, new_h = int(src_w * scale), int(src_h * scale)
                    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                    offset_x = (target_width - new_w) // 2
                    offset_y = (target_height - new_h) // 2
                    canvas.paste(resized, (offset_x, offset_y))
                    canvas.save(cached_path, "JPEG", quality=92, optimize=True)
                else:
                    img.resize((target_width, target_height), Image.Resampling.LANCZOS).save(
                        cached_path, "JPEG", quality=92
                    )

            asset.normalized_path = str(cached_path.resolve())
            return cached_path
        except Exception as exc:
            logger.warning("Image normalization failed for %s: %s. Using source.", source_path, exc)
            asset.normalized_path = str(source_path.resolve())
            return source_path
