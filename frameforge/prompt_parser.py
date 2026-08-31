"""FrameForge V3 Prompt Parser module supporting JSON and CSV mapping files."""
from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("frameforge.prompt_parser")


class PromptParseError(ValueError):
    pass


class PromptMappingParser:
    """Parses JSON or CSV prompt files and maps descriptions to image filenames."""

    @staticmethod
    def _normalize_key(filename: str) -> str:
        """Normalize filename for robust case-insensitive and extension-tolerant lookup."""
        clean = Path(filename.strip().lower()).name
        return clean

    @staticmethod
    def _stem_key(filename: str) -> str:
        """Extract lowercase stem without extension."""
        return Path(filename.strip().lower()).stem

    @classmethod
    def parse_json(cls, content: str | bytes) -> dict[str, str]:
        """Parse JSON text into a {filename: prompt} dictionary."""
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        try:
            data = json.loads(content.strip())
        except Exception as exc:
            raise PromptParseError(f"Invalid JSON prompt mapping: {exc}") from exc

        mapping: dict[str, str] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    mapping[str(k).strip()] = v.strip()
                elif isinstance(v, dict) and "prompt" in v:
                    mapping[str(k).strip()] = str(v["prompt"]).strip()
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    fn = item.get("filename") or item.get("file") or item.get("id") or item.get("name")
                    prompt = item.get("prompt") or item.get("description") or item.get("caption") or ""
                    if fn and prompt:
                        mapping[str(fn).strip()] = str(prompt).strip()
        else:
            raise PromptParseError("JSON prompt format must be an object or array of objects.")

        return mapping

    @classmethod
    def parse_csv(cls, content: str | bytes) -> dict[str, str]:
        """Parse CSV text into a {filename: prompt} dictionary."""
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        mapping: dict[str, str] = {}
        reader = csv.reader(io.StringIO(content.strip()))
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]
        if not rows:
            return mapping

        # Check for header row
        first_row = [c.strip().lower() for c in rows[0]]
        start_idx = 0
        fn_idx = 0
        prompt_idx = 1

        if any(h in first_row for h in ["filename", "file", "image", "id", "name", "prompt", "description", "caption"]):
            start_idx = 1
            if "filename" in first_row:
                fn_idx = first_row.index("filename")
            elif "file" in first_row:
                fn_idx = first_row.index("file")
            elif "image" in first_row:
                fn_idx = first_row.index("image")
            elif "id" in first_row:
                fn_idx = first_row.index("id")

            if "prompt" in first_row:
                prompt_idx = first_row.index("prompt")
            elif "description" in first_row:
                prompt_idx = first_row.index("description")
            elif "caption" in first_row:
                prompt_idx = first_row.index("caption")

        for row in rows[start_idx:]:
            if len(row) > max(fn_idx, prompt_idx):
                fn = row[fn_idx].strip()
                prompt = row[prompt_idx].strip()
                if fn and prompt:
                    mapping[fn] = prompt
            elif len(row) == 2:
                mapping[row[0].strip()] = row[1].strip()

        return mapping

    @classmethod
    def parse_file(cls, file_path: str | Path) -> dict[str, str]:
        """Auto-detect and parse a JSON or CSV prompt file from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise PromptParseError(f"Prompt file not found: {file_path}")

        raw_bytes = path.read_bytes()
        suffix = path.suffix.lower()
        if suffix in {".json", ".js"}:
            return cls.parse_json(raw_bytes)
        elif suffix in {".csv", ".tsv", ".txt"}:
            return cls.parse_csv(raw_bytes)

        # Try JSON first, fallback to CSV
        try:
            return cls.parse_json(raw_bytes)
        except Exception:
            return cls.parse_csv(raw_bytes)

    @classmethod
    def apply_prompts_to_assets(
        cls,
        assets: list[Any],
        prompt_mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Apply mapped prompts to ImageAsset objects with exact and stem fallback matching."""
        # Index mapping by normalized filename and by stem
        exact_map = {cls._normalize_key(k): v for k, v in prompt_mapping.items()}
        stem_map = {cls._stem_key(k): v for k, v in prompt_mapping.items()}

        mapped_count = 0
        missing_count = 0

        for asset in assets:
            fn = getattr(asset, "filename", "")
            norm_fn = cls._normalize_key(fn)
            stem_fn = cls._stem_key(fn)

            if norm_fn in exact_map and exact_map[norm_fn].strip():
                asset.prompt = exact_map[norm_fn].strip()
                asset.prompt_source = "supplied_file"
                mapped_count += 1
            elif stem_fn in stem_map and stem_map[stem_fn].strip():
                asset.prompt = stem_map[stem_fn].strip()
                asset.prompt_source = "supplied_file"
                mapped_count += 1
            elif getattr(asset, "prompt", "").strip():
                asset.prompt_source = "supplied_manual"
                mapped_count += 1
            else:
                asset.prompt_source = "auto_generated"
                missing_count += 1

        return {
            "total_images": len(assets),
            "prompts_mapped": mapped_count,
            "prompts_missing": missing_count,
        }
