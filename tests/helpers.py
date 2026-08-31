"""Shared test helpers for generating real image files used by FrameForge test suites."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# Deterministic seed so tests are reproducible
_SEED = 42

# Visual themes with distinct color palettes so CLIP embeddings differ
THEMES: dict[str, dict] = {
    "ocean": {"bg": (10, 40, 120), "fg": (200, 220, 255), "accent": (0, 100, 200)},
    "storm": {"bg": (30, 30, 40), "fg": (180, 180, 200), "accent": (80, 80, 120)},
    "lighthouse": {"bg": (200, 180, 100), "fg": (255, 240, 180), "accent": (255, 200, 50)},
    "forest": {"bg": (20, 80, 30), "fg": (100, 200, 80), "accent": (40, 140, 50)},
    "desert": {"bg": (210, 180, 120), "fg": (240, 220, 160), "accent": (180, 140, 80)},
    "space": {"bg": (5, 5, 20), "fg": (200, 200, 255), "accent": (80, 80, 200)},
    "fire": {"bg": (120, 30, 10), "fg": (255, 160, 50), "accent": (255, 80, 20)},
    "arctic": {"bg": (200, 220, 240), "fg": (255, 255, 255), "accent": (150, 200, 230)},
    "sunset": {"bg": (180, 80, 40), "fg": (255, 180, 100), "accent": (200, 100, 60)},
    "city": {"bg": (40, 40, 60), "fg": (100, 100, 140), "accent": (180, 180, 220)},
    "jungle": {"bg": (15, 60, 20), "fg": (60, 160, 40), "accent": (30, 120, 30)},
    "volcano": {"bg": (80, 20, 10), "fg": (220, 80, 20), "accent": (255, 120, 40)},
    "coral": {"bg": (180, 60, 80), "fg": (240, 140, 160), "accent": (200, 100, 120)},
    "savanna": {"bg": (200, 170, 80), "fg": (230, 200, 120), "accent": (160, 130, 60)},
    "neon": {"bg": (10, 10, 30), "fg": (0, 255, 200), "accent": (255, 0, 200)},
    "autumn": {"bg": (140, 80, 30), "fg": (220, 140, 50), "accent": (180, 60, 20)},
}

ALL_THEME_NAMES = list(THEMES.keys())


def _rng(seed: int = _SEED) -> random.Random:
    return random.Random(seed)


def create_real_image(
    path: Path,
    theme: str = "ocean",
    size: tuple[int, int] = (256, 256),
    seed: int | None = None,
    label: str = "",
) -> Path:
    """Generate a real, visually-distinct JPEG/PNG image with shapes and color palette.

    Returns the path to the written file.
    """
    rng = _rng(seed if seed is not None else hash(theme + str(size) + label) & 0xFFFFFFFF)
    palette = THEMES.get(theme, THEMES["ocean"])

    img = Image.new("RGB", size, palette["bg"])
    draw = ImageDraw.Draw(img)

    # Draw several random shapes using the palette
    num_shapes = rng.randint(5, 15)
    for _ in range(num_shapes):
        shape_type = rng.choice(["rect", "ellipse", "line", "arc"])
        x1 = rng.randint(0, size[0] - 10)
        y1 = rng.randint(0, size[1] - 10)
        x2 = x1 + rng.randint(10, size[0] // 2)
        y2 = y1 + rng.randint(10, size[1] // 2)
        color = rng.choice([palette["fg"], palette["accent"], palette["bg"]])

        if shape_type == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=palette["fg"])
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=palette["accent"])
        elif shape_type == "line":
            draw.line([x1, y1, x2, y2], fill=color, width=rng.randint(1, 4))
        elif shape_type == "arc":
            draw.arc([x1, y1, x2, y2], start=0, end=rng.randint(90, 270), fill=color, width=2)

    # Apply a slight blur for naturalistic look
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Draw text label if provided
    if label:
        try:
            draw2 = ImageDraw.Draw(img)
            draw2.text((10, size[1] - 30), label, fill=palette["fg"])
        except Exception:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def create_real_images_batch(
    directory: Path,
    count: int,
    themes: list[str] | None = None,
    prefix: str = "img",
    size: tuple[int, int] = (256, 256),
) -> list[Path]:
    """Generate a batch of visually-distinct real images.

    Returns list of file paths in order.
    """
    if themes is None:
        themes = ALL_THEME_NAMES
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        theme = themes[i % len(themes)]
        fp = directory / f"{prefix}_{i:04d}.png"
        create_real_image(fp, theme=theme, size=size, seed=i, label=f"{prefix}_{i:04d}")
        paths.append(fp)
    return paths


def create_audio_stub(path: Path) -> Path:
    """Create a minimal valid WAV file (silent, 1 second) for tests that need audio file existence."""
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1 second of silence
        wf.writeframes(b"\x00\x00" * 16000)
    return path
