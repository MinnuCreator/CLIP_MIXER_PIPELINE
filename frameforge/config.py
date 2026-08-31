"""FrameForge V3 Configuration module with 3-Way Multimodal Alignment weights."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from dataclasses import dataclass, field


@dataclass
class FrameForgeConfig:
    # Versioning for cache invalidation
    config_version: str = "3.0.0"
    visual_model_version: str = "clip-vit-base-patch32-v2"
    text_model_version: str = "all-MiniLM-L6-v2-v1"
    whisper_model_version: str = "faster-whisper-base-v1"

    # Image & scalability limits
    max_images: int = 700
    image_workers: int = 4
    embedding_batch_size: int = 64
    top_k_candidates: int = 20

    # Timeline & scene durations (seconds)
    min_scene_duration: float = 1.5
    preferred_scene_duration: float = 3.5
    max_scene_duration: float = 12.0
    pause_split_threshold: float = 0.7

    # V3 Three-Way Multimodal Alignment Weights
    w_voice_image: float = 0.35      # Voiceover narration <-> Image pixel similarity (CLIP)
    w_voice_prompt: float = 0.35     # Voiceover narration <-> Image prompt text similarity
    w_prompt_image: float = 0.10     # Image prompt <-> Image pixel consistency (CLIP)
    w_entity: float = 0.10           # Salient keyword/entity token overlap
    w_quality: float = 0.10          # Image resolution, blur & contrast quality

    # Viterbi Transition & Regularization Weights
    continuity_boost: float = 0.08
    recent_use_penalty: float = 0.12
    duplicate_penalty: float = 0.20
    recent_history_window: int = 4
    dp_transition_weight: float = 0.25
    dp_topic_shift_threshold: float = 0.35

    # Video Render & Sync Configuration
    target_fps: int = 30
    audio_duration_tolerance: float = 0.05  # seconds
    default_render_path: str = "fast"  # 'fast' (concat demuxer) or 'transition' (filter script)

    # Sequential editing mode: preserve user image order (V4)
    sequential_mode: bool = True   # True = sequential (forward-only), False = global reorder (Viterbi)
    min_shot_duration: float = 1.5  # Minimum seconds per visual shot
    target_shot_duration: float = 3.0  # Ideal seconds per visual shot
    max_shot_duration: float = 6.0  # Maximum seconds per visual shot
    min_duration_for_cut: float = 0.8  # Minimum held duration before a cut is allowed
    max_image_uses: int = 2  # Maximum times a single image may appear
    cut_threshold: float = 0.28  # Base threshold for cut-decision score
    sub_segment_gap: float = 0.15  # Min word-timestamp gap (s) to create a sub-beat split

    # V5 LLM / Gemini story planner
    llm_enabled: bool = True  # Enable Gemini story/beat planning
    llm_model: str = "gemini-2.5-flash"  # Gemini model name
    llm_api_key: str = ""  # Set via FRAMEFORGE_LLM_API_KEY env var or .env file
    llm_base_url: str = "https://api.aicredits.in/v1"  # OpenAI-compatible endpoint
    llm_timeout: int = 120  # Seconds for LLM request timeout
    llm_max_retries: int = 2  # Retries on transient LLM failures

    # AI Models & acceleration
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    visual_model: str = "openai/clip-vit-base-patch32"
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "auto"  # 'cuda', 'cpu', or 'auto'

    # Cache & directories
    cache_dir: Path = field(default_factory=lambda: Path(".frameforge-cache"))
    uploads_dir: Path = field(default_factory=lambda: Path("uploads"))
    static_dir: Path = field(default_factory=lambda: Path("static"))
    renders_dir: Path = field(default_factory=lambda: Path("static/renders"))
    db_path: Path = field(default_factory=lambda: Path("frameforge.db"))

    # Render presets
    render_resolutions: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "vertical": (1080, 1920),
            "landscape": (1920, 1080),
            "square": (1080, 1080),
            "preview": (540, 960),
            "preview_vertical": (540, 960),
            "preview_landscape": (960, 540),
        }
    )

    @classmethod
    def from_env(cls) -> FrameForgeConfig:
        """Create config with environment variable overrides."""
        cfg = cls()
        if "MAX_IMAGES" in os.environ:
            cfg.max_images = int(os.environ["MAX_IMAGES"])
        if "TOP_K" in os.environ:
            cfg.top_k_candidates = int(os.environ["TOP_K"])
        if "MIN_SCENE_DURATION" in os.environ:
            cfg.min_scene_duration = float(os.environ["MIN_SCENE_DURATION"])
        if "MAX_SCENE_DURATION" in os.environ:
            cfg.max_scene_duration = float(os.environ["MAX_SCENE_DURATION"])
        if "IMAGE_WORKERS" in os.environ:
            cfg.image_workers = int(os.environ["IMAGE_WORKERS"])
        if "WHISPER_MODEL" in os.environ:
            cfg.whisper_model = os.environ["WHISPER_MODEL"]
        if "DEVICE" in os.environ:
            cfg.device = os.environ["DEVICE"]
        if "RENDER_PATH" in os.environ:
            cfg.default_render_path = os.environ["RENDER_PATH"]
        if "FRAMEFORGE_LLM_API_KEY" in os.environ:
            cfg.llm_api_key = os.environ["FRAMEFORGE_LLM_API_KEY"]
        if "FRAMEFORGE_LLM_MODEL" in os.environ:
            cfg.llm_model = os.environ["FRAMEFORGE_LLM_MODEL"]
        if "W_VOICE_IMAGE" in os.environ:
            cfg.w_voice_image = float(os.environ["W_VOICE_IMAGE"])
        if "W_VOICE_PROMPT" in os.environ:
            cfg.w_voice_prompt = float(os.environ["W_VOICE_PROMPT"])
        if "W_PROMPT_IMAGE" in os.environ:
            cfg.w_prompt_image = float(os.environ["W_PROMPT_IMAGE"])
        return cfg


# Global default configuration instance
DEFAULT_CONFIG = FrameForgeConfig.from_env()
