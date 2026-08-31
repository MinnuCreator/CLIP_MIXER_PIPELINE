"""FRAMEFORGE V3 — REAL End-to-End Production Validation.

700 REAL images + 700 prompts + real voiceover → real CLIP → real Viterbi → real FFmpeg → real MP4.
"""
from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
import tracemalloc as _tm
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---- Image generation (visually distinct, deterministic) --------------------

THEMES = {
    "lighthouse":    {"bg": (180, 160, 90),  "fg": (255, 240, 180), "shapes": "tall"},
    "storm":         {"bg": (30, 30, 50),    "fg": (100, 100, 160), "shapes": "chaos"},
    "ocean":         {"bg": (10, 40, 120),   "fg": (60, 120, 200),  "shapes": "waves"},
    "forest":        {"bg": (20, 80, 30),    "fg": (60, 160, 50),   "shapes": "trees"},
    "desert":        {"bg": (220, 190, 130), "fg": (200, 170, 110), "shapes": "dunes"},
    "fire":          {"bg": (100, 30, 10),   "fg": (255, 120, 40),  "shapes": "flames"},
    "space":         {"bg": (5, 5, 20),      "fg": (200, 200, 255), "shapes": "stars"},
    "city":          {"bg": (40, 40, 60),    "fg": (180, 180, 220), "shapes": "buildings"},
    "arctic":        {"bg": (200, 220, 240), "fg": (240, 255, 255), "shapes": "ice"},
    "savanna":       {"bg": (200, 170, 80),  "fg": (160, 130, 60),  "shapes": "flat"},
    "volcano":       {"bg": (80, 20, 10),    "fg": (220, 80, 20),   "shapes": "erupt"},
    "jungle":        {"bg": (15, 60, 20),    "fg": (40, 140, 40),   "shapes": "dense"},
    "coral_reef":    {"bg": (180, 60, 80),   "fg": (240, 140, 160), "shapes": "bubbles"},
    "mountain":      {"bg": (120, 130, 140), "fg": (200, 210, 220), "shapes": "peaks"},
    "sunset":        {"bg": (180, 80, 40),   "fg": (255, 180, 100), "shapes": "horizon"},
    "neon_night":    {"bg": (10, 10, 30),    "fg": (0, 255, 200),   "shapes": "neon"},
    "autumn":        {"bg": (140, 80, 30),   "fg": (220, 140, 50),  "shapes": "leaves"},
    "waterfall":     {"bg": (30, 80, 60),    "fg": (180, 220, 240), "shapes": "cascade"},
    "northern_lights": {"bg": (5, 15, 30),   "fg": (100, 255, 150), "shapes": "aurora"},
    "rice_paddy":    {"bg": (80, 140, 50),   "fg": (150, 200, 100), "shapes": "rows"},
}

THEME_LIST = list(THEMES.keys())


def _generate_image(idx: int, out_dir: Path) -> Path:
    """Generate a visually distinct image with shapes derived from index."""
    theme_name = THEME_LIST[idx % len(THEME_LIST)]
    palette = THEMES[theme_name]
    rng = np.random.RandomState(idx)
    size = (256, 256)
    img = Image.new("RGB", size, palette["bg"])
    draw = ImageDraw.Draw(img)

    n = rng.randint(8, 20)
    for _ in range(n):
        x1 = rng.randint(0, 230)
        y1 = rng.randint(0, 230)
        x2 = x1 + rng.randint(10, 40)
        y2 = y1 + rng.randint(10, 40)
        c = tuple(rng.randint(max(0, v - 40), min(255, v + 40)) for v in palette["fg"])
        kind = rng.choice(["rect", "ellipse", "line"])
        if kind == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=c, outline=palette["fg"])
        elif kind == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=c, outline=palette["fg"])
        else:
            draw.line([x1, y1, x2, y2], fill=c, width=rng.randint(1, 4))

    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    fp = out_dir / f"e2e_{idx:04d}.png"
    img.save(str(fp), format="PNG")
    return fp


def _generate_prompt(idx: int, theme_name: str) -> str:
    """Generate a descriptive prompt for the given image theme."""
    return f"A cinematic photograph of {theme_name.replace('_', ' ')} scene variation {idx}"


# ---- Whisper transcript simulation (reuse actual if available) ---------------

def _load_or_transcribe_voiceover(audio_path: Path, cache_dir: Path):
    """Transcribe the real voiceover or load cached transcript."""
    from frameforge.transcript import WhisperTranscriber, SemanticSegmenter
    from frameforge.config import FrameForgeConfig

    config = FrameForgeConfig(cache_dir=cache_dir)
    transcriber = WhisperTranscriber(config=config)
    raw_segments, duration = transcriber.transcribe_with_duration(str(audio_path))
    seg = SemanticSegmenter(pause_seconds=config.pause_split_threshold)
    segments = seg.split(raw_segments)
    return raw_segments, segments, duration


# ---- Main E2E validation -----------------------------------------------------

def run_e2e():
    from frameforge.config import FrameForgeConfig
    from frameforge.models import ImageAsset, Scene, Timeline
    from frameforge.image_processor import ImagePreprocessor
    from frameforge.alignment import SceneAlignmentEngine
    from frameforge.validation import validate_timeline
    from frameforge.renderer import FFmpegRenderer

    print("=" * 100)
    print("FRAMEFORGE V3 — REAL END-TO-END PRODUCTION VALIDATION")
    print("=" * 100)

    # Paths
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / ".frameforge-cache"
    e2e_dir = project_root / ".e2e-validation"
    e2e_dir.mkdir(exist_ok=True)
    images_dir = e2e_dir / "images"
    images_dir.mkdir(exist_ok=True)
    audio_path = project_root / "uploads" / "voiceover.mp3"
    output_mp4 = e2e_dir / "e2e_final.mp4"

    assert audio_path.exists(), f"Voiceover not found: {audio_path}"

    config = FrameForgeConfig(cache_dir=cache_dir, min_scene_duration=0.5)

    # ================================================================
    # PHASE 0: Generate 700 real images + 700 prompts
    # ================================================================
    print("\n[PHASE 0] Generating 700 real images + prompts ...")
    t_start = time.perf_counter()
    NUM_IMAGES = 700

    prompt_map = {}
    image_paths = []
    for i in range(NUM_IMAGES):
        fp = _generate_image(i, images_dir)
        image_paths.append(fp)
        theme = THEME_LIST[i % len(THEME_LIST)]
        prompt_map[fp.name] = _generate_prompt(i, theme)

    prompts_json_path = e2e_dir / "prompts.json"
    prompts_json_path.write_text(json.dumps(prompt_map, indent=1), encoding="utf-8")

    t_gen = time.perf_counter() - t_start
    print(f"  Generated {len(image_paths)} images + {len(prompt_map)} prompts in {t_gen:.1f}s")

    # ================================================================
    # PHASE 1: Build assets list
    # ================================================================
    print("\n[PHASE 1] Building asset list ...")
    assets = []
    for i, fp in enumerate(image_paths):
        theme = THEME_LIST[i % len(THEME_LIST)]
        prompt = prompt_map[fp.name]
        assets.append(ImageAsset(
            id=f"image-{i+1}",
            file_path=str(fp.resolve()),
            filename=fp.name,
            prompt=prompt,
            prompt_source="supplied_file",
        ))
    print(f"  {len(assets)} assets ready")

    # ================================================================
    # PHASE 2: Image preprocessing
    # ================================================================
    print("\n[PHASE 2] Preprocessing 700 images ...")
    t_pre = time.perf_counter()
    preprocessor = ImagePreprocessor(config)
    assets = preprocessor.preprocess_batch(assets)
    t_pre_done = time.perf_counter() - t_pre
    hashes_ok = sum(1 for a in assets if a.file_hash)
    phash_ok = sum(1 for a in assets if a.phash)
    quality_scores = [a.quality_score for a in assets]
    print(f"  Preprocessed: {len(assets)} images in {t_pre_done:.1f}s")
    print(f"  Hashes computed: {hashes_ok}, pHashes: {phash_ok}")
    print(f"  Quality score range: {min(quality_scores):.4f} - {max(quality_scores):.4f}")

    # ================================================================
    # PHASE 3: CLIP image embeddings
    # ================================================================
    print("\n[PHASE 3] CLIP image encoding (700 images) ...")
    t_clip_img = time.perf_counter()
    from frameforge.visual import VisualUnderstandingEngine
    vis_engine = VisualUnderstandingEngine(config)
    clip_status = vis_engine.model_status()
    print(f"  CLIP model: {clip_status['visual_model']}")
    print(f"  CLIP loaded: {clip_status['has_clip']}")
    print(f"  Embedding dim: {clip_status['embedding_dim']}")
    print(f"  Device: {clip_status['device']}")
    assert vis_engine.has_clip, f"CLIP FAILED: {clip_status.get('failure_reason')}"

    img_embeddings = vis_engine.encode_images(assets, batch_size=32)
    t_clip_img_done = time.perf_counter() - t_clip_img
    assert img_embeddings is not None, "Image embeddings are None!"
    assert img_embeddings.shape == (NUM_IMAGES, 512), f"Wrong shape: {img_embeddings.shape}"

    norms = np.linalg.norm(img_embeddings, axis=1)
    nonzero = int(np.sum(norms > 0.01))
    identical_count = 0
    for i in range(min(50, NUM_IMAGES)):
        for j in range(i + 1, min(50, NUM_IMAGES)):
            if np.allclose(img_embeddings[i], img_embeddings[j], atol=1e-4):
                identical_count += 1

    print(f"  Image embeddings shape: {img_embeddings.shape}")
    print(f"  Non-zero embeddings: {nonzero}/{NUM_IMAGES}")
    print(f"  Norm range: {norms.min():.4f} - {norms.max():.4f}")
    print(f"  Identical pairs (first 50): {identical_count}")
    print(f"  CLIP image encoding time: {t_clip_img_done:.1f}s")
    assert nonzero == NUM_IMAGES, f"Only {nonzero}/{NUM_IMAGES} non-zero embeddings"

    # ================================================================
    # PHASE 4: CLIP text embeddings (prompts)
    # ================================================================
    print("\n[PHASE 4] CLIP text encoding (700 prompts) ...")
    t_clip_txt = time.perf_counter()
    prompts_list = [a.prompt for a in assets]
    prompt_embeddings = vis_engine.encode_text(prompts_list, batch_size=64)
    t_clip_txt_done = time.perf_counter() - t_clip_txt
    assert prompt_embeddings is not None
    assert prompt_embeddings.shape == (NUM_IMAGES, 512)
    print(f"  Prompt embeddings shape: {prompt_embeddings.shape}")
    print(f"  CLIP text encoding time: {t_clip_txt_done:.1f}s")

    # ================================================================
    # PHASE 5: Whisper transcription
    # ================================================================
    print("\n[PHASE 5] Whisper transcription ...")
    t_whisper = time.perf_counter()
    raw_segments, narration_segments, audio_duration = _load_or_transcribe_voiceover(audio_path, cache_dir)
    t_whisper_done = time.perf_counter() - t_whisper
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Raw segments: {len(raw_segments)}")
    print(f"  Narration segments: {len(narration_segments)}")
    print(f"  Whisper time: {t_whisper_done:.1f}s")

    # ================================================================
    # PHASE 6: CLIP narration text encoding
    # ================================================================
    print("\n[PHASE 6] CLIP narration text encoding ...")
    t_narr = time.perf_counter()
    narr_texts = [s.text for s in narration_segments]
    narr_embeddings = vis_engine.encode_text(narr_texts, batch_size=64)
    t_narr_done = time.perf_counter() - t_narr
    print(f"  Narration embeddings shape: {narr_embeddings.shape}")
    print(f"  Encoding time: {t_narr_done:.1f}s")

    # ================================================================
    # PHASE 7: Cross-modal similarity matrix
    # ================================================================
    print("\n[PHASE 7] Cross-modal similarity analysis ...")
    # Voice -> Image
    vi_matrix = np.dot(narr_embeddings, img_embeddings.T)  # (N_seg, N_img)
    vi_normed = (np.clip(vi_matrix, -1.0, 1.0) + 1.0) / 2.0

    # Voice -> Prompt
    vp_matrix = np.dot(narr_embeddings, prompt_embeddings.T)
    vp_normed = (np.clip(vp_matrix, -1.0, 1.0) + 1.0) / 2.0

    # Prompt -> Image (diagonal)
    pi_diag = np.sum(prompt_embeddings * img_embeddings, axis=1)
    pi_normed = (np.clip(pi_diag, -1.0, 1.0) + 1.0) / 2.0

    print(f"  Voice->Image  range: {vi_normed.min():.4f} - {vi_normed.max():.4f} (delta={vi_normed.max()-vi_normed.min():.4f})")
    print(f"  Voice->Prompt range: {vp_normed.min():.4f} - {vp_normed.max():.4f} (delta={vp_normed.max()-vp_normed.min():.4f})")
    print(f"  Prompt->Image range: {pi_normed.min():.4f} - {pi_normed.max():.4f} (delta={pi_normed.max()-pi_normed.min():.4f})")

    assert vi_normed.max() - vi_normed.min() > 0.01, "Voice->Image scores are constant!"
    assert vp_normed.max() - vp_normed.min() > 0.01, "Voice->Prompt scores are constant!"

    # ================================================================
    # PHASE 8: Multimodal alignment + Viterbi
    # ================================================================
    print("\n[PHASE 8] Three-way multimodal alignment + Viterbi ...")
    t_align = time.perf_counter()
    alignment_engine = SceneAlignmentEngine(config=config)
    align_status = alignment_engine.model_status()
    print(f"  Alignment method: {align_status['alignment_method']}")
    print(f"  Active signals: {align_status['active_scoring_signals']}")

    scenes, method = alignment_engine.align(narration_segments, assets, audio_duration)
    t_align_done = time.perf_counter() - t_align
    print(f"  Scenes produced: {len(scenes)}")
    print(f"  Alignment time: {t_align_done:.1f}s")

    # Collect per-scene scores
    vi_scores = [s.rerank_breakdown.get("voice_image", 0) for s in scenes]
    vp_scores = [s.rerank_breakdown.get("voice_prompt", 0) for s in scenes]
    pi_scores = [s.rerank_breakdown.get("prompt_image", 0) for s in scenes]
    final_scores = [s.rerank_breakdown.get("final_score", 0) for s in scenes]

    print(f"\n  Voice->Image  scene scores: min={min(vi_scores):.4f} max={max(vi_scores):.4f}")
    print(f"  Voice->Prompt scene scores: min={min(vp_scores):.4f} max={max(vp_scores):.4f}")
    print(f"  Prompt->Image scene scores: min={min(pi_scores):.4f} max={max(pi_scores):.4f}")
    print(f"  Final scores:               min={min(final_scores):.4f} max={max(final_scores):.4f}")

    # Scene collapse check
    print(f"\n  Narration segments: {len(narration_segments)}")
    print(f"  Timeline scenes:   {len(scenes)}")
    assert len(scenes) >= len(narration_segments) * 0.8, \
        f"Scene collapse: {len(scenes)} scenes from {len(narration_segments)} segments"

    # Unique images used
    used_ids = [s.image_id for s in scenes]
    unique_used = len(set(used_ids))
    print(f"  Unique images used: {unique_used}/{NUM_IMAGES}")
    print(f"  Repeated images: {len(used_ids) - unique_used}")

    # Score table (first 15 scenes)
    print(f"\n  {'Scene':>8s} | {'Start':>6s} | {'End':>6s} | {'Image':>12s} | {'V>I':>6s} | {'V>P':>6s} | {'P>I':>6s} | {'Final':>6s}")
    print("  " + "-" * 80)
    for s in scenes[:15]:
        bd = s.rerank_breakdown
        print(f"  {s.scene_id:>8s} | {s.start:6.2f} | {s.end:6.2f} | {s.image_id:>12s} | "
              f"{bd.get('voice_image',0):6.4f} | {bd.get('voice_prompt',0):6.4f} | "
              f"{bd.get('prompt_image',0):6.4f} | {bd.get('final_score',0):6.4f}")
    if len(scenes) > 15:
        print(f"  ... ({len(scenes) - 15} more scenes)")

    # ================================================================
    # PHASE 9: Timeline validation
    # ================================================================
    print("\n[PHASE 9] Timeline validation ...")
    timeline = Timeline(str(audio_path), audio_duration, scenes, method, raw_segments)
    validated = validate_timeline(timeline, assets, verify_files=False)
    print(f"  Validated scenes: {len(validated.scenes)}")
    print(f"  Timeline start: {validated.scenes[0].start:.4f}")
    print(f"  Timeline end: {validated.scenes[-1].end:.4f}")
    print(f"  Audio duration: {audio_duration:.4f}")
    print(f"  Duration delta: {abs(validated.scenes[-1].end - audio_duration):.6f}")

    # Check gap-free
    for i in range(len(validated.scenes) - 1):
        gap = validated.scenes[i + 1].start - validated.scenes[i].end
        assert abs(gap) < 0.01, f"Gap/overlap between scene {i} and {i+1}: {gap:.6f}"

    # ================================================================
    # PHASE 10: FFmpeg render
    # ================================================================
    print("\n[PHASE 10] FFmpeg rendering ...")
    t_render = time.perf_counter()
    renderer = FFmpegRenderer("ffmpeg", config=config)
    generated = renderer.render(
        validated,
        assets,
        str(output_mp4.resolve()),
        preset="landscape",
        fit="cover",
        transition="cut",
        render_path="fast",
    )
    t_render_done = time.perf_counter() - t_render
    print(f"  Render time: {t_render_done:.1f}s")
    print(f"  Output: {generated}")

    # ================================================================
    # PHASE 11: ffprobe verification
    # ================================================================
    print("\n[PHASE 11] ffprobe verification ...")
    metadata = renderer.probe(generated)
    print(f"  Video codec: {metadata.get('video_codec', 'unknown')}")
    print(f"  Audio codec: {metadata.get('audio_codec', 'unknown')}")
    print(f"  Resolution: {metadata.get('width', '?')}x{metadata.get('height', '?')}")
    print(f"  FPS: {metadata.get('fps', 'unknown')}")
    print(f"  Video duration: {metadata.get('duration', 'unknown')}s")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Duration delta: {abs(float(metadata.get('duration', 0)) - audio_duration):.4f}s")
    print(f"  File size: {Path(generated).stat().st_size / 1024 / 1024:.1f} MB")

    assert metadata["video_codec"] == "h264", f"Wrong video codec: {metadata['video_codec']}"
    assert metadata["audio_codec"] == "aac", f"Wrong audio codec: {metadata['audio_codec']}"
    assert abs(float(metadata["duration"]) - audio_duration) < 0.2, \
        f"Duration mismatch: {metadata['duration']} vs {audio_duration}"

    # ================================================================
    # PHASE 12: Scene-by-scene content check
    # ================================================================
    print("\n[PHASE 12] Scene-by-scene content check ...")
    print(f"  {'Scene':>8s} | {'Duration':>8s} | {'Image ID':>12s} | {'Narration (first 50 chars)'}")
    print("  " + "-" * 90)
    for s in validated.scenes:
        print(f"  {s.scene_id:>8s} | {s.duration:8.2f} | {s.image_id:>12s} | {s.narration[:50]}")

    # ================================================================
    # TIMING SUMMARY
    # ================================================================
    t_total = time.perf_counter() - t_start
    peak_mem = _get_peak_ram()

    print("\n" + "=" * 100)
    print("FINAL VALIDATION REPORT")
    print("=" * 100)
    print(f"  Number of images:              {NUM_IMAGES}")
    print(f"  Number of prompts:             {len(prompt_map)}")
    print(f"  Audio duration:                {audio_duration:.2f}s")
    print(f"  Narration segments:            {len(narration_segments)}")
    print(f"  Final scenes:                  {len(validated.scenes)}")
    print(f"  Unique images used:            {unique_used}")
    print(f"  Repeated images:               {len(used_ids) - unique_used}")
    print(f"  Voice->Image  range:           {min(vi_scores):.4f} - {max(vi_scores):.4f}")
    print(f"  Voice->Prompt range:           {min(vp_scores):.4f} - {max(vp_scores):.4f}")
    print(f"  Prompt->Image range:           {min(pi_scores):.4f} - {max(pi_scores):.4f}")
    print(f"  CLIP model:                    {clip_status['visual_model']}")
    print(f"  CLIP embedding dim:            {clip_status['embedding_dim']}")
    print(f"  Alignment method:              {method}")
    print(f"  Preprocessing time:            {t_pre_done:.1f}s")
    print(f"  CLIP image encoding time:      {t_clip_img_done:.1f}s")
    print(f"  CLIP text encoding time:       {t_clip_txt_done:.1f}s")
    print(f"  Whisper time:                  {t_whisper_done:.1f}s")
    print(f"  Alignment + Viterbi time:      {t_align_done:.1f}s")
    print(f"  FFmpeg render time:            {t_render_done:.1f}s")
    print(f"  Total cold time:               {t_total:.1f}s")
    print(f"  Peak RAM:                      {peak_mem}")
    print(f"  Output resolution:             {metadata.get('width', '?')}x{metadata.get('height', '?')}")
    print(f"  FPS:                           {metadata.get('fps', '?')}")
    print(f"  Video codec:                   {metadata.get('video_codec', '?')}")
    print(f"  Audio codec:                   {metadata.get('audio_codec', '?')}")
    print(f"  Video duration:                {metadata.get('duration', '?')}s")
    print(f"  Duration delta:                {abs(float(metadata.get('duration', 0)) - audio_duration):.4f}s")
    print(f"  File size:                     {Path(generated).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Final MP4:                     {generated}")
    print("=" * 100)

    return {
        "images": NUM_IMAGES,
        "prompts": len(prompt_map),
        "segments": len(narration_segments),
        "scenes": len(validated.scenes),
        "unique_used": unique_used,
        "audio_duration": audio_duration,
        "video_duration": float(metadata.get("duration", 0)),
        "vi_range": (min(vi_scores), max(vi_scores)),
        "vp_range": (min(vp_scores), max(vp_scores)),
        "pi_range": (min(pi_scores), max(pi_scores)),
        "cold_time": t_total,
        "render_time": t_render_done,
        "peak_ram": peak_mem,
        "mp4_path": str(generated),
        "resolution": f"{metadata.get('width', '?')}x{metadata.get('height', '?')}",
        "fps": metadata.get("fps", "?"),
        "video_codec": metadata.get("video_codec", "?"),
        "audio_codec": metadata.get("audio_codec", "?"),
    }


def _get_peak_ram() -> str:
    try:
        import tracemalloc
        if tracemalloc.is_tracing():
            peak = tracemalloc.get_traced_memory()[1]
            return f"{peak / 1024 / 1024:.1f} MB"
    except Exception:
        pass
    return "N/A"


if __name__ == "__main__":
    tracemalloc.start()
    result = run_e2e()
    print(f"\nJSON: {json.dumps(result, indent=2)}")
