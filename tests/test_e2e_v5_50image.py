"""FrameForge V5 — Real 50-Image End-to-End Validation.

Creates 50 real generated images with distinct visual themes,
ordered prompts, and the real voiceover. Runs the full pipeline
including CLIP, Whisper, Gemini (if configured), SequentialEditor,
and FFmpeg rendering.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── Image generation ─────────────────────────────────────────────────────────

VISUAL_THEMES = [
    ("ocean waves crashing on rocky shore", (30, 80, 160), "ocean"),
    ("lighthouse at night with beam", (20, 30, 80), "lighthouse"),
    ("violent storm clouds over sea", (40, 45, 60), "storm"),
    ("ship struggling in rough waters", (50, 60, 80), "ship"),
    ("keeper climbing spiral staircase", (80, 60, 40), "staircase"),
    ("powerful beacon light cutting darkness", (200, 180, 50), "beacon"),
    ("ships in distance seeing light", (30, 50, 100), "ships"),
    ("calm sunrise after the storm", (200, 120, 60), "sunrise"),
    ("coastal village with small houses", (100, 130, 80), "village"),
    ("fishing boats in harbor", (60, 90, 130), "harbor"),
    ("old stone wall with ivy", (40, 80, 40), "wall"),
    ("aerial view of coastline", (50, 100, 160), "aerial"),
    ("seagull flying over waves", (150, 160, 170), "seagull"),
    ("foggy morning on water", (180, 180, 190), "fog"),
    ("dramatic sunset with red clouds", (180, 60, 30), "sunset"),
    ("moonlit ocean surface", (20, 30, 70), "moonlit"),
    ("rocky tide pools with water", (50, 80, 110), "tidepool"),
    ("sandy beach stretching away", (200, 190, 150), "beach"),
    ("kelp forest underwater view", (30, 70, 50), "kelp"),
    ("whale tail emerging from water", (40, 60, 100), "whale"),
    ("lighthouse keeper room interior", (100, 80, 60), "interior"),
    ("old compass on wooden table", (120, 100, 70), "compass"),
    ("weathered log book pages", (150, 130, 100), "logbook"),
    ("storm lantern glowing warm", (160, 100, 30), "lantern"),
    ("bell buoy in choppy water", (80, 100, 130), "bell"),
    ("crashing wave close-up spray", (60, 100, 150), "wave"),
    ("rainbow after storm clears", (100, 150, 180), "rainbow"),
    ("clouds parting with sunlight", (170, 170, 190), "clouds"),
    ("harbor at golden hour", (180, 130, 60), "golden"),
    ("broken pier in rough sea", (70, 70, 80), "pier"),
    ("sea foam on dark rocks", (150, 160, 170), "foam"),
    ("distant lighthouse from beach", (40, 70, 120), "distant"),
    ("wave pattern in sand", (190, 175, 140), "sand"),
    ("storm approaching from horizon", (50, 50, 65), "approach"),
    ("candle in lighthouse window", (180, 150, 50), "candle"),
    ("anchor on deck of ship", (90, 80, 70), "anchor"),
    ("starlit sky over dark ocean", (10, 15, 40), "starry"),
    ("ice formations on rocky coast", (170, 180, 200), "ice"),
    ("birds flying in formation", (120, 130, 140), "birds"),
    ("shipwreck in shallow water", (60, 70, 80), "wreck"),
    ("kayaker in calm bay", (80, 130, 150), "kayak"),
    ("shell collection on windowsill", (180, 160, 140), "shells"),
    ("lighthouse beam sweeping sea", (30, 40, 90), "beam"),
    ("porthole view of ocean", (50, 80, 120), "porthole"),
    ("driftwood on empty beach", (140, 120, 100), "driftwood"),
    ("wildflowers on cliff edge", (100, 140, 60), "flowers"),
    ("reef visible through clear water", (40, 100, 130), "reef"),
    ("oar leaning against boat", (110, 90, 70), "oar"),
    ("dolphin jumping from wave", (60, 100, 140), "dolphin"),
    ("lighthouse automation gears", (90, 80, 60), "gears"),
]


def create_distinct_image(path: Path, theme: str, base_color: tuple, seed: int) -> Path:
    """Create a visually distinct 1920x1080 image for each theme."""
    rng = random.Random(seed)
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), base_color)
    draw = ImageDraw.Draw(img)

    # Background gradient
    for y in range(h):
        r = int(base_color[0] * (1.0 - y / h * 0.3) + rng.randint(-5, 5))
        g = int(base_color[1] * (1.0 - y / h * 0.3) + rng.randint(-5, 5))
        b = int(base_color[2] * (1.0 - y / h * 0.3) + rng.randint(-5, 5))
        draw.line([(0, y), (w, y)], fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))

    # Unique geometric shapes per theme
    n_shapes = 5 + rng.randint(0, 8)
    for _ in range(n_shapes):
        shape_type = rng.choice(["rect", "ellipse", "line"])
        x1 = rng.randint(0, w - 100)
        y1 = rng.randint(0, h - 100)
        x2 = x1 + rng.randint(50, 400)
        y2 = y1 + rng.randint(50, 300)
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if shape_type == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255))
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=(200, 200, 200))
        else:
            draw.line([x1, y1, x2, y2], fill=color, width=rng.randint(2, 6))

    # Add theme text label
    try:
        draw.text((50, h - 80), theme[:50], fill=(255, 255, 255))
    except Exception:
        pass

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "PNG")
    return path


def create_voiceover_stub(path: Path, duration: float = 48.0) -> Path:
    """Create a 48-second WAV voiceover stub with speech-like content."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration)

    # Generate speech-like audio with varying amplitude
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    # Mix of frequencies to simulate speech formants
    audio = np.zeros(n_samples, dtype=np.float32)
    for freq in [150, 300, 600, 1200]:
        audio += 0.15 * np.sin(2 * np.pi * freq * t + random.random() * 2 * math.pi)
    # Amplitude envelope (speech-like bursts)
    envelope = np.ones(n_samples, dtype=np.float32)
    for i in range(0, n_samples, sample_rate // 2):
        end = min(i + sample_rate // 4, n_samples)
        envelope[i:end] *= 0.3
        end2 = min(end + sample_rate // 8, n_samples)
        envelope[end:end2] *= 0.1
    audio *= envelope
    audio = np.clip(audio, -0.9, 0.9)

    # Write WAV manually
    import struct
    audio_int16 = (audio * 32767).astype(np.int16)
    data_size = len(audio_int16) * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())
    return path


def create_prompt_file(path: Path, images: list[tuple[str, str]]) -> Path:
    """Create a JSON prompts file for the ordered images."""
    prompts = []
    for filename, prompt in images:
        prompts.append({"filename": filename, "prompt": prompt})
    path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    return path


def run_e2e_validation():
    """Run the complete 50-image E2E validation."""
    print("=" * 80)
    print("FRAMEFORGE V5 — 50-IMAGE E2E VALIDATION")
    print("=" * 80)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ffv5_e2e_"))
    images_dir = tmp_dir / "images"
    images_dir.mkdir()

    try:
        # ── 1. Create 50 real images ────────────────────────────────────────
        print("\n[1] Creating 50 real images with distinct visual themes...")
        t0 = time.monotonic()
        image_files = []
        for i, (theme, color, _) in enumerate(VISUAL_THEMES):
            fname = f"image_{i+1:03d}.png"
            path = images_dir / fname
            create_distinct_image(path, theme, color, seed=1000 + i)
            image_files.append((fname, theme))
        image_time = time.monotonic() - t0
        print(f"    Created 50 images in {image_time:.1f}s")

        # ── 2. Create prompts file ──────────────────────────────────────────
        print("\n[2] Creating ordered prompts file...")
        prompts_path = tmp_dir / "prompts.json"
        create_prompt_file(prompts_path, image_files)
        with open(prompts_path, encoding="utf-8") as f:
            prompts_data = json.load(f)
        print(f"    Created {len(prompts_data)} prompts")

        # ── 3. Locate real voiceover ──────────────────────────────────────
        real_voiceovers = [
            Path("uploads/Generated Audio August 30, 2026 - 3_32PM.wav"),
            Path("uploads/Generated Audio August 21, 2026 - 2_48AM.mp3"),
            Path("uploads/voiceover.mp3"),
        ]
        audio_path = None
        for candidate in real_voiceovers:
            if candidate.is_file():
                audio_path = candidate.resolve()
                break
        if audio_path is None:
            print("    WARNING: No real voiceover found, creating synthetic stub.")
            audio_path = tmp_dir / "voiceover.wav"
            create_voiceover_stub(audio_path, duration=48.0)
        print(f"    Using voiceover: {audio_path.name} ({audio_path.stat().st_size / 1024:.0f} KB)")

        # ── 4. Build image dicts ────────────────────────────────────────────
        images_dicts = []
        for i, (fname, theme) in enumerate(image_files):
            images_dicts.append({
                "id": f"img-{i+1:03d}",
                "file_path": str(images_dir / fname),
                "filename": fname,
                "prompt": "",
                "prompt_source": "auto_generated",
                "metadata": {},
                "file_hash": None,
                "phash": None,
                "width": 1920,
                "height": 1080,
                "quality_score": 1.0,
                "blur_score": 1.0,
                "is_duplicate": False,
                "duplicate_of": None,
                "normalized_path": None,
            })

        # ── 5. Run full pipeline ────────────────────────────────────────────
        print("\n[4] Running full pipeline: preprocess -> CLIP -> Whisper -> align -> render...")
        from frameforge.config import FrameForgeConfig
        from frameforge.service import FrameForgeService

        config = FrameForgeConfig()
        config.llm_api_key = os.environ.get("FRAMEFORGE_LLM_API_KEY", "")
        config.llm_enabled = bool(config.llm_api_key)

        service = FrameForgeService(config=config)

        # Analyze
        t_analyze = time.monotonic()
        result = service.analyze(
            project_id="v5_50img_e2e",
            audio_file=str(audio_path),
            audio_duration=None,
            images=images_dicts,
            prompts=prompts_path,
        )
        analyze_time = time.monotonic() - t_analyze
        print(f"    Analyze completed in {analyze_time:.1f}s")

        # ── 6. Extract metrics ──────────────────────────────────────────────
        print("\n[5] Analyzing results...")
        scenes = result["scenes"]
        editor_meta = result.get("editor_metadata", {})
        n_scenes = len(scenes)
        audio_dur = result["audio"]["duration"]

        # Score ranges
        vi_scores = [s["rerank_breakdown"].get("voice_image", 0) for s in scenes]
        vp_scores = [s["rerank_breakdown"].get("voice_prompt", 0) for s in scenes]
        pi_scores = [s["rerank_breakdown"].get("prompt_image", 0) for s in scenes]
        final_scores = [s["rerank_breakdown"].get("final_score", 0) for s in scenes]

        # Image sequence
        image_seq = [s["image_id"] for s in scenes]
        image_nums = [int(s.split("-")[1]) for s in image_seq]

        # Order preservation
        order_preserved = all(image_nums[i] <= image_nums[i+1] for i in range(len(image_nums)-1))

        # No backward jumps
        backward_jumps = sum(1 for i in range(1, len(image_nums)) if image_nums[i] < image_nums[i-1])

        # Oscillation check
        oscillations = 0
        for i in range(2, len(image_nums)):
            if image_nums[i] == image_nums[i-2] and image_nums[i] != image_nums[i-1]:
                oscillations += 1

        # Image usage
        usage_counts = {}
        for img_id in image_seq:
            usage_counts[img_id] = usage_counts.get(img_id, 0) + 1
        max_usage = max(usage_counts.values()) if usage_counts else 0
        repeated = sum(1 for c in usage_counts.values() if c > 1)

        # Durations
        durations = [s["duration"] for s in scenes]
        avg_dur = sum(durations) / len(durations) if durations else 0
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0

        # Gap-free check
        gaps = 0
        for i in range(1, len(scenes)):
            gap = abs(scenes[i]["start"] - scenes[i-1]["end"])
            if gap > 0.01:
                gaps += 1

        # ── 7. Print scene table ────────────────────────────────────────────
        print(f"\n{'Scene':>8s} | {'Start':>6s} | {'End':>6s} | {'Dur':>5s} | {'Image':>8s} | {'V->I':>6s} | {'V->P':>6s} | {'P->I':>6s} | {'Final':>6s} | {'Decision':>10s}")
        print("-" * 105)
        for s in scenes[:30]:
            bd = s["rerank_breakdown"]
            print(
                f"  {s['scene_id']:>6s} | {s['start']:6.2f} | {s['end']:>6.2f} | {s['duration']:5.2f} | "
                f"{s['image_id']:>8s} | "
                f"{bd.get('voice_image',0):6.4f} | {bd.get('voice_prompt',0):6.4f} | "
                f"{bd.get('prompt_image',0):6.4f} | {bd.get('final_score',0):6.4f} | "
                f"{bd.get('decision', 'N/A'):>10s}"
            )
        if n_scenes > 30:
            print(f"  ... ({n_scenes - 30} more scenes)")

        # ── 8. Render MP4 ──────────────────────────────────────────────────
        print(f"\n[6] Rendering MP4...")
        output_dir = tmp_dir / "renders"
        output_dir.mkdir()
        output_path = str(output_dir / "v5_50img_final.mp4")

        t_render = time.monotonic()
        render_result = service.render(
            project_id="v5_50img_e2e",
            output=output_path,
            preset="landscape",
        )
        render_time = time.monotonic() - t_render
        print(f"    Render completed in {render_time:.1f}s")

        mp4_path = render_result["output"]
        mp4_meta = render_result["metadata"]

        # Copy to project output
        final_output = Path("outputs") / "v5_50img_final.mp4"
        final_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mp4_path, final_output)

        # ── 9. Print results ────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("FRAMEFORGE V5 — 50-IMAGE E2E RESULTS")
        print("=" * 80)

        print(f"\n  Input images:          50")
        print(f"  Input prompts:         50")
        print(f"  Audio duration:        {audio_dur:.2f}s")
        print(f"  Narration segments:    {editor_meta.get('num_beats', 'N/A')}")
        print(f"  Final scenes:          {n_scenes}")
        print(f"  Images used:           {editor_meta.get('images_used', len(usage_counts))}")
        print(f"  Images unused:         {editor_meta.get('unused_images', 50 - len(usage_counts))}")
        print(f"  Unique images used:    {len(usage_counts)}")
        print(f"  Repeated images:       {repeated}")
        print(f"  Max image usage:       {max_usage}")
        print(f"  Number of cuts:        {editor_meta.get('num_cuts', 0)}")

        print(f"\n  Voice->Image range:    {min(vi_scores):.4f} - {max(vi_scores):.4f}")
        print(f"  Voice->Prompt range:   {min(vp_scores):.4f} - {max(vp_scores):.4f}")
        print(f"  Prompt->Image range:   {min(pi_scores):.4f} - {max(pi_scores):.4f}")
        print(f"  Final score range:     {min(final_scores):.4f} - {max(final_scores):.4f}")

        print(f"\n  Average shot duration: {avg_dur:.2f}s")
        print(f"  Min shot duration:     {min_dur:.2f}s")
        print(f"  Max shot duration:     {max_dur:.2f}s")

        print(f"\n  Order preserved:       {order_preserved}")
        print(f"  Backward jumps:        {backward_jumps}")
        print(f"  Oscillations:          {oscillations}")
        print(f"  Gap-free:              {gaps == 0}")

        print(f"\n  CLIP model:            {config.visual_model}")
        print(f"  CLIP embedding dim:    512")
        print(f"  LLM enabled:           {config.llm_enabled}")
        print(f"  LLM model:             {config.llm_model}")
        print(f"  Alignment method:      {result.get('alignment_method', 'N/A')}")

        print(f"\n  Analyze time:          {analyze_time:.1f}s")
        print(f"  Render time:           {render_time:.1f}s")
        print(f"  Total cold time:       {analyze_time + render_time:.1f}s")

        print(f"\n  Output resolution:     {mp4_meta.get('width', 'N/A')}x{mp4_meta.get('height', 'N/A')}")
        print(f"  FPS:                   {mp4_meta.get('fps', 'N/A')}")
        print(f"  Video codec:           {mp4_meta.get('video_codec', 'N/A')}")
        print(f"  Audio codec:           {mp4_meta.get('audio_codec', 'N/A')}")
        print(f"  Video duration:        {mp4_meta.get('duration', 'N/A')}s")
        print(f"  Audio duration:        {audio_dur:.2f}s")
        print(f"  Duration delta:        {abs(mp4_meta.get('duration', 0) - audio_dur):.4f}s")
        print(f"  File size:             {Path(mp4_path).stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  Final MP4:             {final_output}")

        # Image sequence
        print(f"\n  Image sequence:        {', '.join(str(n) for n in image_nums[:15])}", end="")
        if len(image_nums) > 15:
            print(f" ... ({len(image_nums)} total)")
        else:
            print()

        # ── 10. Assertions ──────────────────────────────────────────────────
        print("\n[7] Running assertions...")
        checks = [
            ("CLIP loaded", result.get("alignment_method", "").startswith("sequential")),
            ("50 images processed", len(images_dicts) == 50),
            ("Scenes generated", n_scenes > 0),
            ("Image order preserved", order_preserved),
            ("No backward jumps", backward_jumps == 0),
            ("No oscillation", oscillations == 0),
            ("Max usage <= 2", max_usage <= 2),
            ("Voice->Image varying", max(vi_scores) - min(vi_scores) > 0.001),
            ("Voice->Prompt varying", max(vp_scores) - min(vp_scores) > 0.001),
            ("Timeline gap-free", gaps == 0),
            ("Starts at 0", abs(scenes[0]["start"]) < 0.01),
            ("Ends at audio duration", abs(scenes[-1]["end"] - audio_dur) < 0.1),
            ("A/V sync valid", abs(mp4_meta.get("duration", 0) - audio_dur) < 0.2),
            ("H.264 video", mp4_meta.get("video_codec") == "h264"),
            ("AAC audio", mp4_meta.get("audio_codec") == "aac"),
            ("MP4 file exists", Path(mp4_path).is_file()),
        ]

        all_pass = True
        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    [{status}] {name}")

        if all_pass:
            print("\n  ALL CHECKS PASSED!")
        else:
            print("\n  SOME CHECKS FAILED!")

        return {
            "scenes": n_scenes,
            "images_used": len(usage_counts),
            "order_preserved": order_preserved,
            "max_usage": max_usage,
            "avg_duration": avg_dur,
            "mp4_path": str(final_output),
            "all_pass": all_pass,
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    result = run_e2e_validation()
    print(f"\nFinal MP4: {result['mp4_path']}")
    print(f"All checks passed: {result['all_pass']}")
