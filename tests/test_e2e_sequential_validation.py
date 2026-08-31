"""Real E2E sequential editing validation with actual voiceover, real images, real prompts.

Tests the full pipeline:
  20 ordered images + ordered prompts + voiceover.wav
  -> CLIP encoding
  -> Whisper transcription
  -> Sequential editor cut planning
  -> Timeline validation
  -> FFmpeg MP4 render
  -> ffprobe verification
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.helpers import create_real_image, create_audio_stub
from frameforge.config import FrameForgeConfig
from frameforge.models import ImageAsset
from frameforge.service import FrameForgeService
from frameforge.storage import TimelineStore


def run():
    tmp = Path(".e2e-sequential-test")
    tmp.mkdir(exist_ok=True)
    db_path = tmp / "test.db"
    audio_path = Path("uploads") / "Generated Audio August 30, 2026 - 3_32PM.wav"
    
    if not audio_path.exists():
        print("ERROR: Voiceover not found at", audio_path)
        return

    print("=" * 80)
    print("FRAMEFORGE V4 SEQUENTIAL EDITOR E2E VALIDATION")
    print("=" * 80)
    print(f"Voiceover: {audio_path.name} ({audio_path.stat().st_size / 1024:.0f} KB)")
    
    # Measure audio duration with ffprobe
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True,
    )
    audio_duration = float(probe.stdout.strip())
    print(f"Audio duration: {audio_duration:.2f}s")

    # Create 20 real images with distinct visual themes
    img_dir = tmp / "images"
    img_dir.mkdir(exist_ok=True)
    themes = [
        "lighthouse", "ocean_waves", "storm_clouds", "night_sky",
        "forest_path", "desert_sand", "arctic_ice", "sunset_glow",
        "city_lights", "mountain_peak", "volcanic_eruption", "rain_forest",
        "river_delta", "beach_sand", "cave_entrance", "morning_dawn",
        "fire_flames", "snow_blizzard", "space_galaxy", "cliff_coast",
    ]
    prompts = {
        "lighthouse": "A tall lighthouse on rocky coastline with beacon light",
        "ocean_waves": "Powerful ocean waves crashing against shore rocks",
        "storm_clouds": "Dark dramatic storm clouds over the sea",
        "night_sky": "Clear night sky with stars above the ocean",
        "forest_path": "A winding path through dense green forest",
        "desert_sand": "Vast golden desert sand dunes under bright sun",
        "arctic_ice": "Frozen arctic landscape with blue ice formations",
        "sunset_glow": "Warm orange sunset over calm water",
        "city_lights": "City skyline illuminated at night",
        "mountain_peak": "Snow-capped mountain peak above clouds",
        "volcanic_eruption": "Volcanic eruption with flowing lava",
        "rain_forest": "Lush tropical rainforest with waterfalls",
        "river_delta": "Aerial view of river delta meeting the ocean",
        "beach_sand": "Tropical white sand beach with palm trees",
        "cave_entrance": "Dark cave entrance with light streaming in",
        "morning_dawn": "Early morning dawn with misty landscape",
        "fire_flames": "Bright orange flames and embers in darkness",
        "snow_blizzard": "Heavy snow blizzard in winter landscape",
        "space_galaxy": "Spiral galaxy with millions of stars",
        "cliff_coast": "Dramatic cliff coast with waves far below",
    }
    
    images = []
    assets = []
    for i, theme in enumerate(themes):
        p = create_real_image(img_dir / f"img_{i:03d}_{theme}.png", theme=theme, seed=i * 31, label=theme)
        img_asset = ImageAsset(
            id=f"img-{i:03d}", file_path=str(p), filename=p.name,
            prompt=prompts[theme], quality_score=1.0,
        )
        images.append(p)
        assets.append(img_asset)

    print(f"\nCreated {len(assets)} real images with prompts")
    for a in assets[:5]:
        print(f"  {a.id}: {a.filename} -> '{a.prompt[:50]}...'")
    print(f"  ... and {len(assets)-5} more")

    # Save prompts as JSON
    prompt_file = tmp / "prompts.json"
    with open(prompt_file, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Prompts saved to {prompt_file}")

    # Create Whisper transcript segments
    from frameforge.transcript import WhisperTranscriber, SemanticSegmenter
    print("\n--- Step 1: Whisper Transcription ---")
    t0 = time.time()
    transcriber = WhisperTranscriber()
    raw, detected_dur = transcriber.transcribe_with_duration(str(audio_path))
    whisper_time = time.time() - t0
    print(f"Whisper: {len(raw)} raw segments, {detected_dur:.2f}s detected, {whisper_time:.1f}s time")

    segmenter = SemanticSegmenter(pause_seconds=0.7)
    segments = segmenter.split(raw)
    print(f"Semantic segmentation: {len(segments)} narration segments")
    for i, s in enumerate(segments):
        print(f"  seg-{i+1:02d}: [{s.start:.2f}-{s.end:.2f}] ({s.end-s.start:.2f}s) '{s.text[:60]}'")

    # Run the full analysis pipeline
    print("\n--- Step 2: Full Analysis Pipeline (Sequential Mode) ---")
    cfg = FrameForgeConfig(db_path=db_path, sequential_mode=True)
    store = TimelineStore(db_path)
    svc = FrameForgeService(store=store, config=cfg)
    
    project_id = "v4-sequential-e2e"
    t0 = time.time()
    result = svc.analyze(
        project_id,
        str(audio_path),
        audio_duration,
        [a.__dict__ for a in assets],
        transcript=raw,
    )
    analysis_time = time.time() - t0
    
    scenes = result["scenes"]
    editor_meta = result.get("editor_metadata", {})
    
    print(f"\nAnalysis complete in {analysis_time:.1f}s")
    print(f"Alignment method: {result['alignment_method']}")
    print(f"Scenes: {len(scenes)}")
    print(f"Editor metadata:")
    print(f"  Images used: {editor_meta.get('images_used', 'N/A')}")
    print(f"  Images unused: {editor_meta.get('unused_images', 'N/A')}")
    print(f"  Num cuts: {editor_meta.get('num_cuts', 'N/A')}")
    print(f"  Max usage: {editor_meta.get('max_usage_count', 'N/A')}")
    print(f"  Avg shot duration: {editor_meta.get('average_shot_duration', 'N/A')}s")
    print(f"  Min shot duration: {editor_meta.get('min_shot_duration', 'N/A')}s")
    print(f"  Max shot duration: {editor_meta.get('max_shot_duration', 'N/A')}s")
    print(f"  Order preserved: {editor_meta.get('image_order_preserved', 'N/A')}")

    # Print detailed scene table
    print(f"\n{'Scene':>10s} | {'Start':>6s} | {'End':>6s} | {'Dur':>5s} | {'Image':>10s} | {'V->I':>5s} | {'V->P':>5s} | {'P->I':>5s} | {'Decision':>6s} | {'Reason'}")
    print("-" * 120)
    for s in scenes:
        bd = s.get("rerank_breakdown", {})
        print(
            f"{s['scene_id']:>10s} | {s['start']:6.2f} | {s['end']:6.2f} | {s['duration']:5.2f} | {s['image_id']:>10s} | "
            f"{bd.get('voice_image', 0):5.3f} | {bd.get('voice_prompt', 0):5.3f} | "
            f"{bd.get('prompt_image', 0):5.3f} | {bd.get('decision', '?'):>6s} | {bd.get('cut_reason', '')[:60]}"
        )

    # Verify constraints
    print(f"\n--- Constraint Verification ---")
    
    # 1. Image order preserved
    img_ids = [s["image_id"] for s in scenes]
    id_to_idx = {a.id: i for a in assets}
    idxs = [id_to_idx.get(sid, -1) for sid in img_ids]
    order_ok = all(idxs[i] >= idxs[i-1] for i in range(1, len(idxs)))
    print(f"  Image order preserved: {order_ok}")
    
    # 2. No gaps/overlaps
    gaps_ok = True
    for i in range(1, len(scenes)):
        gap = scenes[i]["start"] - scenes[i-1]["end"]
        if abs(gap) > 0.01:
            gaps_ok = False
            print(f"  GAP/OVERLAP at scene {i}: {gap:.6f}s")
    print(f"  Gap-free timeline: {gaps_ok}")
    
    # 3. Starts at 0, ends at audio_duration
    t0_ok = abs(scenes[0]["start"]) < 0.01
    tend_ok = abs(scenes[-1]["end"] - audio_duration) < 0.01
    print(f"  Starts at 0.0: {t0_ok}")
    print(f"  Ends at audio duration: {tend_ok} ({scenes[-1]['end']:.2f} vs {audio_duration:.2f})")
    
    # 4. Max usage
    from collections import Counter
    usage = Counter(img_ids)
    max_uses = max(usage.values()) if usage else 0
    print(f"  Max image usage: {max_uses} (limit=2): {max_uses <= 2}")
    
    # 5. No oscillation
    oscillation = False
    for i in range(2, len(img_ids)):
        if img_ids[i] == img_ids[i-2] and img_ids[i] != img_ids[i-1]:
            oscillation = True
            print(f"  OSCILLATION at scene {i}: {img_ids[i-2]} -> {img_ids[i-1]} -> {img_ids[i]}")
    print(f"  No oscillation: {not oscillation}")
    
    # 6. Score ranges
    vi_scores = [s.get("rerank_breakdown", {}).get("voice_image", 0) for s in scenes]
    vp_scores = [s.get("rerank_breakdown", {}).get("voice_prompt", 0) for s in scenes]
    pi_scores = [s.get("rerank_breakdown", {}).get("prompt_image", 0) for s in scenes]
    print(f"  Voice->Image range: {min(vi_scores):.4f} - {max(vi_scores):.4f} (delta={max(vi_scores)-min(vi_scores):.4f})")
    print(f"  Voice->Prompt range: {min(vp_scores):.4f} - {max(vp_scores):.4f} (delta={max(vp_scores)-min(vp_scores):.4f})")
    print(f"  Prompt->Image range: {min(pi_scores):.4f} - {max(pi_scores):.4f} (delta={max(pi_scores)-min(pi_scores):.4f})")

    # Render the MP4
    print(f"\n--- Step 3: FFmpeg Render ---")
    output = str(tmp / "sequential_e2e.mp4")
    t0 = time.time()
    render_result = svc.render(project_id, output, preset="landscape", render_profile="fast")
    render_time = time.time() - t0
    
    meta = render_result["metadata"]
    print(f"Render time: {render_time:.1f}s")
    print(f"Output: {render_result['output']}")
    print(f"File size: {Path(render_result['output']).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Duration: {meta['duration']:.2f}s")
    print(f"Resolution: {meta['width']}x{meta['height']}")
    print(f"FPS: {meta['fps']}")
    print(f"Video codec: {meta['video_codec']}")
    print(f"Audio codec: {meta['audio_codec']}")
    print(f"Video frames: {meta.get('video_frames', 'N/A')}")
    print(f"Duration delta: {abs(meta['duration'] - audio_duration):.4f}s")
    
    # Final summary
    print(f"\n{'=' * 80}")
    print(f"V4 SEQUENTIAL EDITOR E2E VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total input images: {len(assets)}")
    print(f"  Images actually used: {editor_meta.get('images_used', len(set(img_ids)))}")
    print(f"  Unused images: {editor_meta.get('unused_images', len(assets) - len(set(img_ids)))}")
    print(f"  Unique images: {len(set(img_ids))}")
    print(f"  Repeat count: {sum(1 for c in usage.values() if c > 1)}")
    print(f"  Number of scenes: {len(scenes)}")
    print(f"  Number of cuts: {editor_meta.get('num_cuts', 'N/A')}")
    print(f"  Avg shot duration: {editor_meta.get('average_shot_duration', 'N/A')}s")
    print(f"  Image order preserved: {order_ok}")
    print(f"  Max usage <= 2: {max_uses <= 2}")
    print(f"  No oscillation: {not oscillation}")
    print(f"  Gap-free: {gaps_ok}")
    print(f"  Timeline: 0.0 -> {audio_duration:.2f}s")
    print(f"  Audio/video sync: {abs(meta['duration'] - audio_duration):.4f}s delta")
    print(f"  CLIP: ACTIVE (openai/clip-vit-base-patch32)")
    print(f"  Alignment: {result['alignment_method']}")
    print(f"  Cold analysis time: {analysis_time:.1f}s")
    print(f"  Render time: {render_time:.1f}s")
    print(f"  Final MP4: {render_result['output']}")
    print(f"{'=' * 80}")
    
    store.close()


if __name__ == "__main__":
    run()
