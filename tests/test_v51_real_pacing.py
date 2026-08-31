"""FrameForge V5.1 — Real Pacing Comparison.

Runs the sequential editor on the real 138s voiceover with 50 images
and reports detailed pacing metrics.
"""
from __future__ import annotations
import os, sys, time, json, tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frameforge.config import FrameForgeConfig
from frameforge.models import ImageAsset
from frameforge.editor import SequentialEditor
from frameforge.alignment import SceneAlignmentEngine
from frameforge.transcript import SemanticSegmenter, WhisperTranscriber
from frameforge.storage import TimelineStore
from tests.helpers import create_real_image, ALL_THEME_NAMES


def main():
    print("=" * 80)
    print("FRAMEFORGE V5.1 — REAL PACING COMPARISON")
    print("=" * 80)

    # Find real voiceover
    vo = None
    for f in Path("uploads").iterdir():
        if f.name.startswith("Generated Audio August 30"):
            vo = f
            break
    if not vo:
        for f in Path("uploads").iterdir():
            if f.suffix.lower() in (".wav", ".mp3"):
                vo = f
                break
    if not vo:
        print("ERROR: No voiceover found")
        return

    print(f"\nVoiceover: {vo.name} ({vo.stat().st_size / 1024:.0f} KB)")

    # Transcribe
    print("\n[1] Transcribing voiceover...")
    cfg = FrameForgeConfig.from_env()
    t0 = time.time()
    transcriber = WhisperTranscriber(config=cfg)
    raw_segments, audio_duration = transcriber.transcribe_with_duration(str(vo))
    t_transcribe = time.time() - t0
    print(f"    Transcribed in {t_transcribe:.1f}s: {audio_duration:.2f}s audio, {len(raw_segments)} segments")

    for i, seg in enumerate(raw_segments):
        word_count = len(seg.words) if seg.words else 0
        print(f"    [{seg.start:.2f}-{seg.end:.2f}] ({seg.end-seg.start:.2f}s) words={word_count}: {seg.text[:80]}")

    # Segment
    segger = SemanticSegmenter(pause_seconds=cfg.pause_split_threshold)
    segments = segger.split(raw_segments)
    print(f"    Segments after semantic split: {len(segments)}")

    # Create 50 ordered images
    print(f"\n[2] Creating 50 ordered images...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        images = []
        for i in range(50):
            theme = ALL_THEME_NAMES[i % len(ALL_THEME_NAMES)]
            img_path = create_real_image(tmp_dir / f"img_{i:03d}.png", theme=theme, seed=i, label=f"img {i}")
            images.append(ImageAsset(
                id=f"img-{i:03d}",
                file_path=str(img_path),
                filename=f"img_{i:03d}.png",
                prompt=f"Image {i}: {theme} scene",
            ))
        print(f"    Created {len(images)} images")

        # Compute CLIP signals
        print("\n[3] Computing CLIP multimodal signals...")
        aligner = SceneAlignmentEngine(config=cfg)
        t0 = time.time()

        import numpy as np

        voice_image_matrix = None
        if aligner.visual_engine.has_clip:
            img_vecs = aligner.visual_engine.encode_images(images)
            narr_vecs = aligner.visual_engine.encode_text([s.text for s in segments])
            if img_vecs is not None and narr_vecs is not None:
                raw_vi = np.dot(narr_vecs, img_vecs.T)
                voice_image_matrix = (np.clip(raw_vi, -1.0, 1.0) + 1.0) / 2.0
            print(f"    CLIP Voice->Image matrix: {voice_image_matrix.shape if voice_image_matrix is not None else 'None'}")

        voice_prompt_matrix = None
        if not aligner.text_engine.is_fallback:
            prompts = [img.prompt for img in images]
            prompt_vecs = aligner.text_engine.encode_texts(prompts, batch_size=64)
            narr_vecs = aligner.text_engine.encode_texts([s.text for s in segments], batch_size=64)
            if prompt_vecs is not None and narr_vecs is not None:
                raw_vp = np.dot(narr_vecs, prompt_vecs.T)
                voice_prompt_matrix = (np.clip(raw_vp, -1.0, 1.0) + 1.0) / 2.0
            print(f"    Text Voice->Prompt matrix: {voice_prompt_matrix.shape if voice_prompt_matrix is not None else 'None'}")

        prompt_image_scores = aligner.visual_engine.compute_prompt_image_consistency(images)
        t_clip = time.time() - t0
        print(f"    CLIP signals computed in {t_clip:.1f}s")
        print(f"    Prompt->Image range: [{min(prompt_image_scores):.4f}, {max(prompt_image_scores):.4f}]")

        # Run V5.1 SequentialEditor
        print(f"\n[4] Running V5.1 SequentialEditor...")
        editor = SequentialEditor(config=cfg)
        t0 = time.time()
        scenes, meta = editor.plan(
            segments, images, audio_duration,
            voice_image_matrix=voice_image_matrix,
            voice_prompt_matrix=voice_prompt_matrix,
            prompt_image_scores=prompt_image_scores,
        )
        t_plan = time.time() - t0
        print(f"    Planned in {t_plan:.3f}s")

        # Report
        durations = [s.duration for s in scenes]
        image_ids = [s.image_id for s in scenes]

        print(f"\n{'=' * 80}")
        print(f"V5.1 PACING RESULTS")
        print(f"{'=' * 80}")
        print(f"Audio duration:       {audio_duration:.2f}s")
        print(f"Narration segments:   {len(segments)}")
        print(f"Final scenes:         {len(scenes)}")
        print(f"Images used:          {meta['images_used']}")
        print(f"Images unused:        {meta['unused_images']}")
        print(f"Num cuts:             {meta['num_cuts']}")
        print(f"Num beats:            {meta['num_beats']}")
        print(f"Average shot:         {meta['average_shot_duration']:.2f}s")
        print(f"Min shot:             {meta['min_shot_duration']:.2f}s")
        print(f"Max shot:             {meta['max_shot_duration']:.2f}s")
        print(f"Max usage:            {meta['max_usage_count']}")
        print(f"Repeat count:         {meta['repeat_count']}")
        print(f"Image order preserved:{meta['image_order_preserved']}")
        print(f"Signal quality:       {meta['signal_quality']:.2f}")
        print(f"Signals active:       {meta['signals_active']}")
        print(f"LLM used:             {meta['llm_used']}")
        print(f"Global target:        {meta['global_target']:.2f}s")

        # Score ranges
        vi_scores = [s.rerank_breakdown.get("voice_image", 0) for s in scenes]
        vp_scores = [s.rerank_breakdown.get("voice_prompt", 0) for s in scenes]
        pi_scores = [s.rerank_breakdown.get("prompt_image", 0) for s in scenes]
        print(f"\nVoice->Image range:   [{min(vi_scores):.4f}, {max(vi_scores):.4f}] (delta={max(vi_scores)-min(vi_scores):.4f})")
        print(f"Voice->Prompt range:  [{min(vp_scores):.4f}, {max(vp_scores):.4f}] (delta={max(vp_scores)-min(vp_scores):.4f})")
        print(f"Prompt->Image range:  [{min(pi_scores):.4f}, {max(pi_scores):.4f}] (delta={max(pi_scores)-min(pi_scores):.4f})")

        # Duration distribution
        print(f"\n--- DURATION DISTRIBUTION ---")
        for lo, hi, label in [(0,2,"<2s"),(2,4,"2-4s"),(4,6,"4-6s"),(6,10,"6-10s"),(10,999,">10s")]:
            count = sum(1 for d in durations if lo <= d < hi)
            bar = "#" * count
            print(f"  {label:>6s}: {count:3d} {bar}")

        # Scene details
        print(f"\n--- SCENE SEQUENCE ---")
        for i, s in enumerate(scenes):
            d = s.duration
            bd = s.rerank_breakdown
            print(f"  {s.scene_id}: {s.image_id}  {s.start:6.2f}-{s.end:6.2f} ({d:.2f}s)  "
                  f"V->I={bd.get('voice_image',0):.3f} V->P={bd.get('voice_prompt',0):.3f}  "
                  f"{bd.get('cut_reason','')[:60]}")

        # Order and oscillation check
        print(f"\n--- ORDER & OSCILLATION CHECK ---")
        prev_idx = -1
        backward = 0
        for s in scenes:
            idx = int(s.image_id.split("-")[1])
            if idx < prev_idx:
                backward += 1
            prev_idx = idx
        print(f"Backward jumps: {backward}")
        osc = 0
        for i in range(2, len(image_ids)):
            if image_ids[i] == image_ids[i-2] and image_ids[i] != image_ids[i-1]:
                osc += 1
        print(f"Oscillations: {osc}")

        # Save for render
        project_id = "v51-render-test"
        store = TimelineStore(cfg.db_path)

        # Get audio file path
        audio_path = str(vo.resolve())
        result_dict = {
            "audio": {"file": audio_path, "duration": audio_duration},
            "scenes": [
                {**s.as_dict(), "duration": s.duration}
                for s in scenes
            ],
            "alignment_method": "sequential_multimodal+llm",
            "images": [
                {"id": img.id, "file_path": img.file_path, "filename": img.filename,
                 "prompt": img.prompt, "prompt_source": img.prompt_source,
                 "metadata": img.metadata, "file_hash": img.file_hash,
                 "phash": img.phash, "width": img.width, "height": img.height,
                 "quality_score": img.quality_score, "is_duplicate": img.is_duplicate,
                 "normalized_path": img.normalized_path}
                for img in images
            ],
        }
        store.save("v51-job", project_id, audio_path, audio_duration, result_dict, "ready", "sequential_multimodal+llm")
        store.close()
        print(f"\nSaved project: {project_id} (for render)")

        # Summary
        print(f"\n{'=' * 80}")
        print(f"SUMMARY")
        print(f"{'=' * 80}")
        print(f"CLIP:              {'ACTIVE' if voice_image_matrix is not None else 'FALLBACK'}")
        print(f"Alignment method:  {result_dict['alignment_method']}")
        print(f"CLIP compute time: {t_clip:.1f}s")
        print(f"Plan time:         {t_plan:.3f}s")
        print(f"Transcribe time:   {t_transcribe:.1f}s")
        print(f"Final MP4:         pending (run render)")


if __name__ == "__main__":
    main()
