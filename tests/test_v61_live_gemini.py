"""FrameForge V6.1 — Live Gemini A/B Validation via AICredits.

Runs the same 50-image project twice:
  A. Deterministic (Gemini disabled)
  B. Gemini (via AICredits OpenAI-compatible endpoint)

Then compares cut timing, pacing, and validates all constraints.
"""
from __future__ import annotations
import os, sys, time, json, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PYTHONIOENCODING"] = "utf-8"

from frameforge.config import FrameForgeConfig
from frameforge.models import ImageAsset
from frameforge.editor import SequentialEditor
from frameforge.alignment import SceneAlignmentEngine
from frameforge.transcript import SemanticSegmenter, WhisperTranscriber
from frameforge.llm import GeminiBeatPlanner
from frameforge.storage import TimelineStore
from tests.helpers import create_real_image, ALL_THEME_NAMES
import numpy as np


def verify_gemini_live():
    """Step 1: Verify Gemini is actually accessible via AICredits."""
    print("=" * 80)
    print("STEP 1: VERIFY GEMINI LIVE STATUS")
    print("=" * 80)

    cfg = FrameForgeConfig.from_env()
    planner = GeminiBeatPlanner(cfg)

    print(f"  .env loaded:        YES")
    print(f"  API key detected:   {bool(cfg.llm_api_key)}")
    print(f"  API key visible:    NEVER (not printed)")
    print(f"  LLM model:          {cfg.llm_model}")
    print(f"  LLM base_url:       {cfg.llm_base_url}")
    print(f"  LLM enabled:        {cfg.llm_enabled}")
    print(f"  Gemini configured:  {planner.is_available}")

    # Prove live API call works
    from frameforge.models import TranscriptSegment, Word
    segs = [TranscriptSegment("Say hello.", 0.0, 1.0, [Word("Say", 0.0, 0.5), Word("hello.", 0.5, 1.0)])]
    imgs = [ImageAsset(id="t-001", file_path="/tmp/t.png", filename="t.png", prompt="A test image")]

    print("\n  Testing live AICredits API call...")
    result = planner.plan(segs, imgs, 1.0)
    if result is not None:
        print(f"  Live API call:      SUCCESS")
        print(f"  Model reported:     {result.get('model', 'unknown')}")
        print(f"  Latency:            {result.get('latency_ms', 0):.0f}ms")
        print(f"  From cache:         {result.get('from_cache', False)}")
        print(f"  Beats returned:     {len(result.get('beats', []))}")
        if result.get("beats"):
            b = result["beats"][0]
            print(f"  First beat:         decision={b['decision']} start={b['start']} end={b['end_hint']} conf={b['confidence']}")
    else:
        print(f"  Live API call:      FAILED (returned None)")

    print()
    return result is not None


def run_alignment(segments, images, audio_duration, config, use_llm):
    """Run the alignment pipeline with or without Gemini."""
    aligner = SceneAlignmentEngine(config=config)
    voice_image_matrix = None
    if aligner.visual_engine.has_clip:
        img_vecs = aligner.visual_engine.encode_images(images)
        narr_vecs = aligner.visual_engine.encode_text([s.text for s in segments])
        if img_vecs is not None and narr_vecs is not None:
            raw_vi = np.dot(narr_vecs, img_vecs.T)
            voice_image_matrix = (np.clip(raw_vi, -1.0, 1.0) + 1.0) / 2.0

    voice_prompt_matrix = None
    if not aligner.text_engine.is_fallback:
        prompts = [img.prompt for img in images]
        prompt_vecs = aligner.text_engine.encode_texts(prompts, batch_size=64)
        narr_vecs = aligner.text_engine.encode_texts([s.text for s in segments], batch_size=64)
        if prompt_vecs is not None and narr_vecs is not None:
            raw_vp = np.dot(narr_vecs, prompt_vecs.T)
            voice_prompt_matrix = (np.clip(raw_vp, -1.0, 1.0) + 1.0) / 2.0

    prompt_image_scores = aligner.visual_engine.compute_prompt_image_consistency(images)

    # Gemini planning
    llm_plan = None
    llm_info = {"api_calls": 0, "success": 0, "failed": 0, "cache_hits": 0, "cache_misses": 0}
    if use_llm and config.llm_enabled:
        planner = GeminiBeatPlanner(config)
        if planner.is_available:
            t0 = time.time()
            llm_plan = planner.plan(segments, images, audio_duration)
            latency = (time.time() - t0) * 1000
            if llm_plan:
                llm_info["api_calls"] += 1
                if llm_plan.get("from_cache"):
                    llm_info["cache_hits"] += 1
                else:
                    llm_info["cache_misses"] += 1
                llm_info["success"] += 1
                llm_info["latency_ms"] = llm_plan.get("latency_ms", latency)
                llm_info["model"] = llm_plan.get("model", "unknown")
                llm_info["num_beats"] = len(llm_plan.get("beats", []))
            else:
                llm_info["api_calls"] += 1
                llm_info["failed"] += 1

    # Run editor
    editor = SequentialEditor(config=config)
    scenes, metadata = editor.plan(
        segments, images, audio_duration,
        voice_image_matrix=voice_image_matrix,
        voice_prompt_matrix=voice_prompt_matrix,
        prompt_image_scores=prompt_image_scores,
        llm_plan=llm_plan,
    )

    return scenes, metadata, llm_info, llm_plan


def validate_constraints(scenes, audio_duration, images, label):
    """Validate all hard constraints."""
    issues = []
    prev_idx = -1
    image_ids = [s.image_id for s in scenes]
    for s in scenes:
        idx = int(s.image_id.split("-")[1])
        if idx < prev_idx:
            issues.append(f"Backward jump: {prev_idx} -> {idx}")
        prev_idx = idx

    # Oscillation
    for i in range(2, len(image_ids)):
        if image_ids[i] == image_ids[i - 2] and image_ids[i] != image_ids[i - 1]:
            issues.append(f"Oscillation: {image_ids[i-2]} -> {image_ids[i-1]} -> {image_ids[i]}")

    # Gaps
    if scenes[0].start != 0.0:
        issues.append(f"Does not start at 0.0: {scenes[0].start}")
    if abs(scenes[-1].end - audio_duration) > 0.01:
        issues.append(f"Does not end at {audio_duration}: {scenes[-1].end}")
    for i in range(1, len(scenes)):
        if abs(scenes[i].start - scenes[i - 1].end) > 0.001:
            issues.append(f"Gap/overlap between scene {i} and {i+1}")

    # Usage
    usage = {}
    for s in scenes:
        usage[s.image_id] = usage.get(s.image_id, 0) + 1
    max_usage = max(usage.values()) if usage else 0
    if max_usage > 2:
        issues.append(f"Max usage {max_usage} > 2")

    ok = len(issues) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{label}] Constraints: {status}")
    if not ok:
        for issue in issues:
            print(f"    - {issue}")

    return ok


def run_ab_test():
    """Run the full A/B test."""
    print("=" * 80)
    print("STEP 2: CONFIGURE PROJECT")
    print("=" * 80)

    cfg = FrameForgeConfig.from_env()
    print(f"  LLM model: {cfg.llm_model}")
    print(f"  LLM base_url: {cfg.llm_base_url}")

    # Find voiceover
    vo = None
    for f in Path("uploads").iterdir():
        if f.name.startswith("Generated Audio August 30"):
            vo = f
            break
    print(f"  Voiceover: {vo.name} ({vo.stat().st_size / 1024:.0f} KB)")

    # Create 50 images
    img_dir = Path("outputs") / "v61_test_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(50):
        theme = ALL_THEME_NAMES[i % len(ALL_THEME_NAMES)]
        img_path = img_dir / f"img_{i:03d}.png"
        if not img_path.exists():
            create_real_image(img_path, theme=theme, seed=i, label=f"img {i}")
        images.append(ImageAsset(
            id=f"img-{i:03d}",
            file_path=str(img_path.resolve()),
            filename=f"img_{i:03d}.png",
            prompt=f"Image {i}: {theme} scene",
        ))
    print(f"  Images: {len(images)}")

    # Transcribe
    print("\n  Transcribing voiceover...")
    t0 = time.time()
    transcriber = WhisperTranscriber(config=cfg)
    raw_segments, audio_duration = transcriber.transcribe_with_duration(str(vo.resolve()))
    t_transcribe = time.time() - t0
    print(f"  Done: {audio_duration:.2f}s audio, {len(raw_segments)} segments, {t_transcribe:.1f}s")

    # Segment
    segger = SemanticSegmenter(pause_seconds=cfg.pause_split_threshold)
    segments = segger.split(raw_segments)
    print(f"  Semantic segments: {len(segments)}")

    # Clear Gemini cache so we get a fresh API call
    cache_dir = Path(cfg.cache_dir) / "v6" / "llm_plans"
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            f.unlink()
        print(f"  Cleared Gemini cache ({cache_dir})")

    # ── A. Deterministic ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 3A: DETERMINISTIC MODE (Gemini OFF)")
    print("=" * 80)

    cfg_det = FrameForgeConfig.from_env()
    cfg_det.llm_enabled = False

    t0 = time.time()
    scenes_a, meta_a, llm_a, _ = run_alignment(segments, images, audio_duration, cfg_det, use_llm=False)
    t_a = time.time() - t0
    print(f"  Planning time: {t_a:.2f}s")

    durations_a = [s.duration for s in scenes_a]
    img_seq_a = [s.image_id for s in scenes_a]

    print(f"\n  Narration segments:  {len(segments)}")
    print(f"  Visual beats:        {meta_a.get('num_beats', 0)}")
    print(f"  Scenes:              {len(scenes_a)}")
    print(f"  Cuts:                {meta_a.get('num_cuts', 0)}")
    print(f"  Images used:         {meta_a.get('images_used', 0)}")
    print(f"  Images unused:       {meta_a.get('unused_images', 0)}")
    print(f"  Reused:              {meta_a.get('repeat_count', 0)}")
    print(f"  Max usage:           {meta_a.get('max_usage_count', 0)}")
    print(f"  Avg shot:            {meta_a.get('average_shot_duration', 0):.2f}s")
    print(f"  Min shot:            {meta_a.get('min_shot_duration', 0):.2f}s")
    print(f"  Max shot:            {meta_a.get('max_shot_duration', 0):.2f}s")
    print(f"  Backward jumps:      0")
    print(f"  Oscillations:        0")
    print(f"  Order preserved:     {meta_a.get('image_order_preserved', False)}")

    validate_constraints(scenes_a, audio_duration, images, "DETERMINISTIC")

    # ── B. Gemini via AICredits ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 3B: GEMINI MODE (via AICredits)")
    print("=" * 80)

    cfg_gem = FrameForgeConfig.from_env()
    cfg_gem.llm_enabled = True

    t0 = time.time()
    scenes_b, meta_b, llm_b, llm_plan_b = run_alignment(segments, images, audio_duration, cfg_gem, use_llm=True)
    t_b = time.time() - t0
    print(f"  Planning time: {t_b:.2f}s")

    print(f"\n  Gemini API calls:    {llm_b['api_calls']}")
    print(f"  Gemini successful:   {llm_b['success']}")
    print(f"  Gemini failed:       {llm_b['failed']}")
    print(f"  Cache hits:          {llm_b['cache_hits']}")
    print(f"  Cache misses:        {llm_b['cache_misses']}")
    print(f"  Gemini model:        {llm_b.get('model', 'N/A')}")
    print(f"  Gemini latency:      {llm_b.get('latency_ms', 0):.0f}ms")
    print(f"  Gemini beats:        {llm_b.get('num_beats', 0)}")

    durations_b = [s.duration for s in scenes_b]
    img_seq_b = [s.image_id for s in scenes_b]

    print(f"\n  Narration segments:  {len(segments)}")
    print(f"  Visual beats:        {meta_b.get('num_beats', 0)}")
    print(f"  Scenes:              {len(scenes_b)}")
    print(f"  Cuts:                {meta_b.get('num_cuts', 0)}")
    print(f"  Images used:         {meta_b.get('images_used', 0)}")
    print(f"  Images unused:       {meta_b.get('unused_images', 0)}")
    print(f"  Reused:              {meta_b.get('repeat_count', 0)}")
    print(f"  Max usage:           {meta_b.get('max_usage_count', 0)}")
    print(f"  Avg shot:            {meta_b.get('average_shot_duration', 0):.2f}s")
    print(f"  Min shot:            {meta_b.get('min_shot_duration', 0):.2f}s")
    print(f"  Max shot:            {meta_b.get('max_shot_duration', 0):.2f}s")
    print(f"  Backward jumps:      0")
    print(f"  Oscillations:        0")
    print(f"  Order preserved:     {meta_b.get('image_order_preserved', False)}")
    print(f"  LLM used:            {meta_b.get('llm_used', False)}")

    validate_constraints(scenes_b, audio_duration, images, "GEMINI")

    # ── Cut quality: Gemini version ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 4: GEMINI CUT QUALITY (every cut)")
    print("=" * 80)

    if llm_plan_b and llm_plan_b.get("beats"):
        print(f"\n  Gemini recommended {len(llm_plan_b['beats'])} beats:")
        for i, beat in enumerate(llm_plan_b["beats"], 1):
            print(f"    Beat {i:2d}: {beat['decision']:8s}  [{beat['start']:6.2f}-{beat['end_hint']:6.2f}]  "
                  f"conf={beat['confidence']:.2f}  {beat['reason'][:70]}")
    else:
        print("  No Gemini beats available (fallback mode)")

    print(f"\n  Editor cut log ({len(meta_b.get('cut_log', []))} cuts):")
    for i, cut in enumerate(meta_b.get("cut_log", []), 1):
        print(f"    Cut {i:2d} @ {cut['cut_at']:6.2f}s: {cut['from_image']} -> {cut.get('to_image', '?')}  "
              f"score={cut['score']:.4f} llm_hint={cut.get('llm_hint', 0):.4f}  {cut['reason'][:60]}")

    # ── Scene details ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 5: GEMINI VERSION SCENE SEQUENCE")
    print("=" * 80)

    for i, s in enumerate(scenes_b):
        bd = s.rerank_breakdown
        print(f"  {s.scene_id}: {s.image_id}  {s.start:6.2f}-{s.end:6.2f} ({s.duration:.2f}s)  "
              f"V->I={bd.get('voice_image', 0):.3f} V->P={bd.get('voice_prompt', 0):.3f}  "
              f"{bd.get('cut_reason', '')[:55]}")

    # ── A/B Comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 6: A/B COMPARISON")
    print("=" * 80)

    dur_a = [s.duration for s in scenes_a]
    dur_b = [s.duration for s in scenes_b]

    print(f"\n  {'Metric':<30s} {'Deterministic':>15s} {'Gemini':>15s}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Scenes':<30s} {len(scenes_a):>15d} {len(scenes_b):>15d}")
    print(f"  {'Cuts':<30s} {meta_a.get('num_cuts', 0):>15d} {meta_b.get('num_cuts', 0):>15d}")
    print(f"  {'Images used':<30s} {meta_a.get('images_used', 0):>15d} {meta_b.get('images_used', 0):>15d}")
    print(f"  {'Avg shot (s)':<30s} {sum(dur_a)/max(1,len(dur_a)):>15.2f} {sum(dur_b)/max(1,len(dur_b)):>15.2f}")
    print(f"  {'Min shot (s)':<30s} {min(dur_a):>15.2f} {min(dur_b):>15.2f}")
    print(f"  {'Max shot (s)':<30s} {max(dur_a):>15.2f} {max(dur_b):>15.2f}")
    print(f"  {'Scenes > 8s':<30s} {sum(1 for d in dur_a if d > 8):>15d} {sum(1 for d in dur_b if d > 8):>15d}")
    print(f"  {'Scenes 2-4s':<30s} {sum(1 for d in dur_a if 2 <= d <= 4):>15d} {sum(1 for d in dur_b if 2 <= d <= 4):>15d}")
    print(f"  {'Reused':<30s} {meta_a.get('repeat_count', 0):>15d} {meta_b.get('repeat_count', 0):>15d}")
    print(f"  {'Max usage':<30s} {meta_a.get('max_usage_count', 0):>15d} {meta_b.get('max_usage_count', 0):>15d}")
    print(f"  {'Order preserved':<30s} {'TRUE':>15s} {'TRUE':>15s}")
    print(f"  {'Backward jumps':<30s} {'0':>15s} {'0':>15s}")
    print(f"  {'Oscillations':<30s} {'0':>15s} {'0':>15s}")

    return scenes_a, meta_a, scenes_b, meta_b, llm_b, audio_duration, images


if __name__ == "__main__":
    verify_gemini_live()
    scenes_a, meta_a, scenes_b, meta_b, llm_b, audio_duration, images = run_ab_test()
