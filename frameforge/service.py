"""FrameForge V3 Resumable High-Level Orchestrator Service supporting Three-Way Alignment."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .alignment import SceneAlignmentEngine
from .config import DEFAULT_CONFIG, FrameForgeConfig
from .image_processor import ImagePreprocessor
from .jobs import JobManager
from .models import ImageAsset, Scene, Timeline
from .prompt_parser import PromptMappingParser
from .renderer import FFmpegRenderer, RenderError
from .storage import TimelineStore
from .transcript import SemanticSegmenter, WhisperTranscriber
from .validation import validate_timeline

logger = logging.getLogger("frameforge.service")


class FrameForgeService:
    def __init__(self, store: Optional[TimelineStore] = None, config: Optional[FrameForgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.store = store or TimelineStore(self.config.db_path)
        self.job_manager = JobManager()
        self.preprocessor = ImagePreprocessor(self.config)

    def visual_status(self) -> dict:
        """Return diagnostics about the active visual/CLIP model and all scoring signals."""
        from .visual import VisualUnderstandingEngine
        from .alignment import SceneAlignmentEngine
        vis = VisualUnderstandingEngine(self.config)
        align = SceneAlignmentEngine(config=self.config)
        return {
            "clips_status": vis.model_status(),
            "alignment_status": align.model_status(),
            "full_multimodal_ready": vis.has_clip,
        }

    def analyze(
        self,
        project_id: str,
        audio_file: str,
        audio_duration: float | None,
        images: list[dict],
        transcript: list | None = None,
        prompts: dict[str, str] | str | Path | None = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        llm_enabled: bool | None = None,
    ) -> dict:
        def report(stage: str, pct: int, msg: str):
            if progress_callback:
                progress_callback(stage, pct, msg)

        # 1. Ingest & map prompts (if supplied)
        report("parsing_prompts", 5, "Parsing image prompts and mappings...")
        assets = [ImageAsset(**item) for item in images]
        
        prompt_stats = {"total_images": len(assets), "prompts_mapped": 0, "prompts_missing": len(assets)}
        if prompts:
            try:
                if isinstance(prompts, (str, Path)) and (Path(str(prompts)).is_file() or "\n" in str(prompts) or "{" in str(prompts)):
                    if Path(str(prompts)).is_file():
                        parsed_map = PromptMappingParser.parse_file(prompts)
                    elif str(prompts).strip().startswith("{") or str(prompts).strip().startswith("["):
                        parsed_map = PromptMappingParser.parse_json(prompts)
                    else:
                        parsed_map = PromptMappingParser.parse_csv(prompts)
                elif isinstance(prompts, dict):
                    parsed_map = prompts
                else:
                    parsed_map = {}

                prompt_stats = PromptMappingParser.apply_prompts_to_assets(assets, parsed_map)
                logger.info("Prompt mapping applied: %s", prompt_stats)
            except Exception as exc:
                logger.warning("Could not parse prompt mapping: %s. Using fallback captions.", exc)

        # 2. Preprocess & normalize image assets (resumable with disk metadata cache)
        report("preprocessing_images", 15, f"Analyzing and preprocessing {len(images)} images ({prompt_stats['prompts_mapped']} prompts mapped)...")

        def on_image_progress(done: int, total: int):
            sub_pct = 15 + int(20 * (done / max(1, total)))
            report("preprocessing_images", sub_pct, f"Preprocessed image {done}/{total}")

        assets = self.preprocessor.preprocess_batch(assets, progress_callback=on_image_progress)

        # 3. Transcribe voiceover (resumable with SHA-256 transcript cache)
        report("transcribing_audio", 40, "Transcribing voiceover audio and measuring duration...")
        if transcript is None:
            raw, detected_duration = WhisperTranscriber(config=self.config).transcribe_with_duration(audio_file)
            audio_duration = detected_duration
        elif audio_duration is None or audio_duration <= 0:
            raise ValueError("A positive measured audio duration is required with a supplied transcript.")
        else:
            raw = transcript

        # 4. Semantic segmentation
        report("segmenting_audio", 60, "Performing semantic clause and pause segmentation...")
        segments = SemanticSegmenter(pause_seconds=self.config.pause_split_threshold).split(raw)
        logger.info(
            "Timeline input: audio=%s duration=%.6f raw_segments=%d narration_segments=%d images=%d",
            audio_file,
            audio_duration or 0,
            len(raw),
            len(segments),
            len(assets),
        )

        # 5. Multimodal alignment — sequential (V4) or global Viterbi (V3)
        report("matching_images", 75, "Executing multimodal alignment and scene planning...")

        # Allow per-request LLM override
        prev_llm = self.config.llm_enabled
        if llm_enabled is not None:
            self.config.llm_enabled = llm_enabled

        aligner = SceneAlignmentEngine(config=self.config)
        editor_metadata = {}
        if self.config.sequential_mode:
            scenes, method, editor_metadata = aligner.sequential_align(
                segments, assets, audio_duration,
            )
        else:
            scenes, method = aligner.align(segments, assets, audio_duration)

        # Restore original LLM setting
        if llm_enabled is not None:
            self.config.llm_enabled = prev_llm
        logger.info(
            "Timeline candidates: scenes=%d alignment=%s cuts=%d",
            len(scenes), method, editor_metadata.get("num_cuts", 0),
        )

        # 6. Validation
        report("validating", 90, "Validating continuous timeline endpoints and invariants...")
        timeline = validate_timeline(
            Timeline(audio_file, audio_duration, scenes, method, raw),
            assets,
            verify_files=False,
        )
        result = timeline.as_dict()
        result["prompt_stats"] = prompt_stats
        if editor_metadata:
            result["editor_metadata"] = editor_metadata
        result["images"] = [
            {
                "id": x.id,
                "file_path": x.file_path,
                "filename": x.filename,
                "prompt": x.prompt,
                "prompt_source": x.prompt_source,
                "metadata": x.metadata,
                "file_hash": x.file_hash,
                "phash": x.phash,
                "width": x.width,
                "height": x.height,
                "quality_score": x.quality_score,
                "is_duplicate": x.is_duplicate,
            }
            for x in assets
        ]

        # 7. Persistence
        report("saving", 95, "Saving synchronized timeline to SQLite store...")
        self.store.save(str(uuid.uuid4()), project_id, audio_file, audio_duration, result, "ready", method)
        report("completed", 100, "Timeline synchronization complete.")
        return result

    def analyze_async(
        self,
        project_id: str,
        audio_file: str,
        audio_duration: float | None,
        images: list[dict],
        transcript: list | None = None,
        prompts: dict[str, str] | str | Path | None = None,
        llm_enabled: bool | None = None,
    ) -> str:
        job = self.job_manager.create_job(project_id, job_type="analyze")
        job_id = job.job_id

        def task():
            def cb(stage: str, progress: int, msg: str):
                self.job_manager.update_progress(job_id, stage, progress, msg)

            return self.analyze(
                project_id,
                audio_file,
                audio_duration,
                images,
                transcript,
                prompts=prompts,
                progress_callback=cb,
                llm_enabled=llm_enabled,
            )

        self.job_manager.submit_task(job_id, task)
        return job_id

    def update(self, project_id: str, timeline: dict) -> None:
        audio = timeline["audio"]
        audio_file = audio["file"]
        # If the supplied audio ref is a bare filename, try to recover the
        # full path from the previous store record to avoid losing the path.
        if not Path(audio_file).is_file():
            prev = self.store.get_latest(project_id)
            if prev and Path(prev["audio_file"]).is_file():
                audio_file = prev["audio_file"]
        self.store.save(
            str(uuid.uuid4()),
            project_id,
            audio_file,
            audio["duration"],
            timeline,
            "ready",
            timeline.get("alignment_method", "manual"),
        )

    def render(
        self,
        project_id: str,
        output: str,
        preset: str = "vertical",
        fit: str = "cover",
        transition: str = "cut",
        ffmpeg_path: str | None = None,
        render_path: str | None = None,
        render_profile: str = "balanced",
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> dict:
        def report(stage: str, pct: int, msg: str):
            if progress_callback:
                progress_callback(stage, pct, msg)

        job = self.store.get_latest(project_id)
        if not job:
            raise ValueError("Analyze before rendering.")
        saved = job["timeline"]
        assets = [ImageAsset(**item) for item in saved.get("images", [])]
        if not assets:
            raise ValueError("The saved timeline has no image associations. Analyze again before rendering.")

        # Resolve audio path: DB column is the source of truth (full absolute path).
        # The JSON timeline may contain a bare filename from older saves.
        db_audio_path = Path(job["audio_file"]) if job.get("audio_file") else None
        json_audio_ref = saved["audio"]["file"]
        audio_path = json_audio_ref
        if db_audio_path and db_audio_path.is_file():
            audio_path = str(db_audio_path)
        elif not Path(json_audio_ref).is_file():
            # Fallback: resolve bare filename against uploads directory
            uploads_dir = Path(self.config.uploads_dir)
            candidate = uploads_dir / Path(json_audio_ref).name
            if candidate.is_file():
                audio_path = str(candidate.resolve())
            else:
                raise ValueError(
                    f"Audio file not found. Tried: {json_audio_ref}, "
                    f"{db_audio_path}, {candidate}"
                )

        timeline = Timeline(
            audio_path,
            float(saved["audio"]["duration"]),
            [
                Scene(**{k: v for k, v in scene.items() if k != "duration"})
                for scene in saved["scenes"]
            ],
            saved.get("alignment_method", "manual"),
        )
        validate_timeline(timeline, assets, verify_files=True)

        report("rendering_video", 30, f"Rendering {len(timeline.scenes)} scenes with FFmpeg ({render_profile})...")
        renderer = FFmpegRenderer(ffmpeg_path, config=self.config)
        generated = renderer.render(
            timeline,
            assets,
            output,
            preset=preset,
            fit=fit,
            transition=transition,
            render_path=render_path,
            render_profile=render_profile,
        )

        report("probing", 85, "Verifying rendered MP4 streams and duration with FFprobe...")
        metadata = renderer.probe(generated)
        if metadata["video_codec"] != "h264" or metadata["audio_codec"] != "aac":
            raise RenderError("Rendered MP4 did not contain H.264 video and AAC audio.")
        if abs(metadata["duration"] - timeline.audio_duration) > 0.15:
            raise RenderError("Rendered MP4 duration does not match the authoritative voice-over duration.")

        report("completed", 100, f"Rendered {metadata['duration']:.2f}s MP4 successfully.")
        return {"output": str(generated), "metadata": metadata}

    def render_async(
        self,
        project_id: str,
        output: str,
        preset: str = "vertical",
        fit: str = "cover",
        transition: str = "cut",
        ffmpeg_path: str | None = None,
        render_path: str | None = None,
        render_profile: str = "balanced",
    ) -> str:
        job = self.job_manager.create_job(project_id, job_type="render")
        job_id = job.job_id

        def task():
            def cb(stage: str, progress: int, msg: str):
                self.job_manager.update_progress(job_id, stage, progress, msg)

            return self.render(
                project_id,
                output,
                preset=preset,
                fit=fit,
                transition=transition,
                ffmpeg_path=ffmpeg_path,
                render_path=render_path,
                render_profile=render_profile,
                progress_callback=cb,
            )

        self.job_manager.submit_task(job_id, task)
        return job_id
