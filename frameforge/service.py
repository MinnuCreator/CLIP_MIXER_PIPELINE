from __future__ import annotations
import uuid
import logging
from .alignment import SceneAlignmentEngine
from .models import ImageAsset, Scene, Timeline
from .renderer import FFmpegRenderer, RenderError
from .storage import TimelineStore
from .transcript import SemanticSegmenter, WhisperTranscriber
from .validation import validate_timeline


logger = logging.getLogger("frameforge.service")


class FrameForgeService:
    def __init__(self, store=None): self.store = store or TimelineStore()
    def analyze(self, project_id: str, audio_file: str, audio_duration: float | None, images: list[dict], transcript=None) -> dict:
        assets = [ImageAsset(**item) for item in images]
        if transcript is None:
            raw, detected_duration = WhisperTranscriber().transcribe_with_duration(audio_file)
            # Media metadata, not a browser form field, is authoritative.
            audio_duration = detected_duration
        elif audio_duration is None or audio_duration <= 0:
            raise ValueError("A positive measured audio duration is required with a supplied transcript.")
        else:
            raw = transcript
        segments = SemanticSegmenter().split(raw)
        logger.info("Timeline input: audio=%s duration=%.6f raw_segments=%d narration_segments=%d images=%d", audio_file, audio_duration or 0, len(raw), len(segments), len(assets))
        scenes, method = SceneAlignmentEngine().align(segments, assets, audio_duration)
        logger.info("Timeline candidates: scenes=%d alignment=%s", len(scenes), method)
        timeline = validate_timeline(Timeline(audio_file, audio_duration, scenes, method, raw), assets, verify_files=False)
        result = timeline.as_dict()
        # Keep the exact file/prompt associations that produced this timeline
        # for inspection, manual editing, and a later render request.
        result["images"] = [{"id": x.id, "file_path": x.file_path, "filename": x.filename, "prompt": x.prompt, "metadata": x.metadata} for x in assets]
        self.store.save(str(uuid.uuid4()), project_id, audio_file, audio_duration, result, "ready", method)
        return result
    def update(self, project_id: str, timeline: dict) -> None:
        audio = timeline["audio"]; self.store.save(str(uuid.uuid4()), project_id, audio["file"], audio["duration"], timeline, "ready", timeline.get("alignment_method", "manual"))

    def render(self, project_id: str, output: str, preset="vertical", fit="cover", transition="cut", ffmpeg_path=None) -> dict:
        job = self.store.get_latest(project_id)
        if not job:
            raise ValueError("Analyze before rendering.")
        saved = job["timeline"]
        assets = [ImageAsset(**item) for item in saved.get("images", [])]
        if not assets:
            raise ValueError("The saved timeline has no image associations. Analyze again before rendering.")
        timeline = Timeline(saved["audio"]["file"], float(saved["audio"]["duration"]), [Scene(**{key: value for key, value in scene.items() if key != "duration"}) for scene in saved["scenes"]], saved.get("alignment_method", "manual"))
        validate_timeline(timeline, assets, verify_files=True)
        renderer = FFmpegRenderer(ffmpeg_path)
        generated = renderer.render(timeline, assets, output, preset, fit, transition)
        metadata = renderer.probe(generated)
        if metadata["video_codec"] != "h264" or metadata["audio_codec"] != "aac":
            raise RenderError("Rendered MP4 did not contain H.264 video and AAC audio.")
        if abs(metadata["duration"] - timeline.audio_duration) > 0.15:
            raise RenderError("Rendered MP4 duration does not match the authoritative voice-over duration.")
        return {"output": str(generated), "metadata": metadata}
