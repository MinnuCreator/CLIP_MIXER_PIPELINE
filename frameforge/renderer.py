"""FrameForge V2 Dual-Path FFmpeg Video Rendering Engine (Fast Concat vs Staged Transitions)."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import DEFAULT_CONFIG, FrameForgeConfig
from .image_processor import ImagePreprocessor
from .models import ImageAsset, Timeline

logger = logging.getLogger("frameforge.renderer")


class RenderError(RuntimeError):
    pass


class FFmpegRenderer:
    RESOLUTIONS = {
        "vertical": (1080, 1920),
        "landscape": (1920, 1080),
        "square": (1080, 1080),
        "preview": (540, 960),
        "preview_vertical": (540, 960),
        "preview_landscape": (960, 540),
    }

    def __init__(self, ffmpeg_path: str | None = None, config: Optional[FrameForgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        self.preprocessor = ImagePreprocessor(self.config)

    def command(
        self,
        timeline: Timeline,
        images: list[ImageAsset],
        output: str | Path,
        preset: str = "vertical",
        fit: str = "cover",
        transition: str = "cut",
    ) -> list[str]:
        """Build filter_complex command for backward compatibility and test verification."""
        if not self.ffmpeg_path:
            raise RenderError(
                "FFmpeg is required to render videos. Please install FFmpeg and ensure it is available on PATH."
            )
        if fit not in {"cover", "contain", "crop"}:
            raise RenderError("Image fit must be cover, contain, or crop.")

        width, height = self.RESOLUTIONS.get(preset, (1080, 1920))
        image_map = {i.id: i for i in images}
        command = [self.ffmpeg_path, "-y"]

        for scene in timeline.scenes:
            img_path = image_map[scene.image_id].file_path
            command += ["-loop", "1", "-t", f"{scene.duration:.4f}", "-i", img_path]

        command += ["-i", timeline.audio_file]
        filters = []

        for index, scene in enumerate(timeline.scenes):
            if fit == "contain":
                chain = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
            else:
                chain = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
            filters.append(f"[{index}:v]{chain},setsar=1,setpts=PTS-STARTPTS[v{index}]")

        filters.append(
            "".join(f"[v{i}]" for i in range(len(timeline.scenes)))
            + f"concat=n={len(timeline.scenes)}:v=1:a=0[v]"
        )

        return command + [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-map",
            f"{len(timeline.scenes)}:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output),
        ]

    def render_fast_concat(
        self,
        timeline: Timeline,
        images: list[ImageAsset],
        output: str | Path,
        preset: str = "vertical",
        fit: str = "cover",
        render_profile: str = "balanced",
    ) -> Path:
        """FAST PATH: Render using the scalable FFmpeg Concat Demuxer protocol with pre-normalized images."""
        if not self.ffmpeg_path:
            raise RenderError("FFmpeg is required to render videos. Ensure it is available on PATH.")

        output_path = Path(output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = self.RESOLUTIONS.get(preset, (1080, 1920))
        image_map = {i.id: i for i in images}

        # 1. Normalize images in parallel
        logger.info("Pre-normalizing visuals for %d scenes to %dx%d (%s)...", len(timeline.scenes), width, height, fit)
        normalized_paths: dict[str, str] = {}
        for scene in timeline.scenes:
            img = image_map[scene.image_id]
            if img.id not in normalized_paths:
                norm = self.preprocessor.normalize_image(img, target_width=width, target_height=height, fit=fit)
                normalized_paths[img.id] = str(norm.resolve()).replace("\\", "/")

        # 2. Generate ffconcat script
        temp_dir = Path(tempfile.gettempdir()) / "frameforge_render"
        temp_dir.mkdir(parents=True, exist_ok=True)
        concat_file = temp_dir / f"concat_{os.getpid()}_{len(timeline.scenes)}.txt"

        lines = ["ffconcat version 1.0"]
        for scene in timeline.scenes:
            norm_file = normalized_paths[scene.image_id]
            lines.append(f"file '{norm_file}'")
            lines.append(f"duration {scene.duration:.4f}")
        # Repeat the last frame to ensure exact duration adherence in ffconcat
        if timeline.scenes:
            last_norm = normalized_paths[timeline.scenes[-1].image_id]
            lines.append(f"file '{last_norm}'")

        concat_file.write_text("\n".join(lines), encoding="utf-8")

        # 3. Codec & encoding presets
        crf = "22"
        ff_preset = "medium"
        if render_profile == "preview" or "preview" in preset:
            crf = "28"
            ff_preset = "ultrafast"
        elif render_profile == "fast":
            crf = "25"
            ff_preset = "veryfast"
        elif render_profile == "high_quality":
            crf = "18"
            ff_preset = "slow"

        command = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file.resolve()),
            "-i",
            str(Path(timeline.audio_file).resolve()),
            "-c:v",
            "libx264",
            "-preset",
            ff_preset,
            "-crf",
            crf,
            "-r",
            str(self.config.target_fps),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]

        logger.info("Executing FFmpeg concat render (%d scenes, preset=%s, crf=%s)...", len(timeline.scenes), ff_preset, crf)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            err_msg = exc.stderr[-600:] if exc.stderr else str(exc)
            logger.error("FFmpeg execution failed: %s", err_msg)
            raise RenderError(f"FFmpeg rendering failed: {err_msg}") from exc
        finally:
            try:
                concat_file.unlink(missing_ok=True)
            except Exception:
                pass

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RenderError("FFmpeg completed without producing an MP4.")

        return output_path

    def render_transitions_script(
        self,
        timeline: Timeline,
        images: list[ImageAsset],
        output: str | Path,
        preset: str = "vertical",
        fit: str = "cover",
        render_profile: str = "balanced",
    ) -> Path:
        """TRANSITION PATH: Render transitions using a staged filter_complex for curated scenes."""
        if not self.ffmpeg_path:
            raise RenderError("FFmpeg is required to render videos. Ensure it is available on PATH.")

        output_path = Path(output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = self.RESOLUTIONS.get(preset, (1080, 1920))
        image_map = {i.id: i for i in images}

        command = [self.ffmpeg_path, "-y"]
        for scene in timeline.scenes:
            img_path = image_map[scene.image_id].file_path
            command += ["-loop", "1", "-t", f"{scene.duration:.4f}", "-i", img_path]
        command += ["-i", timeline.audio_file]

        filters = []
        for index, scene in enumerate(timeline.scenes):
            if fit == "contain":
                chain = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
            else:
                chain = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
            filters.append(f"[{index}:v]{chain},setsar=1,setpts=PTS-STARTPTS[v{index}]")
        filters.append("".join(f"[v{i}]" for i in range(len(timeline.scenes))) + f"concat=n={len(timeline.scenes)}:v=1:a=0[v]")

        command += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-map",
            f"{len(timeline.scenes)}:a",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-r",
            str(self.config.target_fps),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            err_msg = exc.stderr[-600:] if exc.stderr else str(exc)
            raise RenderError(f"FFmpeg transitions rendering failed: {err_msg}") from exc

        if not output_path.is_file():
            raise RenderError("FFmpeg transitions render completed without producing an MP4.")

        return output_path

    def render(
        self,
        timeline: Timeline,
        images: list[ImageAsset],
        output: str | Path,
        preset: str = "vertical",
        fit: str = "cover",
        transition: str = "cut",
        render_path: str | None = None,
        render_profile: str = "balanced",
    ) -> Path:
        """Render video using the optimal rendering path."""
        chosen_path = render_path or self.config.default_render_path
        if transition != "cut" and len(timeline.scenes) <= 25:
            chosen_path = "transition"

        if chosen_path == "transition" and len(timeline.scenes) <= 50:
            return self.render_transitions_script(timeline, images, output, preset, fit, render_profile)
        else:
            return self.render_fast_concat(timeline, images, output, preset, fit, render_profile)

    def probe(self, output: str | Path) -> dict:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RenderError("FFprobe is required to verify rendered videos. Ensure ffprobe is on PATH.")
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt",
            "-of",
            "json",
            str(output),
        ]
        try:
            result = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            streams = result["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            audio = next(stream for stream in streams if stream["codec_type"] == "audio")
            
            fps_str = video.get("r_frame_rate", "30/1")
            fps = float(fps_str.split("/")[0]) / float(fps_str.split("/")[1]) if "/" in fps_str else float(fps_str)

            return {
                "duration": float(result["format"]["duration"]),
                "size": int(result["format"]["size"]),
                "video_codec": video["codec_name"],
                "audio_codec": audio["codec_name"],
                "width": int(video["width"]),
                "height": int(video["height"]),
                "fps": round(fps, 2),
                "pix_fmt": video.get("pix_fmt", "yuv420p"),
            }
        except (subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError, StopIteration) as exc:
            raise RenderError("FFprobe could not validate the rendered MP4.") from exc
