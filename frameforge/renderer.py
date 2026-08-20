from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
from .models import ImageAsset, Timeline


class RenderError(RuntimeError): pass


class FFmpegRenderer:
    RESOLUTIONS = {"vertical": (1080, 1920), "landscape": (1920, 1080), "square": (1080, 1080)}
    def __init__(self, ffmpeg_path: str | None = None): self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
    def command(self, timeline: Timeline, images: list[ImageAsset], output: str | Path, preset="vertical", fit="cover", transition="cut") -> list[str]:
        if not self.ffmpeg_path: raise RenderError("FFmpeg is required to render videos. Please install FFmpeg and ensure it is available on PATH.")
        if fit not in {"cover", "contain", "crop"}: raise RenderError("Image fit must be cover, contain, or crop.")
        width, height = self.RESOLUTIONS[preset]
        image_map = {i.id: i for i in images}; command = [self.ffmpeg_path, "-y"]
        for scene in timeline.scenes: command += ["-loop", "1", "-t", f"{scene.duration:.4f}", "-i", image_map[scene.image_id].file_path]
        command += ["-i", timeline.audio_file]
        filters = []
        for index, scene in enumerate(timeline.scenes):
            if fit == "contain": chain = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
            else: chain = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
            filters.append(f"[{index}:v]{chain},setsar=1,setpts=PTS-STARTPTS[v{index}]")
        filters.append("".join(f"[v{i}]" for i in range(len(timeline.scenes))) + f"concat=n={len(timeline.scenes)}:v=1:a=0[v]")
        return command + ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", f"{len(timeline.scenes)}:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(output)]
    def render(self, *args, **kwargs) -> Path:
        command = self.command(*args, **kwargs); output = Path(command[-1])
        try: subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc: raise RenderError("FFmpeg rendering failed: " + exc.stderr[-500:]) from exc
        if not output.is_file(): raise RenderError("FFmpeg completed without producing an MP4.")
        return output

    def probe(self, output: str | Path) -> dict:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RenderError("FFprobe is required to verify rendered videos. Install FFmpeg and ensure ffprobe is on PATH.")
        command = [ffprobe, "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height", "-of", "json", str(output)]
        try:
            result = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            streams = result["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            audio = next(stream for stream in streams if stream["codec_type"] == "audio")
            return {"duration": float(result["format"]["duration"]), "size": int(result["format"]["size"]), "video_codec": video["codec_name"], "audio_codec": audio["codec_name"], "width": int(video["width"]), "height": int(video["height"])}
        except (subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError, StopIteration) as exc:
            raise RenderError("FFprobe could not validate the rendered MP4.") from exc
