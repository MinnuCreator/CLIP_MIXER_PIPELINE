from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from .models import TranscriptSegment, Word


class TranscriptionError(RuntimeError):
    pass


class WhisperTranscriber:
    """Lazy local faster-whisper adapter; never contacts a paid API."""

    def __init__(self, cache_dir: str | Path = ".frameforge-cache", model_name: str = "base"):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name

    def transcribe(self, audio_file: str | Path) -> list[TranscriptSegment]:
        return self.transcribe_with_duration(audio_file)[0]

    def transcribe_with_duration(self, audio_file: str | Path) -> tuple[list[TranscriptSegment], float]:
        audio_file = Path(audio_file)
        if not audio_file.is_file():
            raise TranscriptionError(f"Audio file does not exist: {audio_file}")
        key = hashlib.sha256(audio_file.read_bytes()).hexdigest()
        cache_file = self.cache_dir / f"transcript-{key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            # Support earlier segment-only cache entries, but still obtain an
            # actual media duration rather than treating narration end as audio
            # duration.
            if isinstance(cached, dict):
                return self._decode(cached["segments"]), float(cached["audio_duration"])
            return self._decode(cached), self.probe_duration(audio_file)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError("Local transcription requires faster-whisper. Install requirements.txt; no API key is needed.") from exc
        try:
            model = WhisperModel(self.model_name, device="auto", compute_type="int8")
            segments, info = model.transcribe(str(audio_file), word_timestamps=True, vad_filter=True)
            result = [TranscriptSegment(s.text.strip(), float(s.start), float(s.end), [Word(w.word.strip(), float(w.start), float(w.end)) for w in (s.words or [])]) for s in segments]
        except Exception as exc:
            raise TranscriptionError("Local Whisper transcription failed. Check the audio format and local model installation.") from exc
        duration = float(getattr(info, "duration", 0) or 0)
        if duration <= 0:
            duration = self.probe_duration(audio_file)
        if not result:
            raise TranscriptionError("Whisper produced no narration segments for this audio.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"audio_duration": duration, "segments": [self._encode(s) for s in result]}), encoding="utf-8")
        return result, duration

    @staticmethod
    def probe_duration(audio_file: Path) -> float:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise TranscriptionError("FFprobe is required to determine the uploaded audio duration. Install FFmpeg and ensure ffprobe is on PATH.")
        command = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio_file)]
        try:
            data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            duration = float(data["format"]["duration"])
        except (subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TranscriptionError("Could not determine the uploaded audio duration with FFprobe.") from exc
        if duration <= 0:
            raise TranscriptionError("Uploaded audio has no positive duration.")
        return duration

    @staticmethod
    def _encode(segment: TranscriptSegment) -> dict:
        return {"text": segment.text, "start": segment.start, "end": segment.end, "words": [w.__dict__ for w in segment.words]}

    @staticmethod
    def _decode(data: list[dict]) -> list[TranscriptSegment]:
        return [TranscriptSegment(x["text"], x["start"], x["end"], [Word(**w) for w in x.get("words", [])]) for x in data]


class SemanticSegmenter:
    """Uses punctuation and pauses rather than arbitrary duration buckets."""

    sentence_end = re.compile(r"[.!?][\"']?$")

    def split(self, raw: list[TranscriptSegment], pause_seconds: float = 0.7) -> list[TranscriptSegment]:
        output: list[TranscriptSegment] = []
        current: TranscriptSegment | None = None
        for source in raw:
            words = source.words or [Word(source.text, source.start, source.end)]
            for word in words:
                if current is None:
                    current = TranscriptSegment("", word.start, word.end, [])
                current.words.append(word)
                current.text = (current.text + " " + word.word).strip()
                current.end = word.end
                following_pause = word is words[-1] and source is not raw[-1] and raw[raw.index(source) + 1].start - word.end >= pause_seconds
                if self.sentence_end.search(word.word) or following_pause:
                    output.append(current)
                    current = None
        if current:
            output.append(current)
        return output or raw
