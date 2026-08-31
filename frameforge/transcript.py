"""FrameForge V2 Local Speech-to-Text Transcription & Semantic Segmentation."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_CONFIG, FrameForgeConfig
from .models import TranscriptSegment, Word

logger = logging.getLogger("frameforge.transcript")


class TranscriptionError(RuntimeError):
    pass


class WhisperTranscriber:
    """Lazy local faster-whisper adapter with singleton model reuse & streaming hashing."""

    _model_instance: Any = None
    _loaded_model_name: str = ""

    def __init__(self, cache_dir: str | Path | None = None, model_name: str | None = None, config: Optional[FrameForgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.cache_dir = Path(cache_dir or self.config.cache_dir)
        self.transcripts_cache_dir = self.cache_dir / "transcripts"
        self.transcripts_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or self.config.whisper_model

    @staticmethod
    def compute_audio_hash(audio_file: Path) -> str:
        """Compute streaming SHA-256 hash without loading full audio into RAM."""
        hasher = hashlib.sha256()
        with open(audio_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def transcribe(self, audio_file: str | Path) -> list[TranscriptSegment]:
        return self.transcribe_with_duration(audio_file)[0]

    def transcribe_with_duration(self, audio_file: str | Path) -> tuple[list[TranscriptSegment], float]:
        audio_file = Path(audio_file)
        if not audio_file.is_file():
            raise TranscriptionError(f"Audio file does not exist: {audio_file}")

        audio_hash = self.compute_audio_hash(audio_file)
        key = f"{audio_hash}_whisper-{self.model_name}"
        
        # Check both the new organized cache and the legacy root cache
        cache_file = self.transcripts_cache_dir / f"transcript-{key}.json"
        legacy_cache = self.cache_dir / f"transcript-{key}.json"

        target_cache = cache_file if cache_file.is_file() else (legacy_cache if legacy_cache.is_file() else None)

        if target_cache:
            try:
                cached = json.loads(target_cache.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and "segments" in cached:
                    return self._decode(cached["segments"]), float(cached["audio_duration"])
                if isinstance(cached, list):
                    return self._decode(cached), self.probe_duration(audio_file)
            except Exception as exc:
                logger.warning("Failed to read cached transcript: %s", exc)

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "Local transcription requires faster-whisper. Install requirements.txt; no API key is needed."
            ) from exc

        # Singleton model loading
        if WhisperTranscriber._model_instance is None or WhisperTranscriber._loaded_model_name != self.model_name:
            device = self.config.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            logger.info("Loading faster-whisper model '%s' on %s (%s)", self.model_name, device, self.config.whisper_compute_type)
            WhisperTranscriber._model_instance = WhisperModel(
                self.model_name, device=device, compute_type=self.config.whisper_compute_type
            )
            WhisperTranscriber._loaded_model_name = self.model_name

        try:
            model = WhisperTranscriber._model_instance
            segments, info = model.transcribe(str(audio_file), word_timestamps=True, vad_filter=True)
            result = [
                TranscriptSegment(
                    s.text.strip(),
                    float(s.start),
                    float(s.end),
                    [Word(w.word.strip(), float(w.start), float(w.end)) for w in (s.words or [])],
                )
                for s in segments
            ]
        except Exception as exc:
            raise TranscriptionError("Local Whisper transcription failed. Check the audio format and local model installation.") from exc

        duration = float(getattr(info, "duration", 0) or 0)
        if duration <= 0:
            duration = self.probe_duration(audio_file)

        if not result:
            raise TranscriptionError("Whisper produced no narration segments for this audio.")

        # Persist to cache
        cache_data = {"audio_duration": duration, "segments": [self._encode(s) for s in result]}
        try:
            cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save transcript cache: %s", exc)

        return result, duration

    @staticmethod
    def probe_duration(audio_file: Path) -> float:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise TranscriptionError(
                "FFprobe is required to determine the uploaded audio duration. Install FFmpeg and ensure ffprobe is on PATH."
            )
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_file),
        ]
        try:
            res = subprocess.run(command, check=True, capture_output=True, text=True)
            data = json.loads(res.stdout)
            duration = float(data["format"]["duration"])
        except (subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TranscriptionError("Could not determine the uploaded audio duration with FFprobe.") from exc
        if duration <= 0:
            raise TranscriptionError("Uploaded audio has no positive duration.")
        return duration

    @staticmethod
    def _encode(segment: TranscriptSegment) -> dict:
        return {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
            "words": [w.__dict__ for w in segment.words],
        }

    @staticmethod
    def _decode(data: list[dict]) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                x["text"],
                x["start"],
                x["end"],
                [Word(**w) for w in x.get("words", [])],
            )
            for x in data
        ]


class SemanticSegmenter:
    """Uses punctuation and pauses rather than arbitrary duration buckets."""

    sentence_end = re.compile(r"[.!?][\"']?$")

    def __init__(self, pause_seconds: float = 0.7):
        self.pause_seconds = pause_seconds

    def split(self, raw: list[TranscriptSegment], pause_seconds: float | None = None) -> list[TranscriptSegment]:
        if not raw:
            return []
        threshold = pause_seconds if pause_seconds is not None else self.pause_seconds
        output: list[TranscriptSegment] = []
        current: TranscriptSegment | None = None
        num_raw = len(raw)

        for src_idx, source in enumerate(raw):
            words = source.words or [Word(source.text, source.start, source.end)]
            num_words = len(words)
            for word_idx, word in enumerate(words):
                if current is None:
                    current = TranscriptSegment("", word.start, word.end, [])
                current.words.append(word)
                current.text = (current.text + " " + word.word).strip()
                current.end = word.end

                # Determine if followed by an audible pause
                following_pause = False
                if word_idx == num_words - 1 and src_idx < num_raw - 1:
                    following_pause = (raw[src_idx + 1].start - word.end) >= threshold

                if self.sentence_end.search(word.word) or following_pause:
                    output.append(current)
                    current = None

        if current:
            output.append(current)
        return output or raw
