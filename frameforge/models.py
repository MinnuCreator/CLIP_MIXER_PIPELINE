from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class ImageAsset:
    id: str
    file_path: str
    filename: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    scene_id: str
    image_id: str
    start: float
    end: float
    narration: str
    confidence: float
    effects: list[str] = field(default_factory=list)
    transition: str = "cut"

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 4)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


@dataclass
class Timeline:
    audio_file: str
    audio_duration: float
    scenes: list[Scene]
    alignment_method: str
    transcript: list[TranscriptSegment] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "audio": {"file": self.audio_file, "duration": self.audio_duration},
            "alignment_method": self.alignment_method,
            "scenes": [scene.as_dict() for scene in self.scenes],
            "transcript": [asdict(segment) for segment in self.transcript],
        }
