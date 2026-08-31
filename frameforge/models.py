"""Data structures and serialization models for FrameForge V3."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


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

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "words": [asdict(w) for w in self.words],
        }


@dataclass
class ImageAsset:
    id: str
    file_path: str
    filename: str
    prompt: str = ""
    prompt_source: str = "auto_generated"  # 'supplied_file', 'supplied_manual', 'auto_generated'
    metadata: dict[str, Any] = field(default_factory=dict)
    file_hash: Optional[str] = None
    phash: Optional[str] = None
    width: int = 1080
    height: int = 1920
    quality_score: float = 1.0
    blur_score: float = 1.0
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    normalized_path: Optional[str] = None


@dataclass
class Scene:
    scene_id: str
    image_id: str
    start: float
    end: float
    narration: str
    confidence: float
    rerank_breakdown: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "image_id": self.image_id,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "narration": self.narration,
            "confidence": round(self.confidence, 4),
            "rerank_breakdown": self.rerank_breakdown,
        }


@dataclass
class Timeline:
    audio_file: str
    audio_duration: float
    scenes: list[Scene]
    alignment_method: str
    raw_transcript: list[TranscriptSegment] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        if not self.scenes:
            return 0.0
        return self.scenes[-1].end - self.scenes[0].start

    def as_dict(self) -> dict[str, Any]:
        return {
            "audio": {
                "file": str(Path(self.audio_file)),
                "duration": round(self.audio_duration, 4),
            },
            "alignment_method": self.alignment_method,
            "scenes": [s.as_dict() for s in self.scenes],
            "raw_transcript": [t.as_dict() for t in self.raw_transcript],
        }


@dataclass
class JobState:
    job_id: str
    project_id: str
    job_type: str  # 'analyze' or 'render'
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: int = 0
    stage: str = "initialized"
    message: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "job_type": self.job_type,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
