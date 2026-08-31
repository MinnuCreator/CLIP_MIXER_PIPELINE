from __future__ import annotations
import json, sqlite3, threading
from pathlib import Path


class TimelineStore:
    def __init__(self, database: str | Path = "frameforge.db"):
        # ThreadingHTTPServer handles each API request in a worker thread.
        # SQLite's default connection guard rejects the main-thread connection
        # there, so serialize access to this shared local connection.
        self.connection = sqlite3.connect(database, check_same_thread=False)
        self.lock = threading.RLock()
        with self.lock:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS sync_jobs ("
                "id TEXT PRIMARY KEY, "
                "project_id TEXT NOT NULL, "
                "audio_file TEXT, "
                "audio_duration REAL, "
                "timeline TEXT, "
                "status TEXT NOT NULL, "
                "alignment_method TEXT, "
                "created_at TEXT DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TEXT DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sync_jobs_project ON sync_jobs(project_id, updated_at DESC)"
            )
            self.connection.commit()

    def save(self, job_id: str, project_id: str, audio_file: str, duration: float, timeline: dict, status: str, method: str) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT INTO sync_jobs(id,project_id,audio_file,audio_duration,timeline,status,alignment_method,updated_at) "
                "VALUES(?,?,?,?,?,?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(id) DO UPDATE SET "
                "audio_file=excluded.audio_file, "
                "audio_duration=excluded.audio_duration, "
                "timeline=excluded.timeline, "
                "status=excluded.status, "
                "alignment_method=excluded.alignment_method, "
                "updated_at=CURRENT_TIMESTAMP",
                (job_id, project_id, audio_file, duration, json.dumps(timeline), status, method),
            )
            self.connection.commit()

    def get_latest(self, project_id: str) -> dict | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT id,audio_file,audio_duration,timeline,status,alignment_method,created_at,updated_at "
                "FROM sync_jobs WHERE project_id=? ORDER BY updated_at DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        if not row:
            return None
        return dict(
            zip(
                ("id", "audio_file", "audio_duration", "timeline", "status", "alignment_method", "created_at", "updated_at"),
                row,
            )
        ) | {"timeline": json.loads(row[3])}

    def close(self) -> None:
        with self.lock:
            self.connection.close()
