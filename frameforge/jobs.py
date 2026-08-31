"""FrameForge V2 Asynchronous Background Job Manager & Progress Tracker."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from .models import JobState

logger = logging.getLogger("frameforge.jobs")


class JobManager:
    _instance: Optional[JobManager] = None

    def __new__(cls, max_workers: int = 2) -> JobManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_manager(max_workers)
        return cls._instance

    def _init_manager(self, max_workers: int) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, JobState] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="frameforge-job")

    def create_job(self, project_id: str, job_type: str = "analyze") -> JobState:
        with self.lock:
            job_id = str(uuid.uuid4())
            job = JobState(
                job_id=job_id,
                project_id=project_id,
                job_type=job_type,
                status="pending",
                stage="queued",
                progress=0,
                message="Job queued for processing.",
            )
            self.jobs[job_id] = job
            return job

    def get_job(self, job_id: str) -> Optional[JobState]:
        with self.lock:
            return self.jobs.get(job_id)

    def update_progress(
        self,
        job_id: str,
        stage: str,
        progress: int,
        message: str = "",
    ) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = "processing"
                job.stage = stage
                job.progress = max(0, min(100, progress))
                if message:
                    job.message = message
                job.updated_at = time.time()
                logger.info("[JOB %s] %s (%d%%) - %s", job_id[:8], stage, progress, message)

    def complete_job(self, job_id: str, result: dict[str, Any], message: str = "Job completed successfully.") -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = "completed"
                job.stage = "completed"
                job.progress = 100
                job.message = message
                job.result = result
                job.updated_at = time.time()
                logger.info("[JOB %s] Completed.", job_id[:8])

    def fail_job(self, job_id: str, error: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = "failed"
                job.stage = "failed"
                job.error = error
                job.message = f"Failed: {error}"
                job.updated_at = time.time()
                logger.error("[JOB %s] Failed: %s", job_id[:8], error)

    def submit_task(self, job_id: str, task_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        def runner():
            try:
                self.update_progress(job_id, "starting", 5, "Initializing pipeline...")
                result = task_fn(*args, **kwargs)
                self.complete_job(job_id, result)
            except Exception as exc:
                logger.exception("Background job %s failed with exception: %s", job_id, exc)
                self.fail_job(job_id, str(exc))

        self.executor.submit(runner)
