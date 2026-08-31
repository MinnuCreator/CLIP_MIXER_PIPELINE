import time
import unittest
from frameforge.jobs import JobManager


class JobManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = JobManager()

    def test_job_lifecycle(self):
        job = self.manager.create_job("test_proj", job_type="analyze")
        self.assertEqual(job.status, "pending")
        self.assertEqual(job.progress, 0)

        self.manager.update_progress(job.job_id, "transcribing", 40, "Transcribing audio")
        updated = self.manager.get_job(job.job_id)
        self.assertEqual(updated.status, "processing")
        self.assertEqual(updated.stage, "transcribing")
        self.assertEqual(updated.progress, 40)

        self.manager.complete_job(job.job_id, {"scenes_count": 5})
        completed = self.manager.get_job(job.job_id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.progress, 100)
        self.assertEqual(completed.result["scenes_count"], 5)

    def test_job_async_execution(self):
        job = self.manager.create_job("async_proj", job_type="analyze")

        def background_work():
            time.sleep(0.05)
            return {"status": "ok"}

        self.manager.submit_task(job.job_id, background_work)
        time.sleep(0.15)
        completed = self.manager.get_job(job.job_id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.result, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
