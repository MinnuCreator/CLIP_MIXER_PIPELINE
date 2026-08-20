import tempfile
import threading
import unittest
from pathlib import Path
from frameforge.storage import TimelineStore


class TimelineStoreThreadTests(unittest.TestCase):
    def test_request_thread_can_save_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TimelineStore(Path(directory) / "test.db")
            failure = []
            def save():
                try: store.save("job", "project", "voice.mp3", 1.0, {"audio": {}}, "ready", "fallback")
                except Exception as exc: failure.append(exc)
            thread = threading.Thread(target=save)
            thread.start(); thread.join()
            self.assertEqual(failure, [])
            self.assertIsNotNone(store.get_latest("project"))
            store.close()
