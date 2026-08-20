import tempfile
import unittest
from pathlib import Path

from frameforge.server import create_service
from frameforge.storage import TimelineStore


class ServiceInitializationTests(unittest.TestCase):
    def test_database_path_creates_a_timeline_store_not_a_path(self):
        with tempfile.TemporaryDirectory() as directory:
            service = create_service(Path(directory) / "frameforge.db")
            self.assertIsInstance(service.store, TimelineStore)
            self.assertTrue(callable(service.store.save))
            service.store.close()
