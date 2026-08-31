import tempfile
import unittest
from pathlib import Path
from frameforge.config import FrameForgeConfig
from frameforge.embeddings import EmbeddingEngine, VectorIndex


class EmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = FrameForgeConfig(cache_dir=Path(self.temp_dir.name))
        self.engine = EmbeddingEngine(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_lexical_similarity(self):
        s1 = self.engine.lexical_similarity("historic lighthouse ocean", "a lonely lighthouse standing beside ocean")
        self.assertGreater(s1, 0.4)
        s2 = self.engine.lexical_similarity("historic lighthouse ocean", "space rocket orbiting mars")
        self.assertEqual(s2, 0.0)

    def test_vector_index_search(self):
        # Test vector index logic directly with mock vectors
        import numpy as np
        # 3 items, 4 dimensions
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.707, 0.707, 0.0, 0.0]
        ], dtype=np.float32)
        idx = VectorIndex(vecs, ["id1", "id2", "id3"])
        
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        res = idx.search(query, top_k=2)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0][0], 0)  # item 0 has highest similarity (1.0)
        self.assertAlmostEqual(res[0][0][1], 1.0, places=3)
        self.assertEqual(res[0][1][0], 2)  # item 2 is next highest (0.707)


if __name__ == "__main__":
    unittest.main()
