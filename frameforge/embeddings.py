"""FrameForge V2 Singleton Embedding Service & Vector Index with Versioned Keys."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_CONFIG, FrameForgeConfig

logger = logging.getLogger("frameforge.embeddings")


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1}


class VectorIndex:
    """Fast in-memory vector index with NumPy cosine similarity search."""

    def __init__(self, vectors: Any, ids: list[str]):
        self.ids = ids
        self.vectors = vectors  # numpy array of shape (N, D)

    def search(self, query_vectors: Any, top_k: int = 20) -> list[list[tuple[int, float]]]:
        """Vectorized Top-K cosine similarity search.
        
        Args:
            query_vectors: numpy array of shape (M, D), normalized.
            top_k: number of nearest candidates to return per query.
        Returns:
            List of length M, each containing up to top_k (index, score) pairs.
        """
        try:
            import numpy as np

            similarity_matrix = np.dot(query_vectors, self.vectors.T)
            num_candidates = self.vectors.shape[0]
            actual_k = min(top_k, num_candidates)

            results: list[list[tuple[int, float]]] = []
            for row in similarity_matrix:
                if actual_k < num_candidates:
                    top_indices = np.argpartition(row, -actual_k)[-actual_k:]
                    sorted_top = top_indices[np.argsort(row[top_indices])[::-1]]
                else:
                    sorted_top = np.argsort(row)[::-1]

                results.append([(int(idx), float(row[idx])) for idx in sorted_top])
            return results
        except Exception as exc:
            logger.warning("NumPy vector search failed: %s. Using fallback.", exc)
            return self._fallback_search(query_vectors, top_k)

    def _fallback_search(self, query_vectors: Any, top_k: int) -> list[list[tuple[int, float]]]:
        results: list[list[tuple[int, float]]] = []
        for q in query_vectors:
            scores = []
            for idx, v in enumerate(self.vectors):
                dot = sum(a * b for a, b in zip(q, v))
                scores.append((idx, float(dot)))
            scores.sort(key=lambda x: x[1], reverse=True)
            results.append(scores[:top_k])
        return results


class EmbeddingEngine:
    _instance: Optional[EmbeddingEngine] = None
    _model: Any = None
    _model_loaded: bool = False
    _is_fallback: bool = False

    def __new__(cls, config: Optional[FrameForgeConfig] = None) -> EmbeddingEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine(config or DEFAULT_CONFIG)
        elif config is not None:
            cls._instance.config = config
            cls._instance.cache_dir = Path(config.cache_dir) / "v2" / "text_embeddings"
            cls._instance.cache_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance

    def _init_engine(self, config: FrameForgeConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir) / "v2" / "text_embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_model()

    def _load_model(self) -> None:
        if self._model_loaded:
            return

        device = self.config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading text embedding model %s on device: %s", self.config.embedding_model, device)
            self._model = SentenceTransformer(self.config.embedding_model, device=device)
            self._is_fallback = False
            self._model_loaded = True
        except Exception as exc:
            logger.warning("Could not load sentence-transformers (%s). Using fallback lexical similarity.", exc)
            self._model = None
            self._is_fallback = True
            self._model_loaded = True

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    def _cache_path(self, text: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw = f"{text.strip()}_text_{self.config.text_model_version}_{self.config.config_version}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> Any:
        """Encode a list of text strings into normalized vector embeddings with multi-attribute caching."""
        if self._is_fallback or self._model is None:
            return None

        import numpy as np

        embeddings: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check disk cache
        for idx, text in enumerate(texts):
            c_path = self._cache_path(text)
            if c_path.is_file():
                try:
                    data = json.loads(c_path.read_text(encoding="utf-8"))
                    embeddings[idx] = np.array(data, dtype=np.float32)
                except Exception:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        # Batch encode any uncached items
        if uncached_texts:
            logger.info("Encoding %d new texts (batch_size=%d)", len(uncached_texts), batch_size)
            new_vectors = self._model.encode(
                uncached_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for local_idx, global_idx in enumerate(uncached_indices):
                vec = new_vectors[local_idx]
                embeddings[global_idx] = vec
                try:
                    self._cache_path(uncached_texts[local_idx]).write_text(
                        json.dumps(vec.tolist()), encoding="utf-8"
                    )
                except Exception as exc:
                    logger.warning("Failed to write embedding cache: %s", exc)

        return np.array(embeddings, dtype=np.float32)

    def create_index(self, texts: list[str], ids: list[str]) -> Optional[VectorIndex]:
        """Build a VectorIndex from a list of text prompts and IDs."""
        if self._is_fallback:
            return None
        vectors = self.encode_texts(texts, batch_size=self.config.embedding_batch_size)
        if vectors is None:
            return None
        return VectorIndex(vectors, ids)

    def lexical_similarity(self, a: str, b: str) -> float:
        """Fallback token intersection similarity."""
        t1, t2 = _tokenize(a), _tokenize(b)
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / math.sqrt(len(t1) * len(t2))
