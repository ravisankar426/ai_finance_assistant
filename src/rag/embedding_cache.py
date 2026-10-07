"""On-disk embedding cache keyed by (model, content hash) (REQ-RAG-02).

Re-ingesting the knowledge base only pays for chunks whose text changed: everything else
is read back from ``<cache_dir>/<model>.npz``. Stored as plain NumPy arrays (no pickle).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from src.utils.logging import get_logger

log = get_logger(__name__)


class EmbeddingCache:
    """Embed texts, reusing previously computed vectors."""

    def __init__(self, cache_dir: Path, model: str, *, batch_size: int = 128) -> None:
        self.path = cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', model)}.npz"
        self.model = model
        self.batch_size = batch_size
        self._vectors: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                self._vectors = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str], embedder: Embeddings) -> tuple[np.ndarray, int]:
        """Return a (len(texts), dim) float32 matrix and how many texts were newly embedded."""
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._vectors]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = embedder.embed_documents([texts[i] for i in batch])
            for i, vec in zip(batch, vectors, strict=True):
                self._vectors[keys[i]] = np.asarray(vec, dtype=np.float32)
        if missing:
            self._save()
            log.info("embeddings_computed", new=len(missing), cached=len(texts) - len(missing))
        return np.vstack([self._vectors[k] for k in keys]), len(missing)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._vectors), dtype=str)
        np.savez(self.path, keys=keys, vectors=np.vstack(list(self._vectors.values())))
