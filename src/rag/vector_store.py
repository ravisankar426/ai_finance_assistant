"""FAISS vector index over knowledge-base chunks (ADR-06).

``IndexFlatIP`` on L2-normalized vectors = exact cosine-similarity search. Exact search is
the right choice at this size (~1k vectors, sub-millisecond); approximate indexes (IVF,
HNSW) only pay off around 100k+ vectors.

Persistence: the FAISS index via ``faiss.write_index`` and chunk metadata as JSON — no
pickle, so loading an index file can't execute code.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.core.errors import KnowledgeBaseError
from src.rag.chunker import Chunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Return a float32 copy with unit-length rows (so inner product = cosine)."""
    out = np.array(vectors, dtype=np.float32, copy=True)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    faiss.normalize_L2(out)
    return out


class FaissStore:
    """Chunks + their normalized vectors + an exact inner-product index."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, manifest: dict[str, Any]) -> None:
        if len(chunks) != len(vectors):
            raise KnowledgeBaseError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = normalize(vectors)
        self.manifest = manifest
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        return int(self.vectors.shape[1])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` (chunk index, cosine) pairs, best first."""
        q = normalize(query_vector)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    def cosine(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query to *every* chunk (cheap at this size)."""
        return np.asarray(self.vectors @ normalize(query_vector)[0], dtype=np.float32)

    def save(self, directory: Path) -> None:
        """Write index, chunk metadata, and manifest to ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        (directory / CHUNKS_FILE).write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1)
        )
        manifest = {**self.manifest, "chunks": len(self.chunks), "dimension": self.dimension}
        manifest.setdefault("built_at", datetime.now(UTC).isoformat(timespec="seconds"))
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1))

    @classmethod
    def load(cls, directory: Path) -> FaissStore:
        """Load a store saved with :meth:`save`."""
        try:
            index = faiss.read_index(str(directory / INDEX_FILE))
            chunks = [Chunk(**c) for c in json.loads((directory / CHUNKS_FILE).read_text())]
            manifest = json.loads((directory / MANIFEST_FILE).read_text())
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise KnowledgeBaseError(f"Could not load index from {directory}: {exc}") from exc
        vectors = index.reconstruct_n(0, index.ntotal)
        return cls(chunks, vectors, manifest)

    @staticmethod
    def read_manifest(directory: Path) -> dict[str, Any] | None:
        """Return the saved manifest, or None if there is no usable index."""
        path = directory / MANIFEST_FILE
        if not (path.exists() and (directory / INDEX_FILE).exists()):
            return None
        try:
            return dict(json.loads(path.read_text()))
        except ValueError:
            return None
