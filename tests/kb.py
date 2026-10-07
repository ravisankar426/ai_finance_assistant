"""Build a real HybridRetriever over the shipped knowledge base with offline embeddings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.core.config import RAGConfig, get_settings
from src.rag.index import build_index
from src.rag.knowledge_base import load_articles
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import FaissStore
from tests.fakes import HashingEmbeddings


@lru_cache(maxsize=1)
def offline_store(tmp_root: str = "") -> FaissStore:
    """Index the real articles with hashing embeddings (cached once per test session)."""
    import tempfile

    base = Path(tmp_root or tempfile.mkdtemp(prefix="kb-test-"))
    rag = get_settings().rag.model_copy(
        update={"index_dir": base / "index", "embedding_cache_dir": base / "cache"}
    )
    return build_index(load_articles(rag.knowledge_base_dir), HashingEmbeddings(), "hashing", rag)


def offline_retriever(**overrides: float) -> HybridRetriever:
    """Hybrid retriever over the real KB with offline embeddings (production gate values)."""
    params: dict[str, float] = {"min_cosine": 0.3, "min_keyword_coverage": 0.5} | overrides
    return HybridRetriever(offline_store(), HashingEmbeddings(), **params)  # type: ignore[arg-type]


def rag_config(base: Path) -> RAGConfig:
    """RAG settings writing index/cache under ``base``."""
    return get_settings().rag.model_copy(
        update={"index_dir": base / "index", "embedding_cache_dir": base / "cache"}
    )
