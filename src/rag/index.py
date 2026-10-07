"""Build, persist, and reload the knowledge-base index (REQ-RAG-02, REQ-RAG-03).

``load_or_build`` is what the app calls at startup:
- if a saved index exists and was built from the *same* articles with the *same* embedding
  model and chunking settings (fingerprint match) -> load it, no API calls;
- otherwise -> chunk, embed (only new/changed chunks hit the API thanks to the cache),
  build, save.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from src.core.config import RAGConfig
from src.rag.chunker import Chunk, chunk_article
from src.rag.embedding_cache import EmbeddingCache
from src.rag.knowledge_base import Article
from src.rag.vector_store import FaissStore
from src.utils.logging import get_logger

log = get_logger(__name__)

# Bump when the saved chunk format changes, so old indexes are rebuilt automatically.
INDEX_SCHEMA_VERSION = 2


def fingerprint(articles: Sequence[Article], embedding_model: str, cfg: RAGConfig) -> str:
    """Hash of everything that determines the index contents."""
    payload = {
        "schema": INDEX_SCHEMA_VERSION,
        "model": embedding_model,
        "chunking": [cfg.chunk_max_words, cfg.chunk_overlap_words],
        "articles": sorted((a.id, a.model_dump_json()) for a in articles),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def chunk_all(articles: Sequence[Article], cfg: RAGConfig) -> list[Chunk]:
    """Chunk every article with the configured settings."""
    return [
        chunk
        for article in articles
        for chunk in chunk_article(
            article, max_words=cfg.chunk_max_words, overlap_words=cfg.chunk_overlap_words
        )
    ]


def build_index(
    articles: Sequence[Article], embeddings: Embeddings, embedding_model: str, cfg: RAGConfig
) -> FaissStore:
    """Chunk + embed (cached) + index, and save to ``cfg.index_dir``."""
    start = time.perf_counter()
    chunks = chunk_all(articles, cfg)
    cache = EmbeddingCache(cfg.embedding_cache_dir, embedding_model)
    vectors, newly_embedded = cache.embed([c.embedding_text for c in chunks], embeddings)
    store = FaissStore(
        chunks,
        vectors,
        {
            "fingerprint": fingerprint(articles, embedding_model, cfg),
            "embedding_model": embedding_model,
            "articles": len(articles),
        },
    )
    store.save(cfg.index_dir)
    log.info(
        "index_built",
        articles=len(articles),
        chunks=len(chunks),
        newly_embedded=newly_embedded,
        seconds=round(time.perf_counter() - start, 2),
    )
    return store


def load_or_build(
    articles: Sequence[Article], embeddings: Embeddings, embedding_model: str, cfg: RAGConfig
) -> FaissStore:
    """Reuse the saved index when it matches the current articles; otherwise rebuild."""
    manifest = FaissStore.read_manifest(cfg.index_dir)
    current = fingerprint(articles, embedding_model, cfg)
    if manifest and manifest.get("fingerprint") == current:
        store = FaissStore.load(cfg.index_dir)
        log.info("index_loaded", chunks=len(store.chunks), built_at=manifest.get("built_at"))
        return store
    log.info("index_stale_rebuilding", had_index=manifest is not None)
    return build_index(articles, embeddings, embedding_model, cfg)
