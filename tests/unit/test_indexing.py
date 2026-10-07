"""Chunking, embedding cache, FAISS store persistence, and index lifecycle."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from structlog.testing import capture_logs

from src.core.config import get_settings
from src.core.errors import KnowledgeBaseError
from src.rag.chunker import chunk_article
from src.rag.embedding_cache import EmbeddingCache
from src.rag.index import build_index, chunk_all, fingerprint, load_or_build
from src.rag.knowledge_base import Article, load_articles
from src.rag.vector_store import FaissStore
from tests.fakes import HashingEmbeddings
from tests.kb import rag_config


def _article(body: str, **kw: object) -> Article:
    base = {
        "id": "basics-x",
        "title": "X Topic",
        "category": "basics",
        "difficulty": "beginner",
        "source_name": "Test",
        "source_url": "https://example.com/x",
        "last_reviewed": date(2026, 10, 7),
    }
    return Article(**(base | kw), body=body)  # type: ignore[arg-type]


# --- chunker --------------------------------------------------------------------------------


def test_chunks_follow_sections_with_context_prefix() -> None:
    """REQ-RAG-05: one chunk per ## section; embedding text carries title + section."""
    body = "# X Topic\nIntro line.\n\n## First part\nAlpha text.\n\n## Second part\nBeta text."
    chunks = chunk_article(_article(body))
    assert [c.section for c in chunks] == ["Overview", "First part", "Second part"]
    assert [c.chunk_id for c in chunks] == ["basics-x#0", "basics-x#1", "basics-x#2"]
    assert chunks[1].text == "Alpha text."
    assert chunks[1].embedding_text == "X Topic — First part\nAlpha text."
    assert chunks[0].url == "https://example.com/x"


def test_long_sections_split_with_overlap() -> None:
    """REQ-RAG-02: sections over max_words become overlapping windows."""
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_article(_article("## Long\n" + " ".join(words)), max_words=40, overlap_words=10)
    assert len(chunks) == 3
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert len(first) == 40
    assert first[-10:] == second[:10]  # overlap preserved
    assert chunks[-1].text.split()[-1] == "w99"  # nothing lost at the end


def test_body_without_sections_is_one_chunk() -> None:
    """REQ-RAG-02: an article with no ## headings still yields a chunk."""
    chunks = chunk_article(_article("# X Topic\nJust a paragraph."))
    assert len(chunks) == 1 and chunks[0].section == "Overview"


def test_real_knowledge_base_chunks_are_reasonable() -> None:
    """REQ-RAG-01: the shipped KB chunks into many small, attributed pieces."""
    rag = get_settings().rag
    chunks = chunk_all(load_articles(rag.knowledge_base_dir), rag)
    assert len(chunks) > 200
    assert max(len(c.text.split()) for c in chunks) <= rag.chunk_max_words
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# --- embedding cache ------------------------------------------------------------------------


def test_cache_only_embeds_new_text(tmp_path: Path) -> None:
    """REQ-RAG-02: unchanged chunks are not re-embedded, even across processes."""
    emb = HashingEmbeddings()
    vectors, new = EmbeddingCache(tmp_path, "m").embed(["a b", "c d"], emb)
    assert vectors.shape == (2, emb.dim) and new == 2
    reloaded = EmbeddingCache(tmp_path, "m")  # simulates a restart
    _, new = reloaded.embed(["a b", "c d", "e f"], emb)
    assert new == 1
    assert emb.document_calls == 3


def test_cache_is_keyed_by_model(tmp_path: Path) -> None:
    """REQ-RAG-02: switching embedding model never reuses the other model's vectors."""
    emb = HashingEmbeddings()
    EmbeddingCache(tmp_path, "model-a").embed(["a b"], emb)
    _, new = EmbeddingCache(tmp_path, "model/b").embed(["a b"], emb)
    assert new == 1
    assert (tmp_path / "model_b.npz").exists()  # unsafe filename characters replaced


# --- vector store ---------------------------------------------------------------------------


def test_store_search_save_and_load(tmp_path: Path) -> None:
    """REQ-RAG-03: exact cosine search; save/load round-trips without pickle."""
    chunks = chunk_article(_article("## A\nalpha beta\n\n## B\ngamma delta"))
    emb = HashingEmbeddings()
    store = FaissStore(
        chunks, np.array(emb.embed_documents([c.embedding_text for c in chunks])), {}
    )
    best, score = store.search(np.array(emb.embed_query("gamma delta")), k=2)[0]
    assert chunks[best].section == "B" and score > 0.5
    store.save(tmp_path)
    loaded = FaissStore.load(tmp_path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in chunks]
    assert np.allclose(loaded.vectors, store.vectors)
    assert FaissStore.read_manifest(tmp_path)["chunks"] == 2  # type: ignore[index]


def test_store_rejects_mismatched_vectors() -> None:
    """REQ-RAG-03: chunk/vector count mismatch is caught."""
    chunks = chunk_article(_article("## A\nalpha"))
    with pytest.raises(KnowledgeBaseError, match="vectors"):
        FaissStore(chunks, np.zeros((3, 4), dtype=np.float32), {})


def test_corrupt_index_fails_clearly(tmp_path: Path) -> None:
    """REQ-RAG-03: a broken index file gives a clear KnowledgeBaseError."""
    (tmp_path / "index.faiss").write_text("not an index")
    (tmp_path / "manifest.json").write_text("{bad json")
    assert FaissStore.read_manifest(tmp_path) is None
    with pytest.raises(KnowledgeBaseError):
        FaissStore.load(tmp_path)


# --- index lifecycle ------------------------------------------------------------------------


def test_load_or_build_reuses_matching_index(tmp_path: Path) -> None:
    """REQ-RAG-02: same articles + model -> load from disk, no embedding calls."""
    rag = rag_config(tmp_path)
    articles = load_articles(rag.knowledge_base_dir)[:5]
    emb = HashingEmbeddings()
    build_index(articles, emb, "hashing", rag)
    calls_after_build = emb.document_calls
    with capture_logs() as logs:
        store = load_or_build(articles, emb, "hashing", rag)
    assert emb.document_calls == calls_after_build
    assert any(e["event"] == "index_loaded" for e in logs)
    assert store.manifest["fingerprint"] == fingerprint(articles, "hashing", rag)


def test_load_or_build_rebuilds_when_articles_change(tmp_path: Path) -> None:
    """REQ-RAG-02: an edited article changes the fingerprint -> rebuild, cache keeps the rest."""
    rag = rag_config(tmp_path)
    articles = load_articles(rag.knowledge_base_dir)[:5]
    emb = HashingEmbeddings()
    build_index(articles, emb, "hashing", rag)
    before = emb.document_calls
    edited = [
        articles[0].model_copy(update={"body": articles[0].body + "\n\n## New\nfresh"}),
        *articles[1:],
    ]
    with capture_logs() as logs:
        load_or_build(edited, emb, "hashing", rag)
    assert any(e["event"] == "index_stale_rebuilding" for e in logs)
    assert emb.document_calls - before == 1  # only the new chunk was embedded


def test_fingerprint_depends_on_model_and_chunking(tmp_path: Path) -> None:
    """REQ-RAG-02: changing the embedding model or chunk size invalidates the index."""
    rag = rag_config(tmp_path)
    articles = load_articles(rag.knowledge_base_dir)[:2]
    base = fingerprint(articles, "m1", rag)
    assert fingerprint(articles, "m2", rag) != base
    assert fingerprint(articles, "m1", rag.model_copy(update={"chunk_max_words": 100})) != base
