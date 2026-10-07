"""Keep the evaluation sets consistent with the knowledge base (REQ-RAG-06)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.core.config import get_settings
from src.rag.knowledge_base import load_articles

RETRIEVAL_SET = Path(__file__).parents[1] / "evals" / "retrieval_set.yaml"


def _cases() -> list[dict[str, Any]]:
    return yaml.safe_load(RETRIEVAL_SET.read_text())  # type: ignore[no-any-return]


def test_retrieval_set_references_real_articles() -> None:
    """REQ-RAG-06: every expected article id exists (no silent typos in the eval set)."""
    ids = {a.id for a in load_articles(get_settings().rag.knowledge_base_dir)}
    missing = {e for c in _cases() for e in c["expected"] if e not in ids}
    assert missing == set()


def test_retrieval_set_shape() -> None:
    """REQ-RAG-06: ~40 cases covering keyword, paraphrase, jargon, and off-topic queries."""
    cases = _cases()
    kinds = {c["kind"] for c in cases}
    assert len(cases) >= 40
    assert kinds == {"keyword", "paraphrase", "jargon", "off_topic"}
    assert all(c["expected"] == [] for c in cases if c["kind"] == "off_topic")


@pytest.mark.live
def test_live_retrieval_recall_meets_target() -> None:
    """REQ-RAG-06: recall@5 >= 0.85 and all off-topic queries rejected (real embeddings)."""
    from scripts.eval_retrieval import evaluate
    from src.core.llm import get_embeddings
    from src.rag.index import load_or_build
    from src.rag.retriever import HybridRetriever

    settings = get_settings()
    rag = settings.rag
    embeddings = get_embeddings(settings)
    store = load_or_build(
        load_articles(rag.knowledge_base_dir), embeddings, settings.llm.embeddings.model, rag
    )
    metrics = evaluate(HybridRetriever(store, embeddings, min_cosine=rag.min_cosine), _cases())
    assert metrics["recall@5"] >= 0.85, metrics
    assert metrics["off_topic_rejected"] == 1.0, metrics
