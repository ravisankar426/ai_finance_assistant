"""Hybrid retriever behavior over the real knowledge base, with offline embeddings."""

from __future__ import annotations

import pytest
from langchain_core.exceptions import ModelAuthenticationError
from structlog.testing import capture_logs

from src.rag.retriever import HybridRetriever, is_config_error
from src.rag.text import tokenize
from tests.fakes import HashingEmbeddings, StatusError
from tests.kb import offline_retriever, offline_store


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    return offline_retriever()


def test_tokenize() -> None:
    """REQ-RAG-03: stopwords dropped, plurals folded, 401(k) kept as one token."""
    assert tokenize("What are Index Funds and the 401(k)?") == ["index", "fund", "401(k)"]
    assert tokenize("class assess") == ["class", "assess"]


@pytest.mark.parametrize(
    ("query", "expected_article"),
    [
        ("What is an index fund?", "investing-index-funds"),
        ("wash sale 30 days", "tax-wash-sale-rule"),
        ("What is the 401(k) contribution limit?", "tax-401k-plans"),
        ("How does compound interest work?", "basics-compound-interest"),
        ("What is a REIT?", "investing-reits"),
    ],
)
def test_relevant_article_ranks_first(
    retriever: HybridRetriever, query: str, expected_article: str
) -> None:
    """REQ-RAG-03: hybrid retrieval puts the matching article first."""
    assert retriever.retrieve(query, k=4)[0].article_id == expected_article


def test_off_topic_returns_nothing(retriever: HybridRetriever) -> None:
    """REQ-QA-02: below the relevance gate -> [], so the agent says 'not covered'."""
    assert retriever.retrieve("How do I bake sourdough bread?", k=4) == []
    assert retriever.retrieve("the and of", k=4) == []  # only stopwords


def test_category_filters(retriever: HybridRetriever) -> None:
    """REQ-RAG-04: include/exclude category filters are honored."""
    tax_only = retriever.retrieve("Roth IRA income limit", k=10, categories=["tax"])
    assert tax_only and all(c.category == "tax" for c in tax_only)
    no_tax = retriever.retrieve("Roth IRA income limit", k=10, exclude_categories=["tax"])
    assert all(c.category != "tax" for c in no_tax)
    assert retriever.retrieve("index fund", k=4, categories=["nonexistent"]) == []


def test_chunks_carry_attribution(retriever: HybridRetriever) -> None:
    """REQ-RAG-05: every chunk carries title, URL, category, section; cosine score in (0, 1]."""
    chunk = retriever.retrieve("What is an index fund?", k=1)[0]
    citation = chunk.citation()
    assert citation.url.startswith("https://www.investor.gov/")
    assert citation.category == "investing"
    assert citation.chunk_id.startswith("investing-index-funds#")
    assert chunk.section
    assert 0 < chunk.score <= 1
    assert chunk.degraded is False


def test_results_respect_k_and_gate(retriever: HybridRetriever) -> None:
    """REQ-QA-02: never more than k results, all above the cosine gate."""
    results = retriever.retrieve("stocks bonds diversification risk", k=3)
    assert 0 < len(results) <= 3
    assert all(r.score >= retriever.min_cosine for r in results)


def test_rrf_rewards_agreement() -> None:
    """REQ-RAG-03: an item ranked well by both methods beats one ranked first by only one."""
    r = offline_retriever()
    fused = r._rrf([1, 2, 3], [2, 4])  # 2 is high in both lists
    assert fused[0] == 2
    assert set(fused) == {1, 2, 3, 4}


def test_degraded_mode_when_embeddings_down() -> None:
    """REQ-LLM-05: a transient embeddings failure -> BM25-only results flagged degraded."""
    r = HybridRetriever(offline_store(), HashingEmbeddings(fail_with=ConnectionError("down")))
    with capture_logs() as logs:
        results = r.retrieve("wash sale 30 days", k=3)
    assert results and all(c.degraded for c in results)
    assert results[0].article_id == "tax-wash-sale-rule"
    assert any(e["event"] == "retrieval_degraded" for e in logs)
    assert r.retrieve("sourdough bread recipe", k=3) == []  # coverage gate still applies


@pytest.mark.parametrize("error", [ModelAuthenticationError("bad key"), StatusError(401)])
def test_config_errors_are_not_degraded(error: Exception) -> None:
    """REQ-LLM-07: a bad key on embeddings fails loud instead of silently degrading."""
    r = HybridRetriever(offline_store(), HashingEmbeddings(fail_with=error))
    with pytest.raises(type(error)):
        r.retrieve("What is an index fund?", k=3)


def test_is_config_error() -> None:
    """REQ-LLM-07: mapped classes and raw SDK status codes are both recognized."""
    assert is_config_error(ModelAuthenticationError("x"))
    assert is_config_error(StatusError(404))
    assert not is_config_error(StatusError(503))
    assert not is_config_error(ConnectionError())
