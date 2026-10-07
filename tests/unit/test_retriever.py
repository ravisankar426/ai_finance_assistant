from __future__ import annotations

import pytest

from src.core.config import get_settings
from src.rag.knowledge_base import load_articles
from src.rag.retriever import KeywordRetriever, tokenize


@pytest.fixture(scope="module")
def retriever() -> KeywordRetriever:
    return KeywordRetriever(load_articles(get_settings().rag.knowledge_base_dir), min_score=0.2)


def test_tokenize() -> None:
    """REQ-QA-01: stopwords dropped, plurals folded, 401(k) kept as one token."""
    assert tokenize("What are Index Funds and the 401(k)?") == ["index", "fund", "401(k)"]
    assert tokenize("class assess") == ["class", "assess"]  # 'ss' endings untouched


@pytest.mark.parametrize(
    ("query", "expected_title"),
    [
        ("What is an index fund?", "Index Funds"),
        ("How do bonds react when interest rates rise?", "Bonds"),
        ("How does compound interest work?", "Compound Interest"),
        ("How much should I keep in an emergency fund?", "Emergency Fund"),
        ("Explain dollar cost averaging", "Dollar-Cost Averaging"),
        ("What is an ETF?", "Exchange-Traded Funds (ETFs)"),
    ],
)
def test_topic_article_ranks_first(
    retriever: KeywordRetriever, query: str, expected_title: str
) -> None:
    """REQ-QA-01: the article *about* the topic outranks ones that merely mention it."""
    results = retriever.retrieve(query, k=3)
    assert results[0].title == expected_title


def test_off_topic_returns_nothing(retriever: KeywordRetriever) -> None:
    """REQ-QA-02: nothing above the threshold -> empty, so the agent says 'not covered'."""
    assert retriever.retrieve("How do I bake sourdough bread?", k=4) == []
    assert retriever.retrieve("the and of", k=4) == []  # only stopwords


def test_category_filters(retriever: KeywordRetriever) -> None:
    """REQ-RAG-04: include/exclude category filters are honored."""
    only_basics = retriever.retrieve("index fund diversification bonds", k=9, categories=["basics"])
    assert only_basics and all(c.category == "basics" for c in only_basics)
    no_investing = retriever.retrieve("index fund", k=9, exclude_categories=["investing"])
    assert all(c.category != "investing" for c in no_investing)


def test_chunks_carry_attribution(retriever: KeywordRetriever) -> None:
    """REQ-RAG-05: every chunk carries title, URL, category; scores are in [0, 1]."""
    chunk = retriever.retrieve("What is an index fund?", k=1)[0]
    citation = chunk.citation()
    assert citation.url.startswith("https://www.investor.gov/")
    assert citation.category == "investing"
    assert citation.chunk_id == "investing-index-funds#0"
    assert 0 < chunk.score <= 1
