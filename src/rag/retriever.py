"""Retrieval interface and a dependency-free keyword retriever.

Agents depend only on the :class:`Retriever` protocol (constitution P6). Day 2 ships
:class:`KeywordRetriever` (IDF-weighted term overlap over whole articles) so the end-to-end
slice works with no index to build; Day 3 swaps in hybrid FAISS + BM25 behind the same
protocol without touching any agent.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from src.core.models import Citation
from src.rag.knowledge_base import Article

_STOPWORD_TEXT = """
a an and are as at be by can do does for from how i if in into is it its me my of on or
should so than that the their them then there these they this to was what when where
which who why will with would you your about tell explain mean means
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_TOKEN = re.compile(r"[a-z0-9]+(?:\(k\))?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords; naive plural stripping ('funds' -> 'fund')."""
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


@dataclass(frozen=True)
class RetrievedChunk:
    """A piece of knowledge-base text plus its source attribution (REQ-RAG-05)."""

    chunk_id: str
    article_id: str
    title: str
    category: str
    url: str
    text: str
    score: float  # 0..1 relevance

    def citation(self) -> Citation:
        """Return the source attribution for this chunk."""
        return Citation(
            title=self.title, url=self.url, category=self.category, chunk_id=self.chunk_id
        )


class Retriever(Protocol):
    """Anything that can find relevant knowledge-base text for a query."""

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        categories: Sequence[str] | None = None,
        exclude_categories: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks above the relevance threshold, best first."""
        ...


class KeywordRetriever:
    """IDF-weighted keyword overlap over whole articles.

    Score ("coverage") = (sum of IDF of query terms found in the article) / (sum of IDF of all
    query terms): in [0, 1], "how much of the question's meaningful vocabulary the article
    covers". The threshold uses coverage; *ranking* adds a bonus for terms in the title so the
    article *about* a topic beats one that merely mentions it. Replaced on Day 3.
    """

    def __init__(self, articles: Sequence[Article], *, min_score: float = 0.2) -> None:
        self.articles = list(articles)
        self.min_score = min_score
        self._terms: list[set[str]] = []
        self._title_terms: list[set[str]] = []
        doc_freq: dict[str, int] = {}
        for article in self.articles:
            terms = set(tokenize(f"{article.title} {article.body}"))
            self._terms.append(terms)
            self._title_terms.append(set(tokenize(article.title)))
            for term in terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        n = max(len(self.articles), 1)
        self._idf = {t: math.log(1 + n / df) for t, df in doc_freq.items()}
        self._unknown_idf = math.log(1 + n)  # a term no article contains is maximally specific

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        categories: Sequence[str] | None = None,
        exclude_categories: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``k`` articles whose score clears ``min_score``."""
        q_terms = set(tokenize(query))
        if not q_terms:
            return []
        total = sum(self._idf.get(t, self._unknown_idf) for t in q_terms)
        scored: list[tuple[float, float, int]] = []  # (rank key, coverage, index)
        for i, article in enumerate(self.articles):
            if categories and article.category not in categories:
                continue
            if exclude_categories and article.category in exclude_categories:
                continue
            matched = q_terms & self._terms[i]
            coverage = sum(self._idf[t] for t in matched) / total
            if coverage < self.min_score:
                continue
            title_bonus = sum(self._idf[t] for t in matched & self._title_terms[i]) / total
            scored.append((coverage + title_bonus, coverage, i))
        scored.sort(key=lambda s: (-s[0], self.articles[s[2]].id))
        return [
            RetrievedChunk(
                chunk_id=f"{a.id}#0",
                article_id=a.id,
                title=a.title,
                category=a.category,
                url=a.source_url,
                text=a.body,
                score=round(coverage, 3),
            )
            for _, coverage, i in scored[:k]
            for a in [self.articles[i]]
        ]
