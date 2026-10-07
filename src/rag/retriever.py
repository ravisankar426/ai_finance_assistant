"""Retrieval interface and the hybrid FAISS + BM25 retriever (REQ-RAG-03..05, REQ-LLM-05).

Agents depend only on the :class:`Retriever` protocol (constitution P6).

How :class:`HybridRetriever` answers a query:
1. Category filter -> the set of allowed chunks (REQ-RAG-04).
2. **Vector search** (FAISS, exact cosine) for meaning: finds "ways to lower investing costs"
   -> the expense-ratio article even with no shared words.
3. **BM25** for exact terms: "401(k)", "wash sale", jargon that embeddings blur.
4. **Reciprocal Rank Fusion**: score = sum(1 / (rrf_k + rank)) over both lists. RRF uses
   ranks only, so it needs no calibration between cosine and BM25 score scales.
5. **Relevance gate** (REQ-QA-02): drop chunks whose cosine is below ``min_cosine``. If none
   survive, return [] and the agent says "not covered" instead of guessing.
6. **Degraded mode** (REQ-LLM-05): if the embedding call fails transiently, fall back to
   BM25 alone with a keyword-coverage gate, and flag results ``degraded=True``.
   Auth/permission/not-found errors are re-raised (REQ-LLM-07).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from src.core.llm import CONFIG_ERRORS
from src.core.models import Citation
from src.rag.chunker import Chunk
from src.rag.text import tokenize
from src.rag.vector_store import FaissStore
from src.utils.logging import get_logger

log = get_logger(__name__)

# HTTP statuses that mean "configuration problem" for SDK errors LangChain doesn't map.
_CONFIG_STATUS = {401, 403, 404}


@dataclass(frozen=True)
class RetrievedChunk:
    """A piece of knowledge-base text plus its source attribution (REQ-RAG-05)."""

    chunk_id: str
    article_id: str
    title: str
    category: str
    url: str
    section: str
    text: str
    score: float  # cosine similarity (normal mode) or keyword coverage (degraded mode)
    degraded: bool = False

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
        """Return up to ``k`` relevant chunks, best first ([] if nothing is relevant)."""
        ...


def is_config_error(exc: BaseException) -> bool:
    """Return True for auth/permission/not-found errors, mapped by LangChain or not."""
    return isinstance(exc, CONFIG_ERRORS) or getattr(exc, "status_code", None) in _CONFIG_STATUS


class HybridRetriever:
    """FAISS (meaning) + BM25 (exact terms), fused with Reciprocal Rank Fusion."""

    def __init__(
        self,
        store: FaissStore,
        embeddings: Embeddings,
        *,
        fetch_k: int = 50,
        rrf_k: int = 60,
        min_cosine: float = 0.3,
        min_keyword_coverage: float = 0.5,
    ) -> None:
        self.store = store
        self.chunks: list[Chunk] = store.chunks
        self.embeddings = embeddings
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k
        self.min_cosine = min_cosine
        self.min_keyword_coverage = min_keyword_coverage
        doc_tokens = [tokenize(c.embedding_text) for c in self.chunks]
        self._doc_terms = [set(toks) for toks in doc_tokens]
        self.bm25 = BM25Okapi(doc_tokens)

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        categories: Sequence[str] | None = None,
        exclude_categories: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``k`` relevant chunks, best first."""
        allowed = self._allowed(categories, exclude_categories)
        q_tokens = tokenize(query)
        if not allowed.any() or not q_tokens:
            return []
        bm25_rank = self._bm25_rank(q_tokens, allowed)
        try:
            query_vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        except Exception as exc:
            if is_config_error(exc):
                raise
            log.warning(
                "retrieval_degraded",
                reason="embeddings_unavailable",
                error_type=type(exc).__name__,
            )
            return self._degraded(q_tokens, bm25_rank, k)

        cosines = self.store.cosine(query_vector)
        vector_rank = [
            i for i, _ in self.store.search(query_vector, len(self.chunks)) if allowed[i]
        ]
        fused = self._rrf(vector_rank[: self.fetch_k], bm25_rank)
        relevant = [i for i in fused if cosines[i] >= self.min_cosine]
        return [self._result(i, float(cosines[i])) for i in relevant[:k]]

    def _allowed(
        self, categories: Sequence[str] | None, exclude: Sequence[str] | None
    ) -> np.ndarray:
        return np.array(
            [
                not (categories and c.category not in categories)
                and not (exclude and c.category in exclude)
                for c in self.chunks
            ],
            dtype=bool,
        )

    def _bm25_rank(self, q_tokens: list[str], allowed: np.ndarray) -> list[int]:
        scores = self.bm25.get_scores(q_tokens)
        order = np.argsort(-scores, kind="stable")
        return [int(i) for i in order if allowed[i] and scores[i] > 0][: self.fetch_k]

    def _rrf(self, *rankings: list[int]) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused, key=lambda i: (-fused[i], self.chunks[i].chunk_id))

    def _coverage(self, q_tokens: list[str], i: int) -> float:
        """IDF-weighted share of the query's terms that appear in chunk ``i``."""
        terms = set(q_tokens)
        max_idf = max(self.bm25.idf.values(), default=1.0)
        total = sum(self.bm25.idf.get(t, max_idf) for t in terms)
        hit = sum(self.bm25.idf[t] for t in terms & self._doc_terms[i])
        return float(hit / total) if total else 0.0

    def _degraded(self, q_tokens: list[str], bm25_rank: list[int], k: int) -> list[RetrievedChunk]:
        scored = [(i, self._coverage(q_tokens, i)) for i in bm25_rank]
        kept = [(i, s) for i, s in scored if s >= self.min_keyword_coverage]
        return [self._result(i, s, degraded=True) for i, s in kept[:k]]

    def _result(self, i: int, score: float, *, degraded: bool = False) -> RetrievedChunk:
        c = self.chunks[i]
        return RetrievedChunk(
            chunk_id=c.chunk_id,
            article_id=c.article_id,
            title=c.title,
            category=c.category,
            url=c.url,
            section=c.section,
            text=c.text,
            score=round(score, 4),
            degraded=degraded,
        )
