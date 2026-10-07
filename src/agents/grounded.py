"""Shared logic for agents that answer from the knowledge base (design section 7).

Pipeline: retrieve -> (nothing relevant? honest "not covered", no LLM call) -> explain with
numbered sources -> keep only cited sources -> suggest what to learn next.

Subclasses set the retrieval scope and add domain rules; :class:`QAAgent` and
:class:`TaxAgent` are each ~20 lines on top of this.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, ClassVar

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.core.models import AgentResult, Citation
from src.rag.retriever import RetrievedChunk, Retriever
from src.workflow.state import AgentInput

LEVEL_STYLE = {
    "beginner": "The user is a beginner: use plain language and define every financial term "
    "the first time you use it. Prefer short paragraphs and a simple example.",
    "intermediate": "The user knows the basics: be concise and skip elementary definitions.",
    "advanced": "The user is experienced: be precise and compact; nuance is welcome.",
}

# Which article difficulties suit each reader for "learn next" suggestions (REQ-QA-03).
_SUITABLE_DIFFICULTY = {
    "beginner": {"beginner", "intermediate"},
    "intermediate": {"beginner", "intermediate", "advanced"},
    "advanced": {"intermediate", "advanced"},
}

BASE_RULES = """Rules:
- Answer ONLY from the numbered sources below. If they don't contain the answer, say so.
- Cite sources inline with their numbers, like [1] or [2][3], after the sentences they support.
- Educate; never tell the user to buy, sell, or hold a specific security.
- Mention risks alongside potential returns; never imply guaranteed returns.
- Keep it under 200 words. Do not add a disclaimer; the system adds one."""

_CITE = re.compile(r"\[(\d+)\]")


class GroundedAgent:
    """Base class: retrieve, answer with inline citations, suggest related articles."""

    name: str = "grounded"
    role_description: ClassVar[str] = "agent"
    extra_rules: ClassVar[str] = ""
    categories: ClassVar[tuple[str, ...] | None] = None
    exclude_categories: ClassVar[tuple[str, ...] | None] = None
    not_covered: ClassVar[str] = "I don't have a verified source on that topic yet."

    def __init__(
        self,
        model: Runnable[Any, Any],
        retriever: Retriever,
        *,
        top_k: int = 4,
        history_messages: int = 6,
        learn_next: int = 3,
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.history_messages = history_messages
        self.learn_next = learn_next

    # -- pipeline ------------------------------------------------------------------------------

    def run(self, inp: AgentInput) -> AgentResult:
        """Retrieve, answer with inline citations, and return cited sources + suggestions."""
        # One retrieval call serves both the answer (top_k) and "learn next" (the rest).
        ranked = self.retriever.retrieve(
            inp["question"],
            k=self.top_k + 6,
            categories=self.categories,
            exclude_categories=self.exclude_categories,
        )
        chunks = ranked[: self.top_k]
        if not chunks:
            return AgentResult(agent=self.name, answer=self.not_covered)  # no LLM call
        reply = self.model.invoke(self._prompt(inp, chunks))
        answer = self.postprocess(reply.text.strip())
        citations = self._cited(answer, chunks)
        return AgentResult(
            agent=self.name,
            answer=answer,
            citations=citations,
            data={
                "retrieval_scores": [c.score for c in chunks],
                "retrieval_degraded": any(c.degraded for c in chunks),  # REQ-LLM-05
                "learn_next": self._learn_next(
                    ranked, {c.chunk_id.split("#")[0] for c in citations}, inp
                ),
            },
        )

    def postprocess(self, answer: str) -> str:
        """Apply deterministic additions to the answer (e.g. the tax referral)."""
        return answer

    # -- helpers ------------------------------------------------------------------------------

    def _source_header(self, chunk: RetrievedChunk) -> str:
        return f"{chunk.title} — {chunk.section} ({chunk.url})"

    def _prompt(self, inp: AgentInput, chunks: Sequence[RetrievedChunk]) -> list[AnyMessage]:
        sources = "\n\n".join(
            f"[{i}] {self._source_header(c)}\n{c.text}" for i, c in enumerate(chunks, start=1)
        )
        system = (
            f"You are the {self.role_description} of an EDUCATIONAL personal-finance assistant.\n"
            f"{BASE_RULES}\n{self.extra_rules}\n"
            f"{LEVEL_STYLE[inp['profile'].knowledge_level]}\n\nSources:\n{sources}"
        )
        history = list(inp["messages"][-(self.history_messages + 1) : -1])
        return [SystemMessage(system), *history, HumanMessage(inp["question"])]

    @staticmethod
    def _cited(answer: str, chunks: Sequence[RetrievedChunk]) -> list[Citation]:
        """Citations for the [n] markers (all sources if none), one per article."""
        return cited_chunks(answer, chunks, fallback_all=True)

    def _learn_next(
        self, ranked: Sequence[RetrievedChunk], cited_articles: set[str], inp: AgentInput
    ) -> list[dict[str, str]]:
        """Up to ``learn_next`` related articles the answer didn't cite, suited to the reader."""
        suitable = _SUITABLE_DIFFICULTY[inp["profile"].knowledge_level]
        picks: list[dict[str, str]] = []
        seen = set(cited_articles)
        for c in ranked:
            if c.article_id in seen or c.difficulty not in suitable:
                continue
            seen.add(c.article_id)
            picks.append({"title": c.title, "url": c.url, "category": c.category})
            if len(picks) == self.learn_next:
                break
        return picks


def cited_chunks(
    answer: str, chunks: Sequence[RetrievedChunk], *, fallback_all: bool = False, offset: int = 0
) -> list[Citation]:
    """One citation per article whose marker ``[offset + n]`` appears in ``answer``.

    ``fallback_all``: cite every chunk when the answer has no markers (the answer is built
    *only* from these sources, e.g. Q&A). Agents mixing data and concepts use False.
    """
    numbers = {int(n) - offset for n in _CITE.findall(answer)}
    used = [chunks[n - 1] for n in sorted(numbers) if 1 <= n <= len(chunks)]
    if not used and fallback_all:
        used = list(chunks)
    seen: set[str] = set()
    citations = []
    for c in used:
        if c.article_id not in seen:  # several chunks of one article -> one citation
            seen.add(c.article_id)
            citations.append(c.citation())
    return citations
