"""Finance Q&A agent: answers general finance questions from the knowledge base.

Pipeline (design section 7): retrieve -> (nothing relevant? say so, no LLM call) -> explain
with numbered sources -> keep only the sources the answer actually cites.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.core.models import AgentResult, Citation
from src.rag.retriever import RetrievedChunk, Retriever
from src.workflow.state import AgentInput

NOT_COVERED = (
    "I don't have a verified source on that topic in my knowledge base yet, so I'd rather not "
    "guess. Try asking about investing basics — stocks, bonds, ETFs, index funds, "
    "diversification, compound interest, or emergency funds."
)

_LEVEL_STYLE = {
    "beginner": "The user is a beginner: use plain language and define every financial term "
    "the first time you use it. Prefer short paragraphs and a simple example.",
    "intermediate": "The user knows the basics: be concise and skip elementary definitions.",
    "advanced": "The user is experienced: be precise and compact; nuance is welcome.",
}

SYSTEM_PROMPT = """You are the Finance Q&A agent of an EDUCATIONAL personal-finance assistant.
Rules:
- Answer ONLY from the numbered sources below. If they don't contain the answer, say so.
- Cite sources inline with their numbers, like [1] or [2][3], after the sentences they support.
- Educate; never tell the user to buy, sell, or hold a specific security.
- Mention risks alongside potential returns; never imply guaranteed returns.
- Keep it under 200 words. Do not add a disclaimer; the system adds one.
{level_style}

Sources:
{sources}"""

_CITE = re.compile(r"\[(\d+)\]")


class QAAgent:
    """General finance education from the knowledge base (REQ-QA-01, REQ-QA-02)."""

    name = "qa"

    def __init__(
        self,
        model: Runnable[Any, Any],
        retriever: Retriever,
        *,
        top_k: int = 4,
        history_messages: int = 6,
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.history_messages = history_messages

    def run(self, inp: AgentInput) -> AgentResult:
        """Retrieve, answer with inline citations, and return only the cited sources."""
        chunks = self.retriever.retrieve(
            inp["question"],
            k=self.top_k,
            exclude_categories=("tax",),  # tax agent owns tax
        )
        if not chunks:
            return AgentResult(agent=self.name, answer=NOT_COVERED)  # REQ-QA-02: no LLM call
        reply = self.model.invoke(self._prompt(inp, chunks))
        answer = reply.text.strip()
        return AgentResult(
            agent=self.name,
            answer=answer,
            citations=self._cited(answer, chunks),
            data={
                "retrieval_scores": [c.score for c in chunks],
                "retrieval_degraded": any(c.degraded for c in chunks),  # REQ-LLM-05
            },
        )

    def _prompt(self, inp: AgentInput, chunks: list[RetrievedChunk]) -> list[AnyMessage]:
        sources = "\n\n".join(
            f"[{i}] {c.title} — {c.section} ({c.url})\n{c.text}"
            for i, c in enumerate(chunks, start=1)
        )
        system = SYSTEM_PROMPT.format(
            level_style=_LEVEL_STYLE[inp["profile"].knowledge_level], sources=sources
        )
        history = list(inp["messages"][-(self.history_messages + 1) : -1])
        return [SystemMessage(system), *history, HumanMessage(inp["question"])]

    @staticmethod
    def _cited(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        """Citations for the [n] markers (all sources if none), one per article."""
        numbers = {int(n) for n in _CITE.findall(answer) if 1 <= int(n) <= len(chunks)}
        used = [chunks[n - 1] for n in sorted(numbers)] or chunks
        seen: set[str] = set()
        citations = []
        for c in used:
            if c.article_id not in seen:  # several chunks of one article -> one citation
                seen.add(c.article_id)
                citations.append(c.citation())
        return citations
