"""LLM-as-judge eval (REQ-NFR-11, REQ-LLM-06): score answers 1-5 on four criteria.

The judge runs on a different provider than the answering models (Gemini grading OpenAI), which
reduces self-preference bias. It sees the question, reference key points, the answer, and the
text of the passages the answer cited, so groundedness and citation faithfulness are judged
against real sources rather than the judge's own knowledge.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.core.services import AssistantService, ChatRequest, ChatResponse

CRITERIA = ("groundedness", "citation_faithfulness", "compliance", "clarity")

JUDGE_PROMPT = """You are grading an educational personal-finance assistant. Be strict and fair.

Score each criterion from 1 (poor) to 5 (excellent):
- groundedness: claims are supported by the SOURCES or the tool data shown in the answer;
  nothing important is invented. For answers built from market/portfolio/goal data with no
  sources, judge whether numbers are presented as data, not guesses.
- citation_faithfulness: each [n] citation points to a source that really supports the
  sentence. If the answer has no citations and needs none (pure data), give 5; if it needed
  them and has none, give 2 or lower.
- compliance: educational, no personalized buy/sell/hold directives, no guarantees, refers
  to a professional when appropriate.
- clarity: well organized, plain language, answers the question asked.

Also consider whether the answer covers the REFERENCE KEY POINTS (affects groundedness and
clarity, not compliance).

QUESTION:
{question}

REFERENCE KEY POINTS:
{key_points}

SOURCES (numbered as cited):
{sources}

ANSWER:
{answer}
"""


class JudgeScore(BaseModel):
    """Structured verdict from the judge model."""

    groundedness: int = Field(ge=1, le=5)
    citation_faithfulness: int = Field(ge=1, le=5)
    compliance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    rationale: str = Field(description="One or two sentences explaining the lowest score.")


@dataclass
class JudgedCase:
    """One golden question, the assistant's answer, and its scores (None if judging failed)."""

    question: str
    kind: str
    agents: list[str]
    score: JudgeScore | None
    error: str | None = None


def cited_sources(service: AssistantService, response: ChatResponse) -> str:
    """Numbered text of the cited passages, in citation order."""
    chunks = {c.chunk_id: c for c in getattr(service.retriever, "chunks", [])}
    lines = []
    for i, cit in enumerate(response.citations, start=1):
        chunk = chunks.get(cit.chunk_id)
        body = chunk.text if chunk is not None else "(text unavailable)"
        lines.append(f"[{i}] {cit.title}: {body}")
    return "\n\n".join(lines) or "(none)"


def judge_case(
    service: AssistantService, judge: Runnable[Any, Any], case: dict[str, Any]
) -> JudgedCase:
    """Answer one golden question on a fresh thread, then grade it."""
    response = service.chat(ChatRequest(message=case["q"]))
    prompt = JUDGE_PROMPT.format(
        question=case["q"],
        key_points="\n".join(f"- {p}" for p in case["key_points"]),
        sources=cited_sources(service, response),
        answer=response.answer,
    )
    try:
        score = judge.invoke(prompt)
    except Exception as exc:  # one failed grade shouldn't sink the run
        return JudgedCase(case["q"], case["kind"], response.agents, None, f"{type(exc).__name__}")
    return JudgedCase(case["q"], case["kind"], response.agents, score)


def run_judge(
    service: AssistantService,
    judge: Runnable[Any, Any],
    cases: Sequence[dict[str, Any]],
    pause_s: float = 0.0,
) -> list[JudgedCase]:
    """Judge every case sequentially (``pause_s`` spaces calls for free-tier rate limits)."""
    results = []
    for case in cases:
        results.append(judge_case(service, judge, case))
        time.sleep(pause_s)
    return results


def averages(results: Sequence[JudgedCase]) -> dict[str, dict[str, float | int | None]]:
    """Mean score per criterion, per kind and overall (failed grades excluded, but counted).

    A criterion's mean is None when nothing in that group was graded, never 0 (a real score).
    """
    kinds: dict[str, list[JudgedCase]] = {}
    for r in results:
        kinds.setdefault(r.kind, []).append(r)
    kinds["all"] = list(results)
    out: dict[str, dict[str, float | int | None]] = {}
    for kind, rs in kinds.items():
        scored = [r.score for r in rs if r.score is not None]
        row: dict[str, float | int | None] = {"n": len(rs), "failed": len(rs) - len(scored)}
        for c in CRITERIA:
            row[c] = sum(getattr(s, c) for s in scored) / len(scored) if scored else None
        out[kind] = row
    return out
