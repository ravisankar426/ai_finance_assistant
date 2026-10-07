"""Domain models shared across agents, workflow, API, and UI (design section 4)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AgentName = Literal["qa", "portfolio", "market", "goals", "news", "tax"]
Intent = Literal["qa", "portfolio", "market", "goals", "news", "tax", "out_of_scope"]

AGENT_NAMES: tuple[AgentName, ...] = ("qa", "portfolio", "market", "goals", "news", "tax")

AGENT_LABELS: dict[str, str] = {
    "qa": "Finance Q&A",
    "portfolio": "Portfolio Analysis",
    "market": "Market Analysis",
    "goals": "Goal Planning",
    "news": "News Synthesizer",
    "tax": "Tax Education",
    "out_of_scope": "Scope check",
}


class Citation(BaseModel):
    """A knowledge-base source used in an answer (REQ-RAG-05)."""

    title: str
    url: str
    category: str
    chunk_id: str


class AgentResult(BaseModel):
    """The single contract every agent returns (design section 7a)."""

    agent: str
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)  # structured payload for UI charts
    error: str | None = None  # set when the agent degraded (REQ-WF-05)
    follow_up_question: str | None = None  # set when inputs are missing (REQ-WF-10)


class UserProfile(BaseModel):
    """Per-session user profile passed to every agent (REQ-WF-08)."""

    knowledge_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] | None = None
    goals: list[str] = Field(default_factory=list)
    horizon_years: int | None = None
