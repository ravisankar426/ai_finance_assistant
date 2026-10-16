"""Domain models shared across agents, workflow, API, and UI (design section 4)."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

AgentName = Literal["qa", "portfolio", "market", "goals", "news", "tax"]
Intent = Literal["qa", "portfolio", "market", "goals", "news", "tax", "out_of_scope"]

AGENT_NAMES: tuple[AgentName, ...] = ("qa", "portfolio", "market", "goals", "news", "tax")
STATE_UPDATE_KEYS = frozenset({"portfolio", "profile"})

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
    # Remembered across turns, e.g. {"portfolio": [...]} or {"profile": ...}. Only keys in
    # STATE_UPDATE_KEYS are applied, so an agent can't overwrite arbitrary graph state.
    state_updates: dict[str, Any] = Field(default_factory=dict)


class Holding(BaseModel):
    """One position the user holds (REQ-PF-01)."""

    ticker: str
    quantity: float = Field(gt=0)
    cost_basis: float | None = Field(default=None, gt=0, description="Price paid per share")
    purchase_date: date | None = Field(
        default=None,
        description="Day bought (YYYY-MM-DD). If cost_basis is blank, that day's closing price "
        "is used as the cost basis.",
    )

    @field_validator("purchase_date")
    @classmethod
    def _not_in_future(cls, value: date | None) -> date | None:
        if value is not None and value > date.today():
            raise ValueError("purchase date can't be in the future")
        return value


class UserProfile(BaseModel):
    """Per-session user profile passed to every agent (REQ-WF-08)."""

    knowledge_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] | None = None
    goals: list[str] = Field(default_factory=list)
    horizon_years: int | None = None
