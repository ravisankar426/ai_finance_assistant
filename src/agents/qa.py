"""Finance Q&A agent: general finance education from the knowledge base (REQ-QA-01..03)."""

from __future__ import annotations

from typing import ClassVar

from src.agents.grounded import GroundedAgent

NOT_COVERED = (
    "I don't have a verified source on that topic in my knowledge base yet, so I'd rather not "
    "guess. Try asking about investing basics — stocks, bonds, ETFs, index funds, "
    "diversification, compound interest, or emergency funds."
)


class QAAgent(GroundedAgent):
    """Answers concept questions from every category except tax (the Tax agent owns tax)."""

    name: str = "qa"
    role_description: ClassVar[str] = "Finance Q&A agent"
    exclude_categories: ClassVar[tuple[str, ...] | None] = ("tax",)
    not_covered: ClassVar[str] = NOT_COVERED
