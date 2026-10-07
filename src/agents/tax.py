"""Tax Education agent: US tax concepts and tax-advantaged accounts (REQ-TX-01..03).

- Retrieves only from the ``tax`` category (REQ-TX-01).
- Each source is labeled with its tax year, and the prompt requires stating the year for any
  figure (REQ-TX-02).
- A referral to a qualified tax professional is appended *deterministically* — not left to
  the model's discretion (REQ-TX-03).
"""

from __future__ import annotations

from typing import ClassVar

from src.agents.grounded import GroundedAgent
from src.rag.retriever import RetrievedChunk

TAX_NOT_COVERED = (
    "I don't have a verified source on that tax question yet. My tax articles cover 401(k)s, "
    "traditional and Roth IRAs, HSAs, 529 plans, capital gains, qualified dividends, tax-loss "
    "harvesting, the wash-sale rule, required minimum distributions, and tax brackets. "
    "For your specific situation, consider a qualified tax professional."
)

TAX_REFERRAL = (
    "Tax rules depend on your full situation and change over time — for decisions about your "
    "own taxes, consider a qualified tax professional such as a CPA or enrolled agent."
)

TAX_RULES = """- Explain US federal tax rules only; say so if asked about state or non-US taxes.
- Whenever you state a dollar amount, limit, rate threshold, or age rule, name the tax year it
  applies to exactly as labeled in the source (e.g. "for tax year 2026"). If a source has no
  tax year, don't present its figures as current-year limits.
- Never claim to know the user's tax situation; explain how the rule works generally."""


class TaxAgent(GroundedAgent):
    """Answers tax questions from the tax articles, with tax years and a referral."""

    name: str = "tax"
    role_description: ClassVar[str] = "Tax Education agent"
    extra_rules: ClassVar[str] = TAX_RULES
    categories: ClassVar[tuple[str, ...] | None] = ("tax",)
    not_covered: ClassVar[str] = TAX_NOT_COVERED

    def _source_header(self, chunk: RetrievedChunk) -> str:
        year = f"tax year {chunk.tax_year}" if chunk.tax_year else "no specific tax year"
        return f"{chunk.title} — {chunk.section} ({year}; {chunk.url})"

    def postprocess(self, answer: str) -> str:
        """End with the professional referral, exactly once (REQ-TX-03)."""
        if "tax professional" in answer.lower():
            return answer
        return f"{answer}\n\n{TAX_REFERRAL}"
