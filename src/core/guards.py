"""Compliance guardrails (REQ-GR-*).

Day 2 implements the deterministic parts: PII redaction of user input (REQ-GR-05), the
educational disclaimer (REQ-GR-03), and the out-of-scope reply (REQ-GR-01). The LLM-backed
checks (prompt injection, directive rewriting) arrive on Day 7.
"""

from __future__ import annotations

from src.utils.redaction import redact

DISCLAIMER = (
    "*This is educational information, not financial, investment, or tax advice. "
    "Consider consulting a licensed professional before making financial decisions.*"
)

OUT_OF_SCOPE_REPLY = (
    "I'm a personal-finance education assistant, so I can't help with that. "
    "I can explain investing concepts, analyze a portfolio you enter, look up market data, "
    "help you plan a financial goal, summarize financial news, or explain US tax-advantaged "
    "accounts. What would you like to learn about?"
)


def redact_user_text(text: str) -> str:
    """Mask PII before the text reaches an LLM, a log line, or the checkpoint store."""
    return redact(text)


def with_disclaimer(answer: str) -> str:
    """Append the educational disclaimer once."""
    if DISCLAIMER in answer:
        return answer
    return f"{answer.rstrip()}\n\n---\n{DISCLAIMER}"
