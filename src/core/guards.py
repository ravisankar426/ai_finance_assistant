"""Compliance guardrails (REQ-GR-01..07).

Detection is deterministic (regexes you can read and test); an LLM is used only to *rewrite*
flagged text, and its output is re-checked. If the rewrite fails or still violates a rule,
the offending sentences are removed. So the guarantee never depends on the model behaving.

- GR-01 out of scope: the router's ``out_of_scope`` intent -> :data:`OUT_OF_SCOPE_REPLY`.
- GR-02 prompt injection: instruction-override sentences are removed from the user message
  before routing; the legitimate remainder is answered (or a refusal if nothing remains).
- GR-03 disclaimer on every finance answer.
- GR-04 personalized buy/sell/hold directives and GR-07 guarantee language in a draft answer
  -> rewritten into educational framing.
- GR-05 PII redaction (``src/utils/redaction.py``).
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.utils.logging import get_logger
from src.utils.redaction import redact

log = get_logger(__name__)

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

INJECTION_REPLY = (
    "I can't change how I work or reveal my instructions, but I'm happy to help with "
    "personal-finance questions — investing concepts, your portfolio, market data, goals, "
    "news, or US tax-advantaged accounts."
)

# --- GR-05 / GR-03 ------------------------------------------------------------------------------


def redact_user_text(text: str) -> str:
    """Mask PII before the text reaches an LLM, a log line, or the checkpoint store."""
    return redact(text)


def with_disclaimer(answer: str) -> str:
    """Append the educational disclaimer once."""
    if DISCLAIMER in answer:
        return answer
    return f"{answer.rstrip()}\n\n---\n{DISCLAIMER}"


# --- sentence helpers ---------------------------------------------------------------------------

_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.M)


def sentences(text: str) -> list[str]:
    """Split text into sentences/lines (good enough for chat messages and answers)."""
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


# --- GR-02 prompt injection ---------------------------------------------------------------------

_INJECTION = re.compile(
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?|"
    r"guardrails?|policies|policy|everything above)\b"
    r"|\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system|hidden|initial)\s+(prompt|instructions?|message)\b"
    r"|\byou are now\b|\bact as (an? )?(unrestricted|unfiltered|different)\b|\bjailbreak\b"
    r"|\bdeveloper mode\b|\bdo anything now\b|\bno (rules|restrictions|limits)\b",
    re.I,
)


def remove_injection(text: str) -> tuple[str, list[str]]:
    """Drop instruction-override sentences; return (remaining text, removed sentences)."""
    kept: list[str] = []
    removed: list[str] = []
    for s in sentences(text):
        (removed if _INJECTION.search(s) else kept).append(s)
    return " ".join(kept), removed


# --- GR-04 / GR-07 output compliance ------------------------------------------------------------

_DIRECTIVE = re.compile(
    r"\byou (should|must|need to|ought to)\s+(definitely\s+|immediately\s+|probably\s+)?"
    r"(buy|sell|short|hold|dump|load up on|invest in|get out of)\b"
    r"|\b(i|we) (recommend|suggest|advise)\s+(that you\s+)?"
    r"(buying|selling|shorting|you buy|you sell)\b"
    r"|\b(buy|sell|short) (it|this stock|these shares|now|today)\b"
    r"|\b(strong buy|strong sell|is a (great|good|must) buy)\b",
    re.I,
)
_GUARANTEE = re.compile(
    r"\bguarantee[sd]?\b(?!.{0,40}\b(no|not|never|isn't|aren't|cannot|can't)\b)"
    r"|\b(risk[- ]free(?! (rate|return|asset|treasur))|can.t lose|cannot lose|sure thing|"
    r"will definitely (rise|go up|grow))\b",
    re.I,
)
_NEGATED_GUARANTEE = re.compile(
    r"\b(no|not|never|nothing|isn't|aren't|cannot|can't)\b.{0,40}\bguarantee", re.I
)
# Warning ABOUT guarantees ("promises of guaranteed returns are a red flag") is scam education,
# not a promise; deleting it would remove the most important sentence of the answer.
_WARNING_CONTEXT = re.compile(
    r"\b(scam|fraud|red flags?|warning signs?|too good to be true|ponzi|beware|be wary|"
    r"suspicious|skeptical)\b",
    re.I,
)


def compliance_issues(text: str) -> list[str]:
    """Sentences that give personalized directives (GR-04) or promise returns (GR-07)."""
    issues = []
    for s in sentences(text):
        promise = (
            _GUARANTEE.search(s)
            and not _NEGATED_GUARANTEE.search(s)
            and not _WARNING_CONTEXT.search(s)
        )
        if _DIRECTIVE.search(s) or promise:
            issues.append(s)
    return issues


def strip_sentences(text: str, bad: list[str]) -> str:
    """Remove exact sentences from text (deterministic last resort)."""
    for s in bad:
        text = text.replace(s, "")
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


REWRITE_PROMPT = """You edit answers from an EDUCATIONAL personal-finance assistant for compliance.
Rewrite ONLY the flagged sentences so they:
- do not tell the user to buy, sell, or hold any specific security (explain considerations instead);
- do not promise or imply guaranteed returns (state that returns are uncertain).
Keep every other sentence, all numbers, citations like [1], and formatting unchanged.
Return the full revised answer and nothing else.

Flagged sentences:
{flagged}"""


class ComplianceRewriter:
    """Rewrite flagged sentences with the guard model; re-check; strip if still flagged."""

    def __init__(self, model: Runnable[Any, Any] | None) -> None:
        self.model = model

    def enforce(self, answer: str) -> tuple[str, list[str]]:
        """Return (compliant answer, sentences that were flagged)."""
        flagged = compliance_issues(answer)
        if not flagged:
            return answer, []
        revised = answer
        if self.model is not None:
            try:
                reply = self.model.invoke(
                    [
                        SystemMessage(
                            REWRITE_PROMPT.format(flagged="\n".join(f"- {s}" for s in flagged))
                        ),
                        HumanMessage(answer),
                    ]
                )
                revised = reply.text.strip() or answer
            except Exception as exc:  # the rewrite is best-effort; stripping is the guarantee
                log.warning("compliance_rewrite_failed", error_type=type(exc).__name__)
        still = compliance_issues(revised)
        if still:
            revised = strip_sentences(revised, still)
        log.info("compliance_enforced", flagged=len(flagged), stripped=len(still))
        return revised, flagged
