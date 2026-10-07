"""Intent router (REQ-WF-01, REQ-WF-03).

Three layers of resilience, cheapest-to-fix first:
1. the router LLM (small model, structured output) — which itself falls back OpenAI -> Gemini
   on transient errors inside the gateway;
2. if that whole call still fails, a deterministic keyword router (REQ-WF-03);
3. configuration errors (bad key, unknown model) are *not* caught: they fail loud (REQ-LLM-07).

The LLM also rewrites follow-ups into a standalone question ("what about its fees?" ->
"What are the fees of index funds?") so retrieval works on multi-turn conversations.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.core.llm import CONFIG_ERRORS
from src.core.models import Intent
from src.utils.logging import get_logger
from src.workflow.state import GraphState

log = get_logger(__name__)


class RouteDecision(BaseModel):
    """Router output."""

    intents: list[Intent] = Field(
        min_length=1,
        description="One or more intents; several only if the user asks several things.",
    )
    standalone_question: str = Field(
        description="The user's latest message rewritten to be understandable without the history."
    )


ROUTER_PROMPT = """You route messages for a personal-finance EDUCATION assistant.
Choose every intent that applies to the user's latest message:
- qa: explaining finance/investing CONCEPTS — what an ETF is, how bonds work, compound interest,
  diversification, emergency funds, budgeting, debt vs. investing, inflation, how indicators like
  moving averages or the P/E ratio work, and whether an offer looks like an investment scam.
- portfolio: analyzing the user's OWN holdings ("analyze my portfolio", "I own 10 AAPL and 5 MSFT").
- market: CURRENT data — today's prices, quotes, how a specific ticker, index, or sector is
  performing or trending right now.
- goals: planning a goal — retirement, house, college, how much to save, risk profile.
- news: recent financial news or headlines about a company or the market.
- tax: US taxes and tax-advantaged accounts (401(k), IRA, Roth, HSA, 529, RMDs, capital gains,
  dividends taxation, wash sale, tax brackets).
- out_of_scope: not about personal finance, money, or investing at all (recipes, coding, sports...).
  Questions about investment scams or fraud are IN scope (qa).
Rules:
- Use several intents only when the message clearly asks about several different things.
- Tax questions go to "tax" alone — do not add "qa" for them. If a message involves a
  tax-advantaged account AND another need (e.g. retirement planning), use "tax" plus that intent.
- A concept question about an indicator ("how do moving averages work?") is "qa"; asking for a
  specific ticker's current values ("what is AAPL's 50-day average?") is "market".
Examples:
- "What is the Roth IRA income limit?" -> [tax]
- "Will putting more into an IRA help me retire by 60?" -> [tax, goals]
- "Is a promise of guaranteed 30% returns a scam?" -> [qa]
- "How is TSLA doing today and what's the latest news?" -> [market, news]
Also rewrite the latest message as a standalone question, resolving pronouns from the history."""


# Ordered (intent, pattern) rules for the keyword fallback. Word-boundary regexes, lowercase.
_KEYWORD_RULES: list[tuple[Intent, re.Pattern[str]]] = [
    (
        "tax",
        re.compile(
            r"\b(tax(es|ed)?|ira|roth|401\(?k\)?|403\(?b\)?|hsa|529|capital gains?|"
            r"wash[- ]sale|deduct\w*|irs)\b"
        ),
    ),
    ("news", re.compile(r"\b(news|headlines?|announce\w*|earnings report)\b")),
    (
        "portfolio",
        re.compile(r"\b(my (portfolio|holdings|investments|stocks)|i own|i hold|shares of)\b"),
    ),
    (
        "market",
        re.compile(
            r"\b(price|quote|trading at|stock market|market today|s&p|nasdaq|dow|"
            r"index (is|was) doing|ticker|how is \w+ doing)\b|\$[a-z]{1,5}\b"
        ),
    ),
    (
        "goals",
        re.compile(
            r"\b(retire\w*|goal|save for|saving for|down payment|college fund|"
            r"how much (should|do) i (save|invest)|risk (profile|tolerance))\b"
        ),
    ),
]


_TAX_PATTERN = _KEYWORD_RULES[0][1]


def apply_tax_guard(intents: list[Intent], *texts: str) -> tuple[list[Intent], bool]:
    """Deterministic safety net: precise tax terms always reach the Tax agent.

    The LLM is good at meaning but sensitive to phrasing ("How does the wash sale rule work?"
    was once routed to qa). Tax vocabulary is precise, so a regex is reliable here. If it
    matches and the LLM didn't pick "tax", replace "qa" with "tax" (the QA agent excludes tax
    articles) or add "tax" alongside the other intents.
    """
    if "tax" in intents or not any(_TAX_PATTERN.search(t.lower()) for t in texts):
        return intents, False
    if intents == ["out_of_scope"]:
        return ["tax"], True
    fixed: list[Intent] = ["tax" if i == "qa" else i for i in intents]
    if "tax" not in fixed:
        fixed.append("tax")
    return list(dict.fromkeys(fixed)), True


def keyword_route(text: str) -> RouteDecision:
    """Deterministic fallback router: every matching rule, else ``qa``."""
    lowered = text.lower()
    intents: list[Intent] = [
        intent for intent, pattern in _KEYWORD_RULES if pattern.search(lowered)
    ]
    return RouteDecision(intents=intents or ["qa"], standalone_question=text)


class Router:
    """LangGraph node: classify the latest message into intents."""

    def __init__(self, model: Runnable[Any, Any], *, history_messages: int = 6) -> None:
        self.model = model
        self.history_messages = history_messages

    def __call__(self, state: GraphState) -> dict[str, Any]:
        """Return the intents and the standalone question for the latest message."""
        messages: Sequence[AnyMessage] = state["messages"]
        latest = messages[-1].text
        start = time.perf_counter()
        try:
            decision = self.model.invoke(self._prompt(messages))
            method = "llm"
        except CONFIG_ERRORS:
            raise
        except Exception as exc:
            log.warning("router_keyword_fallback", error_type=type(exc).__name__)
            decision = keyword_route(latest)
            method = "keywords"
        intents = list(dict.fromkeys(decision.intents))  # dedupe, keep order
        if len(intents) > 1 and "out_of_scope" in intents:
            intents.remove("out_of_scope")  # a real question wins over "out of scope"
        question = decision.standalone_question or latest
        intents, overridden = apply_tax_guard(intents, latest, question)
        log.info(
            "router_decision",
            intents=intents,
            method=method,
            tax_guard=overridden,
            latency_ms=round((time.perf_counter() - start) * 1000),
        )
        return {"intents": intents, "question": question}

    def _prompt(self, messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        history = list(messages[-(self.history_messages + 1) : -1])
        transcript = "\n".join(f"{m.type}: {m.text}" for m in history) or "(no earlier messages)"
        return [
            SystemMessage(ROUTER_PROMPT),
            HumanMessage(
                f"Conversation so far:\n{transcript}\n\nLatest message:\n{messages[-1].text}"
            ),
        ]
