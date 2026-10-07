"""Ticker extraction shared by the Market and News agents.

"How's Apple doing?" -> AAPL needs world knowledge, so a small LLM call (router tier) does it.
If that call fails, a regex fallback catches explicit symbols ("$NVDA", "MSFT") — the same
LLM-plus-deterministic-fallback pattern as the router. Config errors still fail loud.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.core.llm import CONFIG_ERRORS
from src.data.providers.base import looks_like_ticker
from src.utils.logging import get_logger

log = get_logger(__name__)


class TickerRequest(BaseModel):
    """What market data the user is asking about."""

    tickers: list[str] = Field(
        default_factory=list,
        description="US ticker symbols mentioned or clearly implied, max 5 (Apple -> AAPL).",
    )
    wants_market_overview: bool = Field(
        default=False,
        description="True for market-wide questions (the market, indices, or sectors).",
    )


EXTRACT_PROMPT = """Extract which stocks/ETFs the user asks about.
- Return US ticker symbols in upper case. Map well-known company or fund names to their primary
  US ticker (Apple -> AAPL, Microsoft -> MSFT, "S&P 500 ETF" -> SPY).
- Do not guess tickers for vague descriptions ("that EV company") — leave the list empty.
- Set wants_market_overview ONLY for questions explicitly about the overall market, an index,
  or sectors ("how is the market doing?", "how are sectors performing?").
- A question about "the stock" or "the price" without naming one is ambiguous: return no
  tickers and wants_market_overview=false, so the assistant can ask which one."""

# Upper-case words that look like tickers but usually aren't, in finance chat.
_NOT_TICKER_TEXT = """
A I AM AN AND ARE AS AT BE BY DO ETF ETFS FOR HOW IF IN IRA IS IT ME MY OF ON OR SO THE TO US USA
USD WHAT WHY CEO CFO IPO SEC FED GDP CPI AI EV NYSE YTD
"""
_NOT_TICKERS = frozenset(_NOT_TICKER_TEXT.split())
_EXPLICIT = re.compile(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?)\b")
_CAPS = re.compile(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b")
_OVERVIEW = re.compile(r"\b(market|markets|stocks today|sectors?|indices|indexes)\b", re.I)


def regex_tickers(text: str) -> TickerRequest:
    """Deterministic fallback: '$sym' always counts; bare CAPS words unless common words."""
    found = [m.upper() for m in _EXPLICIT.findall(text)]
    found += [m for m in _CAPS.findall(text) if m not in _NOT_TICKERS]
    tickers = [t for t in dict.fromkeys(found) if looks_like_ticker(t)][:5]
    return TickerRequest(
        tickers=tickers, wants_market_overview=not tickers and bool(_OVERVIEW.search(text))
    )


class TickerExtractor:
    """LLM extraction with a regex fallback."""

    def __init__(self, model: Runnable[Any, Any]) -> None:
        self.model = model

    def extract(self, question: str) -> TickerRequest:
        """Return the tickers (validated shape, de-duplicated) and overview intent."""
        try:
            req: TickerRequest = self.model.invoke(
                [SystemMessage(EXTRACT_PROMPT), HumanMessage(question)]
            )
        except CONFIG_ERRORS:
            raise
        except Exception as exc:
            log.warning("ticker_extraction_fallback", error_type=type(exc).__name__)
            req = regex_tickers(question)
        tickers = [t.strip().upper().lstrip("$") for t in req.tickers]
        clean = [t for t in dict.fromkeys(tickers) if looks_like_ticker(t)][:5]
        return TickerRequest(tickers=clean, wants_market_overview=req.wants_market_overview)
