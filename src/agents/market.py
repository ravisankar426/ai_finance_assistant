"""Market Analysis agent: live quotes, trends, and the market overview (REQ-MK-01..04).

Pipeline (design section 7): check inputs -> compute (market service + analytics) ->
retrieve concept articles (REQ-WF-09) -> explain. The LLM receives a fact block of numbers
computed in code and is told not to compute or predict anything.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.agents.entities import TickerExtractor
from src.agents.grounded import LEVEL_STYLE, cited_chunks
from src.core.errors import AppError
from src.core.market_analytics import describe_snapshot, trend_snapshot
from src.core.models import AgentResult
from src.data.market_service import MarketDataService
from src.data.overview import describe_overview, market_overview
from src.rag.retriever import Retriever
from src.workflow.state import AgentInput

FOLLOW_UP = (
    "Which stock, ETF, or index would you like me to look at? For example “AAPL”, “SPY”, or "
    "“how is the overall market doing today?”"
)

SYSTEM = """You are the Market Analysis agent of an EDUCATIONAL personal-finance assistant.
Rules:
- Explain ONLY the facts in the MARKET DATA block. Never compute new numbers, never invent data.
- Explain what the indicators mean in plain terms (price vs. moving averages, returns,
  52-week range, volatility). Use the numbered concept sources and cite them like [1].
- NEVER predict future prices, and never say to buy, sell, or hold. Past trends don't
  predict future returns — say so when you describe a trend.
- A single day's change is NOT a trend. "Trend" means the moving-average classification and
  multi-month returns in the data; for the overall market use the S&P 500 (SPY) trend facts.
- If data is marked STALE, UNADJUSTED, or unavailable, say so plainly.
- Keep it under 220 words. Do not add a disclaimer; the system adds one.
{level_style}

MARKET DATA (computed by the system):
{facts}

Concept sources:
{sources}"""


class MarketAgent:
    """Quotes + trend indicators for up to ``max_tickers`` symbols, and/or the overview."""

    name = "market"

    def __init__(
        self,
        model: Runnable[Any, Any],
        extractor: TickerExtractor,
        market: MarketDataService,
        retriever: Retriever,
        *,
        max_tickers: int = 3,
    ) -> None:
        self.model = model
        self.extractor = extractor
        self.market = market
        self.retriever = retriever
        self.max_tickers = max_tickers

    def run(self, inp: AgentInput) -> AgentResult:
        """Fetch, compute, ground, and explain."""
        req = self.extractor.extract(inp["question"])
        if not req.tickers and not req.wants_market_overview:
            return AgentResult(agent=self.name, follow_up_question=FOLLOW_UP)  # REQ-WF-10

        facts: list[str] = []
        notes: list[str] = []
        data: dict[str, Any] = {"tickers": [], "snapshots": []}
        sources: set[str] = set()
        stale = False
        tickers = list(req.tickers[: self.max_tickers])
        if req.wants_market_overview and "SPY" not in tickers:
            tickers.append("SPY")  # the market's trend, not just today's move
        for ticker in tickers:
            try:
                quote = self.market.get_quote(ticker)
                snap = trend_snapshot(self.market.get_history(ticker))
            except AppError as exc:
                notes.append(exc.user_message)
                continue
            name = self._name(ticker)
            facts.append(
                f"{describe_snapshot(snap, name)}\n  today: ${quote.price:,.2f} "
                f"({quote.change_pct:+.2f}% vs previous close) as of {quote.market_date}"
            )
            data["tickers"].append(ticker)
            data["snapshots"].append(snap.model_dump(mode="json"))
            sources.update({quote.source, snap.source})
            stale = stale or quote.stale or snap.stale
        if req.wants_market_overview:
            overview = market_overview(self.market)
            if overview.indices or overview.sectors:
                facts.append(describe_overview(overview))
                data["overview"] = overview.model_dump(mode="json")
                sources.update(r.source for r in (*overview.indices, *overview.sectors))
                stale = stale or overview.any_stale
            else:
                notes.append("The market overview is unavailable right now.")

        if not facts:  # nothing to explain: report what went wrong, no LLM call
            return AgentResult(agent=self.name, answer=" ".join(notes), data=data)

        concepts = self.retriever.retrieve(
            f"{inp['question']} moving average trend volatility 52-week",
            k=3,
            categories=("markets", "portfolio"),
        )
        numbered = "\n\n".join(
            f"[{i}] {c.title} — {c.section}\n{c.text}" for i, c in enumerate(concepts, 1)
        )
        system = SYSTEM.format(
            level_style=LEVEL_STYLE[inp["profile"].knowledge_level],
            facts="\n\n".join([*facts, *(f"NOTE: {n}" for n in notes)]),
            sources=numbered or "(none)",
        )
        reply = self.model.invoke([SystemMessage(system), HumanMessage(inp["question"])])
        answer = reply.text.strip() + "\n\n" + self._data_footer(sources, stale)
        return AgentResult(
            agent=self.name,
            answer=answer,
            citations=cited_chunks(reply.text, concepts),
            data=data,
        )

    def _name(self, ticker: str) -> str | None:
        try:
            return self.market.get_profile(ticker).name
        except AppError:
            return None  # name is nice-to-have; Alpha Vantage free tier can't provide it

    @staticmethod
    def _data_footer(sources: set[str], stale: bool) -> str:
        """Deterministic data-freshness line (REQ-MK-04)."""
        line = f"_Market data: {', '.join(sorted(sources))}. Quotes may be delayed._"
        if stale:
            line += " ⚠️ _Some values are from cache because live data was unavailable._"
        return line
