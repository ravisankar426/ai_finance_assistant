"""Portfolio Analysis agent (REQ-PF-01..05, REQ-WF-09, REQ-WF-10).

check inputs (holdings from this message, or remembered from earlier turns / the Portfolio tab)
-> compute (portfolio engine) -> retrieve concept articles -> explain with educational,
non-security-specific recommendations. Parsed holdings are remembered for follow-ups.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.agents.grounded import LEVEL_STYLE, renumber_citations, user_turn
from src.core.config import PortfolioConfig
from src.core.llm import CONFIG_ERRORS
from src.core.models import AgentResult, Holding
from src.core.portfolio_engine import analyze_portfolio, describe_analysis
from src.data.market_service import MarketDataService
from src.rag.retriever import Retriever
from src.utils.logging import get_logger
from src.workflow.state import AgentInput

log = get_logger(__name__)

FOLLOW_UP = (
    "Please share your holdings — for example “10 AAPL at $150, 5 MSFT, 20 VTI” (the price you "
    "paid is optional) — or upload a CSV in the Portfolio tab."
)


class HoldingsRequest(BaseModel):
    """Holdings stated in the user's message."""

    holdings: list[Holding] = Field(
        default_factory=list,
        description="Positions the user says they own: ticker (map names like Apple -> AAPL), "
        "share quantity, and cost_basis = price paid PER SHARE if stated.",
    )


EXTRACT_PROMPT = """Extract the investment positions the user says they currently own.
- ticker: US symbol in upper case (map well-known names: Apple -> AAPL,
  Vanguard total market -> VTI).
- quantity: number of shares. If only a dollar amount is given, skip that position.
- cost_basis: price paid per share, only if stated (convert "I paid $1,500 for 10 shares" to 150).
Return an empty list if the message doesn't list holdings (e.g. "how diversified am I?")."""

_PAIR = re.compile(r"(\d+(?:\.\d+)?)\s*(?:shares?\s+(?:of\s+)?)?\$?([A-Z]{1,5})\b")

SYSTEM = """You are the Portfolio Analysis agent of an EDUCATIONAL personal-finance assistant.
Rules:
- Other specialists answer other parts of the user's message (prices, news, taxes, goals,
  portfolios). Cover only your part; never say you lack information for the other parts.
- Explain ONLY the facts in the PORTFOLIO block; never compute new numbers.
- Explain what the metrics mean (weights, sector mix, HHI, volatility, max drawdown, Sharpe,
  beta) in plain terms, citing the numbered concept sources like [1].
- Explain each FLAG and give EDUCATIONAL, non-security-specific observations — e.g. how
  diversification across sectors or broad index funds reduces concentration risk, what
  rebalancing is. NEVER tell the user to buy, sell, or hold any specific security, and never
  name a specific fund to buy.
- Past returns and volatility don't predict the future — say so once.
- Mention skipped tickers and data notes plainly. Keep it under 250 words; no disclaimer.
{level_style}

PORTFOLIO (computed by the system):
{facts}

Concept sources:
{sources}"""


def regex_holdings(text: str) -> list[Holding]:
    """Fallback parser for '10 AAPL', '5 shares of MSFT' (no cost basis)."""
    return [Holding(ticker=t, quantity=float(q)) for q, t in _PAIR.findall(text)]


class PortfolioAgent:
    """Analyzes the user's holdings with the deterministic engine."""

    name = "portfolio"

    def __init__(
        self,
        model: Runnable[Any, Any],
        extractor: Runnable[Any, Any],
        market: MarketDataService,
        retriever: Retriever,
        config: PortfolioConfig,
    ) -> None:
        self.model = model
        self.extractor = extractor  # structured output -> HoldingsRequest
        self.market = market
        self.retriever = retriever
        self.config = config

    def _parse(self, text: str) -> list[Holding]:
        try:
            req: HoldingsRequest = self.extractor.invoke(
                [SystemMessage(EXTRACT_PROMPT), HumanMessage(text)]
            )
            return req.holdings
        except CONFIG_ERRORS:
            raise
        except Exception as exc:
            log.warning("holdings_extraction_fallback", error_type=type(exc).__name__)
            return regex_holdings(text)

    def run(self, inp: AgentInput) -> AgentResult:
        """Analyze holdings from this message, or the remembered portfolio."""
        latest = inp["messages"][-1].text
        new = self._parse(latest)
        holdings = new or list(inp.get("portfolio") or [])
        if not holdings:
            return AgentResult(agent=self.name, follow_up_question=FOLLOW_UP)  # REQ-WF-10
        updates = {"portfolio": new} if new else {}

        analysis = analyze_portfolio(holdings, self.market, self.config)
        data: dict[str, Any] = {
            "analysis": analysis.model_dump(
                mode="json", exclude={"performance"}
            )  # keep checkpoints small
        }
        if not analysis.holdings:
            names = ", ".join(analysis.skipped) or "those holdings"
            return AgentResult(
                agent=self.name,
                answer=f"I couldn't find market data for {names}, so I can't analyze them. "
                "Please check the ticker symbols.",
                data=data,
                state_updates=updates,
            )

        concepts = self.retriever.retrieve(
            "diversification concentration risk volatility beta Sharpe ratio rebalancing",
            k=4,
            categories=("portfolio",),
        )
        sources = "\n\n".join(
            f"[{i}] {c.title} — {c.section}\n{c.text}" for i, c in enumerate(concepts, 1)
        )
        system = SYSTEM.format(
            level_style=LEVEL_STYLE[inp["profile"].knowledge_level],
            facts=describe_analysis(analysis),
            sources=sources or "(none)",
        )
        reply = self.model.invoke([SystemMessage(system), user_turn(inp)])
        footer = (
            f"_Prices: {', '.join(analysis.sources)}. Metrics use past data and don't predict "
            "future results._"
        )
        body, citations = renumber_citations(reply.text.strip(), [c.citation() for c in concepts])
        return AgentResult(
            agent=self.name,
            answer=f"{body}\n\n{footer}",
            citations=citations,
            data=data,
            state_updates=updates,
        )
