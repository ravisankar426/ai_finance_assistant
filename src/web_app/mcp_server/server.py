"""MCP server (REQ-MCP-01): market data, portfolio analysis, goal planning, and KB search as tools.

Protocol design:
- Transport is stdio (what Claude Desktop launches). stdout carries JSON-RPC, so logs go to
  stderr.
- Tools are thin wrappers over :class:`AssistantService`, the same facade as the API and UI.
  They return Pydantic models, so clients get both text and a typed ``structuredContent``.
- Tools are deterministic (no LLM calls); the client's own model does the reasoning, and the
  tool descriptions say the data is educational, not advice.
- Expected failures (unknown ticker, data unavailable) become tool errors carrying the
  user-safe message; internals never reach the client.
- The service is built on the first call, so the handshake is instant; misconfiguration
  (missing key, no index) surfaces as a clear tool error.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from src.core.config import get_settings
from src.core.errors import AppError
from src.core.goal_engine import GoalPlan, RiskProfile
from src.core.models import Holding
from src.core.portfolio_engine import PortfolioAnalysis
from src.core.services import AssistantService, GoalPlanRequest, KnowledgeHit, build_service
from src.data.models import Quote
from src.data.overview import MarketOverview
from src.rag.knowledge_base import Category
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

INSTRUCTIONS = (
    "Educational personal-finance tools: quotes, market overview, portfolio metrics, goal "
    "projections, and a curated knowledge base (US). Results describe data; they are not "
    "investment, tax, or legal advice. Cite knowledge-base URLs when you use them."
)


def tools_only_service() -> AssistantService:
    """Build data, engines, and search without the chat workflow (clients bring a model)."""
    return build_service(chat=False)


def create_server(factory: Callable[[], AssistantService] = tools_only_service) -> MCPServer:
    """Build the MCP server; ``factory`` is swapped for a fake in tests."""
    server = MCPServer("ai-finance-assistant", instructions=INSTRUCTIONS, version="0.1.0")
    holder: dict[str, AssistantService] = {}

    def svc() -> AssistantService:
        if "s" not in holder:
            holder["s"] = factory()
        return holder["s"]

    def call[T](name: str, fn: Callable[[AssistantService], T]) -> T:
        try:
            return fn(svc())
        except AppError as exc:
            log.warning("mcp_tool_error", tool=name, error_type=type(exc).__name__)
            raise ToolError(exc.user_message) from exc
        except Exception as exc:
            log.exception("mcp_tool_failed", tool=name, error_type=type(exc).__name__)
            raise ToolError("Internal error; see the server logs.") from exc

    @server.tool()
    def get_quote(ticker: str) -> Quote:
        """Latest price for a US stock or ETF ticker (e.g. AAPL, SPY), with source and freshness."""
        return call("get_quote", lambda s: s.quote(ticker))

    @server.tool()
    def market_overview() -> MarketOverview:
        """Today's moves for the major index ETFs and the 11 S&P 500 sector ETFs."""
        return call("market_overview", lambda s: s.overview())

    @server.tool()
    def analyze_portfolio(holdings: list[Holding]) -> PortfolioAnalysis:
        """Measure a portfolio's mix, concentration, risk, and flags.

        Returns weights, asset-class and sector mix, HHI, volatility, max drawdown, Sharpe,
        beta vs SPY, and concentration flags.

        Each holding: ticker, quantity (shares), optional cost_basis (price paid per share).
        """
        return call("analyze_portfolio", lambda s: s.analyze_portfolio(holdings))

    @server.tool()
    def plan_goal(
        target_amount: float,
        years: float,
        risk_profile: RiskProfile,
        current_savings: float = 0,
        monthly_contribution: float | None = None,
        goal: str = "my goal",
    ) -> GoalPlan:
        """Project a savings goal under pessimistic, base, and optimistic returns.

        The target is in today's dollars. Returns the required monthly saving, yearly
        balances per scenario, an illustrative mix, and ways to close a gap.
        """
        req = GoalPlanRequest(
            goal=goal,
            target_amount=target_amount,
            years=years,
            current_savings=current_savings,
            monthly_contribution=monthly_contribution,
            risk_profile=risk_profile,
        )
        return call("plan_goal", lambda s: s.plan_goal(req))

    @server.tool()
    def search_knowledge_base(
        query: str, k: int = 4, category: Category | None = None
    ) -> list[KnowledgeHit]:
        """Search the curated finance-education articles; returns passages with source URLs."""
        k = max(1, min(k, 10))
        return call("search_knowledge_base", lambda s: s.search_knowledge_base(query, k, category))

    return server


def main() -> None:
    """Run over stdio with logs on stderr."""
    settings = get_settings()
    configure_logging(level=settings.app.log_level, fmt="json", stream=sys.stderr)
    create_server().run("stdio")
