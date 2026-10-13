"""Service layer shared by the REST API, the Streamlit UI, and the MCP server (design §1).

Every business operation lives here once; the interfaces only translate HTTP/UI/MCP calls.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.core.config import Settings, get_settings
from src.core.goal_engine import (
    QUESTIONNAIRE,
    GoalPlan,
    RiskProfile,
    plan_goal,
    risk_profile_from_answers,
)
from src.core.llm import track_usage
from src.core.market_analytics import TrendSnapshot, trend_snapshot
from src.core.models import AGENT_LABELS, Citation, Holding, UserProfile
from src.core.portfolio_engine import PortfolioAnalysis, analyze_portfolio
from src.data.market_service import MarketDataService, build_market_service
from src.data.models import PriceBar, PriceHistory, Quote
from src.data.overview import MarketOverview, market_overview
from src.rag.knowledge_base import Article, load_articles

AGENT_NODE_PREFIX = "agent_"


class ChatRequest(BaseModel):
    """One user message."""

    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=100)
    profile: UserProfile | None = None
    portfolio: list[Holding] | None = None


class ChatResponse(BaseModel):
    """The assistant's reply plus everything the UI needs to render it."""

    thread_id: str
    request_id: str
    answer: str
    agents: list[str]
    citations: list[Citation]
    data: dict[str, Any]  # per-agent structured payloads (charts)
    learn_next: list[dict[str, str]]
    llm_calls: int
    cost_usd: float
    latency_ms: int


class GoalPlanRequest(BaseModel):
    """Inputs for a deterministic goal plan."""

    goal: str = "my goal"
    target_amount: float = Field(gt=0)
    years: float = Field(gt=0, le=80)
    current_savings: float = Field(default=0, ge=0)
    monthly_contribution: float | None = Field(default=None, ge=0)
    risk_profile: RiskProfile


class TrendResponse(BaseModel):
    """Trend indicators plus the daily bars for charting."""

    snapshot: TrendSnapshot
    bars: list[PriceBar]


class ArticleSummary(BaseModel):
    """Knowledge-base article for the Learn tab."""

    id: str
    title: str
    category: str
    difficulty: str
    source_name: str
    source_url: str
    tax_year: int | None
    body: str


class AssistantService:
    """Facade over the graph, market data, engines, and knowledge base."""

    def __init__(
        self,
        graph: CompiledStateGraph[Any, Any, Any, Any],
        market: MarketDataService,
        settings: Settings,
        articles: list[Article],
    ) -> None:
        self.graph = graph
        self.market = market
        self.settings = settings
        self.articles = articles

    # -- chat ----------------------------------------------------------------------------------

    def _inputs(self, req: ChatRequest, request_id: str) -> dict[str, Any]:
        inputs: dict[str, Any] = {"messages": [HumanMessage(req.message)], "request_id": request_id}
        if req.profile is not None:
            inputs["profile"] = req.profile  # only override what the client sent
        if req.portfolio is not None:
            inputs["portfolio"] = req.portfolio
        return inputs

    @staticmethod
    def _response(
        final: dict[str, Any], thread_id: str, request_id: str, usage: Any, start: float
    ) -> ChatResponse:
        results = final.get("agent_results", [])
        return ChatResponse(
            thread_id=thread_id,
            request_id=request_id,
            answer=final.get("final_answer", ""),
            agents=[
                AGENT_LABELS.get(r.agent, r.agent)
                for r in results
                if r.agent not in ("out_of_scope", "guard")
            ],
            citations=final.get("citations", []),
            data={r.agent: r.data for r in results if r.data},
            learn_next=[a for r in results for a in r.data.get("learn_next", [])][:3],
            llm_calls=usage.calls,
            cost_usd=round(usage.cost_usd, 6),
            latency_ms=round((time.perf_counter() - start) * 1000),
        )

    def chat(self, req: ChatRequest, request_id: str | None = None) -> ChatResponse:
        """Run one turn and return the final answer (REQ-API-01)."""
        thread_id = req.thread_id or uuid.uuid4().hex
        request_id = request_id or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        with track_usage() as usage:
            final = self.graph.invoke(
                self._inputs(req, request_id), {"configurable": {"thread_id": thread_id}}
            )
        return self._response(final, thread_id, request_id, usage, start)

    @staticmethod
    def _agent_token(chunk: Any) -> str:
        """Text of an agent's streamed token; empty for the router, guards, and summaries."""
        message, meta = chunk
        node = str(meta.get("langgraph_node", ""))
        if isinstance(message, AIMessageChunk) and node.startswith(AGENT_NODE_PREFIX):
            return message.text
        return ""

    def chat_events(
        self, req: ChatRequest, request_id: str | None = None
    ) -> Iterator[tuple[str, Any]]:
        """Stream for Streamlit (sync): ("token", text)..., then ("final", ChatResponse)."""
        thread_id = req.thread_id or uuid.uuid4().hex
        request_id = request_id or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        final: dict[str, Any] = {}
        with track_usage() as usage:
            for item in self.graph.stream(
                self._inputs(req, request_id),
                {"configurable": {"thread_id": thread_id}},
                stream_mode=["messages", "values"],
            ):
                mode, chunk = cast(tuple[str, Any], item)
                if mode == "values":
                    final = cast(dict[str, Any], chunk)
                elif text := self._agent_token(chunk):
                    yield "token", text
        yield "final", self._response(final, thread_id, request_id, usage, start)

    async def chat_stream(
        self, req: ChatRequest, request_id: str | None = None
    ) -> AsyncIterator[tuple[str, Any]]:
        """Stream for the API's SSE endpoint (async twin of :meth:`chat_events`)."""
        thread_id = req.thread_id or uuid.uuid4().hex
        request_id = request_id or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        final: dict[str, Any] = {}
        with track_usage() as usage:
            stream = self.graph.astream(
                self._inputs(req, request_id),
                {"configurable": {"thread_id": thread_id}},
                stream_mode=["messages", "values"],
            )
            async for item in stream:
                mode, chunk = cast(tuple[str, Any], item)
                if mode == "values":
                    final = cast(dict[str, Any], chunk)
                elif text := self._agent_token(chunk):
                    yield "token", text
        yield "final", self._response(final, thread_id, request_id, usage, start)

    # -- deterministic tools (also used by the UI tabs and MCP) ----------------------------------

    def analyze_portfolio(self, holdings: list[Holding]) -> PortfolioAnalysis:
        """Portfolio metrics and flags (REQ-PF-02)."""
        return analyze_portfolio(holdings, self.market, self.settings.portfolio)

    def quote(self, ticker: str) -> Quote:
        """Latest quote (REQ-MK-01)."""
        return self.market.get_quote(ticker)

    def overview(self) -> MarketOverview:
        """Index and sector snapshot (REQ-MK-03)."""
        return market_overview(self.market)

    def trend(self, ticker: str) -> TrendResponse:
        """Trend indicators plus bars for charts (REQ-MK-02)."""
        history = self.market.get_history(ticker)
        return TrendResponse(snapshot=trend_snapshot(history), bars=history.bars)

    def plan_goal(self, req: GoalPlanRequest) -> GoalPlan:
        """Deterministic goal plan (REQ-GP-02..05)."""
        return plan_goal(
            goal=req.goal,
            target_today=req.target_amount,
            years=req.years,
            current_savings=req.current_savings,
            planned_monthly=req.monthly_contribution,
            risk_profile=req.risk_profile,
            cfg=self.settings.planning,
        )

    @staticmethod
    def questionnaire() -> list[dict[str, object]]:
        """Risk questionnaire definition (REQ-GP-01)."""
        return QUESTIONNAIRE

    @staticmethod
    def risk_profile(answers: dict[str, str]) -> RiskProfile:
        """Score questionnaire answers (REQ-GP-01)."""
        return risk_profile_from_answers(answers)

    def history(self, ticker: str) -> PriceHistory:
        """Daily price history (REQ-UI-04 charts)."""
        return self.market.get_history(ticker)

    def categories(self) -> list[str]:
        """Knowledge-base categories, sorted (REQ-UI-06)."""
        return sorted({a.category for a in self.articles})

    def list_articles(self, category: str | None = None) -> list[ArticleSummary]:
        """Knowledge-base articles, optionally by category (REQ-UI-06)."""
        return [
            ArticleSummary(**a.model_dump(exclude={"last_reviewed"}))
            for a in self.articles
            if category is None or a.category == category
        ]


def build_service(settings: Settings | None = None) -> AssistantService:
    """Build everything once: one market cache and one index shared by agents and endpoints."""
    from src.workflow.graph import build_default_graph, build_retriever

    settings = settings or get_settings()
    market = build_market_service(settings)
    graph = build_default_graph(settings, market=market, retriever=build_retriever(settings))
    return AssistantService(graph, market, settings, load_articles(settings.rag.knowledge_base_dir))
