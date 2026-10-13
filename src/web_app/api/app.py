"""REST API (REQ-API-01..03). Run with ``make api`` -> http://localhost:8000/docs.

Error contract (REQ-API-03): validation problems -> 422 with readable messages; expected
failures -> their status with a user-safe message; anything else -> 500 with only a
``request_id``. Every response carries an ``X-Request-ID`` header that matches the logs.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.exceptions import ModelError
from pydantic import BaseModel
from starlette.responses import Response

from src.core.config import get_settings
from src.core.errors import (
    AppError,
    ConfigurationError,
    InvalidTickerError,
    KnowledgeBaseError,
    LLMUnavailableError,
    MarketDataUnavailableError,
)
from src.core.goal_engine import GoalPlan, RiskProfile
from src.core.models import Holding
from src.core.portfolio_engine import PortfolioAnalysis
from src.core.services import (
    ArticleSummary,
    AssistantService,
    ChatRequest,
    ChatResponse,
    GoalPlanRequest,
    TrendResponse,
    build_service,
)
from src.data.models import Quote
from src.data.overview import MarketOverview
from src.utils.logging import configure_logging, get_logger, new_request_id, request_context

log = get_logger(__name__)

STATUS: dict[type[AppError], int] = {
    InvalidTickerError: 404,
    MarketDataUnavailableError: 503,
    LLMUnavailableError: 503,
    KnowledgeBaseError: 503,
    ConfigurationError: 500,
}


class PortfolioRequest(BaseModel):
    """Holdings to analyze."""

    holdings: list[Holding]


class RiskAnswers(BaseModel):
    """Questionnaire answers keyed by question id."""

    answers: dict[str, str]


class RiskResult(BaseModel):
    """Scored risk profile."""

    risk_profile: RiskProfile


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(
    service: AssistantService | None = None, factory: Callable[[], AssistantService] = build_service
) -> FastAPI:
    """Build the API. Pass ``service`` in tests; otherwise it's built once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(level=settings.app.log_level, fmt=settings.app.log_format)
        app.state.service = service or factory()  # fail fast on misconfiguration
        yield

    app = FastAPI(
        title="AI Finance Assistant API",
        version="0.1.0",
        description="Educational personal-finance assistant: multi-agent chat, portfolio "
        "analysis, market data, goal planning. Not financial advice.",
        lifespan=lifespan,
    )

    def svc(request: Request) -> AssistantService:
        return request.app.state.service  # type: ignore[no-any-return]

    # -- cross-cutting: request ids and errors ------------------------------------------------

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or new_request_id()
        request.state.request_id = rid
        with request_context(rid, path=request.url.path):
            response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    def _error(request: Request, status: int, message: str, **extra: Any) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status,
            content={"error": message, "request_id": rid, **extra},
            headers={"X-Request-ID": rid or ""},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
        ]
        return _error(request, 422, "Invalid request.", details=details)

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        status = next((code for cls, code in STATUS.items() if isinstance(exc, cls)), 500)
        log.warning("api_app_error", error_type=type(exc).__name__, status=status)
        return _error(request, status, exc.user_message)

    @app.exception_handler(ModelError)
    async def model_error(request: Request, exc: ModelError) -> JSONResponse:
        log.error("api_model_error", error_type=type(exc).__name__)
        return _error(request, 503, "The AI service is unavailable or misconfigured.")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("api_unexpected_error", error_type=type(exc).__name__)
        return _error(request, 500, "Internal error.")  # no internals (REQ-API-03)

    # -- endpoints ----------------------------------------------------------------------------

    @app.get("/health", tags=["ops"])
    def health(request: Request) -> dict[str, Any]:
        """Liveness plus a few readiness facts."""
        s = svc(request)
        return {
            "status": "ok",
            "market_providers": [p.name for p in s.market.providers],
            "articles": len(s.articles),
        }

    @app.post("/chat", tags=["chat"])
    async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        """Chat with Server-Sent Events: ``token`` events while agents write, then ``final``."""
        s, rid = svc(request), request.state.request_id

        async def events() -> AsyncIterator[str]:
            try:
                async for kind, payload in s.chat_stream(body, rid):
                    data = (
                        payload.model_dump(mode="json")
                        if isinstance(payload, BaseModel)
                        else {"text": payload}
                    )
                    yield _sse(kind, data)
            except AppError as exc:
                yield _sse("error", {"error": exc.user_message, "request_id": rid})
            except Exception as exc:  # stream already started: report in-band
                log.exception("chat_stream_failed", error_type=type(exc).__name__)
                yield _sse("error", {"error": "Internal error.", "request_id": rid})

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/chat/sync", tags=["chat"], response_model=ChatResponse)
    def chat_sync(body: ChatRequest, request: Request) -> ChatResponse:
        """Chat without streaming (simple clients, scripts)."""
        return svc(request).chat(body, request.state.request_id)

    @app.post("/portfolio/analyze", tags=["portfolio"], response_model=PortfolioAnalysis)
    def analyze(body: PortfolioRequest, request: Request) -> PortfolioAnalysis:
        """Deterministic portfolio metrics and flags."""
        return svc(request).analyze_portfolio(body.holdings)

    @app.get("/market/quote/{ticker}", tags=["market"], response_model=Quote)
    def quote(ticker: str, request: Request) -> Quote:
        """Latest quote with source and freshness."""
        return svc(request).quote(ticker)

    @app.get("/market/trend/{ticker}", tags=["market"], response_model=TrendResponse)
    def trend(ticker: str, request: Request) -> TrendResponse:
        """Return moving averages, returns, 52-week range, volatility, and daily bars."""
        return svc(request).trend(ticker)

    @app.get("/market/overview", tags=["market"], response_model=MarketOverview)
    def overview(request: Request) -> MarketOverview:
        """Index ETFs and the 11 sector ETFs."""
        return svc(request).overview()

    @app.post("/goals/plan", tags=["goals"], response_model=GoalPlan)
    def goal_plan(body: GoalPlanRequest, request: Request) -> GoalPlan:
        """Return required saving, scenarios, levers, and an illustrative allocation."""
        return svc(request).plan_goal(body)

    @app.get("/goals/questionnaire", tags=["goals"])
    def questionnaire(request: Request) -> list[dict[str, object]]:
        """Return the 5-question risk questionnaire."""
        return svc(request).questionnaire()

    @app.post("/goals/risk-profile", tags=["goals"], response_model=RiskResult)
    def risk(body: RiskAnswers, request: Request) -> RiskResult:
        """Score questionnaire answers."""
        try:
            return RiskResult(risk_profile=svc(request).risk_profile(body.answers))
        except ValueError as exc:
            raise RequestValidationError(
                [{"loc": ("answers",), "msg": str(exc), "type": "value_error"}]
            ) from exc

    @app.get("/learn/articles", tags=["learn"], response_model=list[ArticleSummary])
    def articles(
        request: Request, category: str | None = Query(default=None)
    ) -> list[ArticleSummary]:
        """Knowledge-base articles, optionally filtered by category."""
        return svc(request).list_articles(category)

    return app


app = create_app()  # for `uvicorn src.web_app.api.app:app`
