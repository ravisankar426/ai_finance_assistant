"""REST API over a fully faked service (scripted models + recorded market data)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import ModelAPIError

from src.agents.placeholder import PlaceholderAgent
from src.agents.qa import QAAgent
from src.core.config import MarketConfig, Settings, get_settings
from src.core.models import AGENT_NAMES
from src.core.services import AssistantService, ChatRequest
from src.data.market_service import MarketDataService
from src.data.providers.fixture_provider import FixtureProvider
from src.rag.knowledge_base import load_articles
from src.web_app.api.app import create_app
from src.workflow.graph import build_graph
from src.workflow.router import RouteDecision, Router
from tests.fakes import ScriptedChatModel, route_json
from tests.kb import offline_retriever


def fake_service(settings: Settings, intents: list[str] | None = None) -> AssistantService:
    market = MarketDataService(
        [FixtureProvider(get_settings().market.fixture_dir)], MarketConfig(rate_limits={})
    )
    graph = build_graph(
        router=Router(
            ScriptedChatModel(reply=route_json(intents or ["qa"])).with_structured_output(
                RouteDecision
            )
        ),
        agents={
            **{n: PlaceholderAgent(n) for n in AGENT_NAMES},
            "qa": QAAgent(
                ScriptedChatModel(reply="Index funds track an index [1]."), offline_retriever()
            ),
        },
    )
    return AssistantService(graph, market, settings, load_articles(settings.rag.knowledge_base_dir))


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(service=fake_service(settings))) as c:
        yield c


def test_health_and_request_id(client: TestClient) -> None:
    """REQ-API-01, REQ-NFR-05: health check; request ids generated or propagated."""
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["articles"] >= 60
    assert len(r.headers["X-Request-ID"]) == 12
    assert (
        client.get("/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"]
        == "abc123"
    )


def test_openapi_schema_lists_endpoints(client: TestClient) -> None:
    """REQ-API-02: OpenAPI schema with request/response models."""
    paths = client.get("/openapi.json").json()["paths"]
    for p in (
        "/chat",
        "/chat/sync",
        "/portfolio/analyze",
        "/market/quote/{ticker}",
        "/market/overview",
        "/market/trend/{ticker}",
        "/goals/plan",
        "/health",
    ):
        assert p in paths


def test_chat_sync(client: TestClient) -> None:
    """REQ-API-01: non-streaming chat returns answer, agents, citations, thread id."""
    r = client.post("/chat/sync", json={"message": "What is an index fund?", "thread_id": "t1"})
    body = r.json()
    assert r.status_code == 200 and body["thread_id"] == "t1"
    assert body["answer"].startswith("Index funds track an index [1].")
    assert body["agents"] == ["Finance Q&A"] and body["citations"]
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_chat_stream_sse(client: TestClient) -> None:
    """REQ-API-01, REQ-WF-06: SSE stream ends with a final event carrying the answer."""
    with client.stream("POST", "/chat", json={"message": "What is an index fund?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [line.split(": ", 1)[1] for line in raw.splitlines() if line.startswith("event: ")]
    assert events[-1] == "final"
    final = json.loads([ln for ln in raw.splitlines() if ln.startswith("data: ")][-1][6:])
    assert final["answer"].startswith("Index funds track an index")


def test_chat_validation_error_is_readable(client: TestClient) -> None:
    """REQ-API-03: an empty message -> 422 with field-level details."""
    r = client.post("/chat/sync", json={"message": ""})
    assert r.status_code == 422 and r.json()["details"][0].startswith("message:")


def test_market_endpoints(client: TestClient) -> None:
    """REQ-API-01, REQ-MK-01..03: quote, trend with bars, overview — from recorded data."""
    q = client.get("/market/quote/aapl").json()
    assert q["ticker"] == "AAPL" and q["source"].startswith("recorded")
    t = client.get("/market/trend/SPY").json()
    assert t["snapshot"]["trend"] in {"uptrend", "downtrend", "mixed"} and len(t["bars"]) > 200
    o = client.get("/market/overview").json()
    assert len(o["indices"]) == 4 and len(o["sectors"]) == 11


def test_invalid_ticker_is_404_with_safe_message(client: TestClient) -> None:
    """REQ-API-03: expected failures map to a status and a user-safe message."""
    r = client.get("/market/quote/ZZQXW")
    assert r.status_code == 404 and "ZZQXW" in r.json()["error"]
    assert client.get("/market/quote/not!valid").status_code == 404


def test_portfolio_and_goal_endpoints(client: TestClient) -> None:
    """REQ-API-01, REQ-PF-02, REQ-GP-02: deterministic tools over HTTP."""
    a = client.post(
        "/portfolio/analyze", json={"holdings": [{"ticker": "AAPL", "quantity": 10}]}
    ).json()
    assert a["holdings"][0]["ticker"] == "AAPL" and a["total_value"] > 0
    plan = client.post(
        "/goals/plan",
        json={
            "target_amount": 60000,
            "years": 5,
            "current_savings": 10000,
            "monthly_contribution": 600,
            "risk_profile": "moderate",
        },
    ).json()
    assert plan["on_track"] is False and round(plan["target_nominal"]) == 69556


def test_goal_validation(client: TestClient) -> None:
    """REQ-API-03: impossible inputs are rejected with readable details."""
    r = client.post("/goals/plan", json={"target_amount": -1, "years": 5, "risk_profile": "wild"})
    assert r.status_code == 422 and len(r.json()["details"]) == 2


def test_questionnaire_and_risk_profile(client: TestClient) -> None:
    """REQ-GP-01: questionnaire served and scored; bad answers -> 422."""
    qs = client.get("/goals/questionnaire").json()
    answers = {q["id"]: list(q["options"])[1] for q in qs}
    assert client.post("/goals/risk-profile", json={"answers": answers}).json() == {
        "risk_profile": "moderate"
    }
    assert client.post("/goals/risk-profile", json={"answers": {"drop": "?"}}).status_code == 422


def test_learn_articles(client: TestClient) -> None:
    """REQ-UI-06: articles by category for the Learn tab."""
    tax = client.get("/learn/articles", params={"category": "tax"}).json()
    assert len(tax) >= 10 and all(a["category"] == "tax" for a in tax)


def test_unexpected_errors_hide_internals(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-API-03: 500 with a request id and no internal details."""
    svc = fake_service(settings)

    def boom(_: str) -> Any:
        raise RuntimeError("database password is hunter2")

    monkeypatch.setattr(svc, "quote", boom)
    with TestClient(create_app(service=svc), raise_server_exceptions=False) as c:
        r = c.get("/market/quote/AAPL")
    assert r.status_code == 500 and r.json()["error"] == "Internal error."
    assert "hunter2" not in r.text and r.json()["request_id"]


def test_model_errors_are_503(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-API-03: LLM provider failures -> 503 with a safe message."""
    svc = fake_service(settings)

    def down(*_: Any, **__: Any) -> Any:
        raise ModelAPIError("upstream")

    monkeypatch.setattr(svc, "chat", down)
    with TestClient(create_app(service=svc)) as c:
        r = c.post("/chat/sync", json={"message": "hi"})
    assert r.status_code == 503 and "AI service" in r.json()["error"]


def test_stream_reports_errors_in_band(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-API-03: once streaming has started, failures arrive as an 'error' event."""
    svc = fake_service(settings)

    async def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("secret")
        yield  # pragma: no cover

    monkeypatch.setattr(svc, "chat_stream", broken)
    with (
        TestClient(create_app(service=svc)) as c,
        c.stream("POST", "/chat", json={"message": "hi"}) as r,
    ):
        raw = "".join(r.iter_text())
    assert "event: error" in raw and "secret" not in raw


def test_service_keeps_remembered_state_unless_overridden(settings: Settings) -> None:
    """REQ-WF-08: the API only overrides profile/portfolio when the client sends them."""
    svc = fake_service(settings)
    inputs = svc._inputs(ChatRequest(message="hi"), "r")
    assert "profile" not in inputs and "portfolio" not in inputs
