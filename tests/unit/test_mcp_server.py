"""MCP server over the real protocol (in-memory client), backed by the fake service."""

from __future__ import annotations

from typing import Any

import pytest
from mcp.client.client import Client

from src.core.config import Settings
from src.core.services import AssistantService
from src.web_app.mcp_server.server import create_server
from tests.services import fake_service

TOOLS = {"get_quote", "market_overview", "analyze_portfolio", "plan_goal", "search_knowledge_base"}


@pytest.fixture
def service(settings: Settings) -> AssistantService:
    return fake_service(settings)


async def call(service: AssistantService, tool: str, args: dict[str, Any]) -> Any:
    async with Client(create_server(lambda: service)) as client:
        return await client.call_tool(tool, args)


async def test_lists_the_five_tools_with_schemas(service: AssistantService) -> None:
    """REQ-MCP-01: the five tools are exposed with input and output schemas."""
    async with Client(create_server(lambda: service)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOLS
    assert tools["plan_goal"].input_schema["required"] == ["target_amount", "years", "risk_profile"]
    assert tools["get_quote"].output_schema is not None
    assert all(t.description for t in tools.values())


async def test_get_quote_structured(service: AssistantService) -> None:
    """REQ-MCP-01: typed structured content plus text."""
    r = await call(service, "get_quote", {"ticker": "aapl"})
    assert not r.is_error
    assert r.structured_content["ticker"] == "AAPL" and r.structured_content["price"] > 0


async def test_market_overview(service: AssistantService) -> None:
    """REQ-MCP-01: indices and sectors."""
    r = await call(service, "market_overview", {})
    assert len(r.structured_content["sectors"]) == 11


async def test_analyze_portfolio(service: AssistantService) -> None:
    """REQ-MCP-01, REQ-PF-02: holdings in, metrics out."""
    r = await call(service, "analyze_portfolio", {"holdings": [{"ticker": "AAPL", "quantity": 10}]})
    assert r.structured_content["holdings"][0]["weight"] == pytest.approx(1.0)


async def test_plan_goal(service: AssistantService) -> None:
    """REQ-MCP-01, REQ-GP-02: same numbers as the API."""
    r = await call(
        service,
        "plan_goal",
        {
            "target_amount": 60000,
            "years": 5,
            "risk_profile": "moderate",
            "current_savings": 10000,
            "monthly_contribution": 600,
        },
    )
    assert round(r.structured_content["target_nominal"]) == 69556
    assert r.structured_content["on_track"] is False


async def test_search_knowledge_base(service: AssistantService) -> None:
    """REQ-MCP-01, REQ-RAG-05: passages carry their source URL; k is clamped."""
    r = await call(service, "search_knowledge_base", {"query": "Roth IRA", "k": 50})
    hits = r.structured_content["result"]
    assert 1 <= len(hits) <= 10 and all(h["url"].startswith("http") for h in hits)
    assert any("Roth" in h["title"] for h in hits)


async def test_expected_errors_are_tool_errors_with_safe_message(
    service: AssistantService,
) -> None:
    """REQ-MCP-01, REQ-API-03: unknown ticker -> is_error with the user-safe message."""
    r = await call(service, "get_quote", {"ticker": "ZZQXW"})
    assert r.is_error and "ZZQXW" in r.content[0].text


async def test_unexpected_errors_hide_internals(
    service: AssistantService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-API-03: internals never reach the client."""

    def boom() -> Any:
        raise RuntimeError("secret-token-123")

    monkeypatch.setattr(service, "overview", boom)
    r = await call(service, "market_overview", {})
    assert r.is_error and "secret-token-123" not in r.content[0].text


async def test_invalid_arguments_rejected(service: AssistantService) -> None:
    """REQ-MCP-01: schema validation rejects bad inputs before the service is called."""
    r = await call(
        service, "plan_goal", {"target_amount": 1000, "years": 2, "risk_profile": "yolo"}
    )
    assert r.is_error


async def test_service_is_built_lazily_once(settings: Settings) -> None:
    """REQ-MCP-01: handshake doesn't build the service; first call does, and only once."""
    built: list[int] = []

    def factory() -> AssistantService:
        built.append(1)
        return fake_service(settings)

    async with Client(create_server(factory)) as client:
        await client.list_tools()
        assert built == []
        await client.call_tool("get_quote", {"ticker": "SPY"})
        await client.call_tool("get_quote", {"ticker": "AAPL"})
    assert built == [1]


def test_tools_only_service_has_no_chat(service: AssistantService) -> None:
    """REQ-MCP-01: a service built without the workflow fails loudly if chat is attempted."""
    from src.core.errors import ConfigurationError

    tools_only = AssistantService(None, service.market, service.settings, service.articles)
    with pytest.raises(ConfigurationError):
        _ = tools_only.graph
    with pytest.raises(ConfigurationError):
        tools_only.search_knowledge_base("anything")
