"""Streamlit app tests with AppTest and a fake graph (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.workflow.graph as graph_module
from src.agents.placeholder import PlaceholderAgent
from src.agents.qa import QAAgent
from src.core.config import PROJECT_ROOT, Settings
from src.core.errors import LLMUnavailableError
from src.core.models import AGENT_NAMES
from src.workflow.router import RouteDecision, Router
from tests.fakes import ScriptedChatModel, route_json
from tests.kb import offline_retriever

APP = str(PROJECT_ROOT / "src" / "web_app" / "ui" / "app.py")


@pytest.fixture(autouse=True)
def _fresh_resource_cache() -> None:
    """st.cache_resource is process-wide; each test must build its own graph."""
    st.cache_resource.clear()


@pytest.fixture
def fake_graph(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def build(_: Any = None) -> Any:
        agents: dict[str, Any] = {n: PlaceholderAgent(n) for n in AGENT_NAMES}
        agents["qa"] = QAAgent(
            ScriptedChatModel(reply="Index funds track an index [1]."),
            offline_retriever(),
        )
        router = Router(
            ScriptedChatModel(reply=route_json(["qa"])).with_structured_output(RouteDecision)
        )
        return graph_module.build_graph(router=router, agents=agents)

    monkeypatch.setattr(graph_module, "build_default_graph", build)


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_layout_tabs_and_disclaimer(fake_graph: None) -> None:
    """REQ-UI-01, REQ-UI-07: five tabs and a disclaimer visible on the page."""
    at = _app()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "💬 Chat",
        "📊 Portfolio",
        "📈 Market",
        "🎯 Goals",
        "📚 Learn",
    ]
    assert any("not financial" in c.value for c in at.caption)


def test_chat_turn_shows_answer_agent_and_sources(fake_graph: None) -> None:
    """REQ-UI-02: answer rendered with the answering agent and expandable sources."""
    at = _app()
    at.chat_input[0].set_value("What is an index fund?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Index funds track an index [1]." in text
    assert any("Answered by: Finance Q&A" in c.value for c in at.caption)
    assert any(e.label == "Sources (1)" for e in at.expander)
    assert len(at.session_state.history) == 2


def test_new_conversation_resets_thread(fake_graph: None) -> None:
    """REQ-WF-04: 'New conversation' starts a fresh thread and clears the history."""
    at = _app()
    at.chat_input[0].set_value("What is an index fund?").run()
    old_thread = at.session_state.thread_id
    at.sidebar.button[0].click().run()
    assert at.session_state.thread_id != old_thread
    assert at.session_state.history == []


def test_app_errors_show_safe_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-API-03: failures show a user-safe message, not a stack trace."""

    def broken(_: Any = None) -> Any:
        raise LLMUnavailableError("openai 503 x3")

    monkeypatch.setattr(graph_module, "build_default_graph", broken)
    at = _app()  # fails on page load, before any question is asked
    assert not at.exception
    assert [e.value for e in at.error] == [LLMUnavailableError.user_message]
    assert "503" not in " ".join(e.value for e in at.error)


def test_app_file_exists() -> None:
    """REQ-UI-01: the documented entry point exists."""
    assert Path(APP).is_file()


def test_startup_misconfiguration_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LLM-07: a startup failure (e.g. missing key) is shown at once, not on first question."""

    def broken(_: Any = None) -> Any:
        raise RuntimeError("no OPENAI_API_KEY")

    monkeypatch.setattr(graph_module, "build_default_graph", broken)
    at = _app()
    assert [e.value for e in at.error] == [
        "The assistant failed to start. Details are in the server logs."
    ]
    assert len(at.chat_input) == 0  # the page stops before offering a chat box


def test_model_error_during_chat_shows_safe_message(
    fake_graph: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-API-03: a provider error mid-conversation shows a safe message."""
    from langchain_core.exceptions import ModelAPIError

    import src.agents.qa as qa_module

    def explode(*_: Any, **__: Any) -> Any:
        raise ModelAPIError("upstream 500")

    at = _app()
    monkeypatch.setattr(qa_module.QAAgent, "run", explode)
    monkeypatch.setattr(graph_module, "run_safely", lambda agent, state: explode())
    at.chat_input[0].set_value("What is an index fund?").run()
    assert any("misconfigured or unavailable" in e.value for e in at.error)
