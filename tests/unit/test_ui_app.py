"""Streamlit app tests with AppTest and a fake service (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.core.services as services_module
import src.workflow.graph as graph_module
from src.core.config import PROJECT_ROOT, Settings
from src.core.errors import LLMUnavailableError
from tests.services import QA_REPLY, fake_service

APP = str(PROJECT_ROOT / "src" / "web_app" / "ui" / "app.py")


@pytest.fixture(autouse=True)
def _fresh_resource_cache() -> None:
    """st.cache_resource is process-wide; each test must build its own service."""
    st.cache_resource.clear()


@pytest.fixture
def fake(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(services_module, "build_service", lambda _=None: fake_service(settings))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def _charts(at: AppTest) -> list[Any]:
    return list(at.get("plotly_chart"))


def test_layout_tabs_and_disclaimer(fake: None) -> None:
    """REQ-UI-01, REQ-UI-07: five tabs; the disclaimer sits above them (visible on every tab)."""
    at = _app()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "💬 Chat",
        "📊 Portfolio",
        "📈 Market",
        "🎯 Goals",
        "📚 Learn",
    ]
    assert "not financial" in at.caption[0].value  # first element under the title


def test_chat_turn_shows_answer_agent_and_sources(fake: None) -> None:
    """REQ-UI-02: answer rendered with the answering agent and expandable sources."""
    at = _app()
    at.chat_input[0].set_value("What is an index fund?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert QA_REPLY in text
    assert any("Answered by: Finance Q&A" in c.value for c in at.caption)
    assert any(e.label == "Sources (1)" for e in at.expander)
    assert any(m.value.startswith("1. [") for m in at.markdown)  # numbered like the [n] markers
    assert len(at.session_state.history) == 2


def test_new_conversation_resets_thread(fake: None) -> None:
    """REQ-WF-04: 'New conversation' starts a fresh thread and clears the history."""
    at = _app()
    at.chat_input[0].set_value("What is an index fund?").run()
    old_thread = at.session_state.thread_id
    at.sidebar.button[0].click().run()
    assert at.session_state.thread_id != old_thread
    assert at.session_state.history == []


def test_portfolio_tab_analyzes_sample(fake: None) -> None:
    """REQ-UI-03: holdings table -> metric cards, donut, sector bars, performance vs SPY."""
    at = _app()
    at.selectbox(key="pf_sample").set_value("concentrated_tech").run()
    at.button(key="pf_analyze").click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"Total value", "Effective holdings", "Beta vs SPY", "Max drawdown"} <= set(labels)
    titles = [c.proto.spec for c in _charts(at)]
    assert any("By asset class" in t for t in titles)
    assert any("Growth of 100 vs SPY" in t for t in titles)
    assert any("⚠" in (w.icon or "") or "%" in w.value for w in at.warning)  # concentration flags
    assert [h.ticker for h in at.session_state.portfolio][:2] == ["NVDA", "AAPL"]


def test_portfolio_shared_with_chat(fake: None) -> None:
    """REQ-WF-08: analyzed holdings are sent with chat requests."""
    at = _app()
    at.button(key="pf_analyze").click().run()
    assert any("Portfolio shared with chat" in c.value for c in at.sidebar.caption)


def test_market_tab_overview_heatmap_and_ticker(fake: None) -> None:
    """REQ-UI-04: index metrics, sector heatmap, SMA chart, as-of caption, refresh control."""
    at = _app()
    titles = [c.proto.spec for c in _charts(at)]
    assert any("Sectors today" in t for t in titles)
    assert any("SPY — daily close" in t and "50-day avg" in t for t in titles)
    assert any(c.value.startswith("As of ") for c in at.caption)
    at.button(key="mk_refresh").click().run()
    assert not at.exception
    at.text_input(key="mk_ticker").set_value("aapl").run()
    assert any("AAPL — daily close" in c.proto.spec for c in _charts(at))


def test_market_tab_invalid_ticker_safe_message(fake: None) -> None:
    """REQ-UI-04, REQ-API-03: an unknown ticker shows a friendly error, no traceback."""
    at = _app()
    at.text_input(key="mk_ticker").set_value("ZZQXW").run()
    assert not at.exception
    assert any("ZZQXW" in e.value for e in at.error)


def test_goals_tab_projection_and_levers(fake: None) -> None:
    """REQ-UI-05: goal inputs -> projection with a scenario band and gap levers."""
    at = _app()
    submit = next(b for b in at.button if b.label == "Project my goal")
    submit.click().run()
    assert not at.exception
    assert any(
        "Projected balance" in c.proto.spec and "Pessimistic" in c.proto.spec for c in _charts(at)
    )
    assert any(m.label == "At your planned saving" and "Short" in m.value for m in at.metric)
    assert any("Ways to close the gap" in i.value for i in at.info)


def test_goals_quiz_sets_risk_profile(fake: None) -> None:
    """REQ-GP-01: the questionnaire sets the profile used by the form and the sidebar."""
    at = _app()
    for r in at.radio:
        if r.key and r.key.startswith("rq_"):
            r.set_value(r.options[2])
    next(b for b in at.button if b.label == "Score my answers").click().run()
    assert at.session_state.risk_profile == "aggressive"
    assert any("aggressive" in s.value for s in at.success)


def test_learn_tab_filters_by_category(fake: None) -> None:
    """REQ-UI-06: articles browsable by category and searchable by title."""
    at = _app()
    category = next(s for s in at.selectbox if s.label == "Category")
    category.set_value("tax").run()
    count = next(c.value for c in at.caption if c.value.endswith("article(s)"))
    assert int(count.split()[0]) >= 10
    at.text_input(key="learn_query").set_value("roth").run()
    assert all("roth" in e.label.lower() for e in at.expander if "·" in e.label)


def test_app_errors_show_safe_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-API-03: failures show a user-safe message, not a stack trace."""

    def broken(_: Any = None) -> Any:
        raise LLMUnavailableError("openai 503 x3")

    monkeypatch.setattr(services_module, "build_service", broken)
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

    monkeypatch.setattr(services_module, "build_service", broken)
    at = _app()
    assert [e.value for e in at.error] == [
        "The assistant failed to start. Details are in the server logs."
    ]
    assert len(at.chat_input) == 0  # the page stops before offering a chat box


def test_model_error_during_chat_shows_safe_message(
    fake: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-API-03: a provider error mid-conversation shows a safe message."""
    from langchain_core.exceptions import ModelAPIError

    def explode(*_: Any, **__: Any) -> Any:
        raise ModelAPIError("upstream 500")

    at = _app()
    monkeypatch.setattr(graph_module, "run_safely", lambda agent, state: explode())
    at.chat_input[0].set_value("What is an index fund?").run()
    assert any("misconfigured or unavailable" in e.value for e in at.error)
