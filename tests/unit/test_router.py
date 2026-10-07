from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.exceptions import ModelAPIError, ModelAuthenticationError
from langchain_core.messages import AIMessage, HumanMessage
from structlog.testing import capture_logs

from src.workflow.router import RouteDecision, Router, apply_tax_guard, keyword_route
from tests.fakes import ScriptedChatModel, route_json, seen_text

CASES: list[dict[str, Any]] = yaml.safe_load(
    (Path(__file__).parents[1] / "fixtures" / "router_cases.yaml").read_text()
)


def _router(
    reply: str = "", *, fail: bool = False, fail_with: type[Exception] = ModelAPIError
) -> tuple[Router, ScriptedChatModel]:
    model = ScriptedChatModel(reply=reply, fail=fail, fail_with=fail_with)
    return Router(model.with_structured_output(RouteDecision), history_messages=4), model


def _state(*texts: str) -> dict[str, Any]:
    msgs = [HumanMessage(t) if i % 2 == 0 else AIMessage(t) for i, t in enumerate(texts)]
    return {"messages": msgs}


def test_llm_decision_is_used() -> None:
    """REQ-WF-01: the router returns the LLM's intents and standalone question."""
    router, _ = _router(route_json(["market", "news"], "What is NVDA's price and news?"))
    out = router(_state("NVDA price and news?"))
    assert out == {"intents": ["market", "news"], "question": "What is NVDA's price and news?"}


def test_intents_deduped_and_out_of_scope_dropped_when_mixed() -> None:
    """REQ-WF-01: a real finance intent wins over 'out_of_scope'; duplicates removed."""
    router, _ = _router(route_json(["qa", "out_of_scope", "qa"]))
    assert router(_state("x"))["intents"] == ["qa"]


def test_history_is_in_router_prompt() -> None:
    """REQ-WF-04: the router sees recent turns to resolve follow-ups."""
    router, model = _router(route_json(["qa"], "What are index fund fees?"))
    router(_state("What is an index fund?", "An index fund tracks...", "what about its fees?"))
    prompt = seen_text(model)
    assert "What is an index fund?" in prompt
    assert "what about its fees?" in prompt


def test_llm_failure_falls_back_to_keywords() -> None:
    """REQ-WF-03: if the routing LLM fails, the keyword router decides."""
    router, _ = _router(fail=True)
    with capture_logs() as logs:
        out = router(_state("What's the current price of NVDA?"))
    assert out["intents"] == ["market"]
    assert out["question"] == "What's the current price of NVDA?"
    assert any(e["event"] == "router_keyword_fallback" for e in logs)


def test_config_errors_are_not_masked_by_keywords() -> None:
    """REQ-LLM-07: a bad key fails loud even here — keywords must not hide it."""
    router, _ = _router(fail=True, fail_with=ModelAuthenticationError)
    with pytest.raises(ModelAuthenticationError):
        router(_state("What is an ETF?"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is a Roth IRA?", ["tax"]),
        ("my portfolio is 10 AAPL", ["portfolio"]),
        ("price of $NVDA", ["market"]),
        ("any news on apple?", ["news"]),
        ("how much should I save for retirement", ["goals"]),
        ("What's NVDA trading at and any news?", ["news", "market"]),
        ("What is an index fund?", ["qa"]),
    ],
)
def test_keyword_route(text: str, expected: list[str]) -> None:
    """REQ-WF-03: keyword rules map obvious phrasing to intents; default is qa."""
    assert keyword_route(text).intents == expected


def test_keyword_router_baseline_accuracy() -> None:
    """REQ-WF-03: the fallback is good enough to keep the app useful (>= 80% on in-scope cases).

    Keywords can't recognize out-of-scope text (they default to qa), so those cases are excluded.
    """
    # Keywords can't judge concept-vs-data nuance; baseline covers the original core cases.
    in_scope = [c for c in CASES if c["expected"] != ["out_of_scope"] and "forbidden" not in c]
    hits = [c for c in in_scope if set(c["expected"]) <= set(keyword_route(c["text"]).intents)]
    accuracy = len(hits) / len(in_scope)
    assert accuracy >= 0.8, f"keyword accuracy {accuracy:.0%}"


@pytest.mark.live
def test_live_llm_router_accuracy() -> None:
    """REQ-WF-01: the real router labels >= 90% of the case set correctly (needs keys)."""
    from src.core.llm import get_chat_model

    router = Router(get_chat_model("router", structured_output=RouteDecision))
    misses = []
    for case in CASES:
        intents = router(_state(case["text"]))["intents"]
        forbidden = set(case.get("forbidden", []))
        if not set(case["expected"]) <= set(intents) or forbidden & set(intents):
            misses.append((case["text"], case["expected"], intents))
    accuracy = 1 - len(misses) / len(CASES)
    assert accuracy >= 0.9, f"accuracy {accuracy:.0%}; misses: {misses}"


@pytest.mark.parametrize(
    ("intents", "text", "expected", "changed"),
    [
        (["qa"], "How does the wash sale rule work?", ["tax"], True),
        (["goals"], "Should I max my 401(k) to retire early?", ["goals", "tax"], True),
        (["out_of_scope"], "Is my Roth IRA taxed?", ["tax"], True),
        (["tax", "qa"], "Roth IRA basics", ["tax", "qa"], False),  # LLM already chose tax
        (["qa"], "What is an index fund?", ["qa"], False),  # no tax terms
        (["market"], "What's AAPL trading at?", ["market"], False),
    ],
)
def test_tax_guard(intents: list[str], text: str, expected: list[str], changed: bool) -> None:
    """REQ-WF-01, REQ-TX-01: precise tax terms always reach the Tax agent."""
    assert apply_tax_guard(intents, text) == (expected, changed)  # type: ignore[arg-type]


def test_router_applies_tax_guard_to_llm_decision() -> None:
    """REQ-TX-01: an LLM mis-route of a tax question is corrected and logged."""
    router, _ = _router(route_json(["qa"], "How does the wash sale rule work?"))
    with capture_logs() as logs:
        out = router(_state("How does the wash sale rule work?"))
    assert out["intents"] == ["tax"]
    assert next(e for e in logs if e["event"] == "router_decision")["tax_guard"] is True
