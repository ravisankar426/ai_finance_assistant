"""Benchmark, judge, and report helpers, run offline against the fake service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.runnables import RunnableLambda

from src.core.config import Settings
from src.evals.benchmark import (
    QUERIES,
    QueryResult,
    find_leaks,
    percentile,
    router_accuracy,
    run_concurrent,
    run_query,
    summarize,
)
from src.evals.judge import CRITERIA, JudgeScore, averages, cited_sources, run_judge
from src.evals.report import fmt, replace_section, table, update_report
from src.workflow.router import keyword_route
from tests.services import QA_REPLY, fake_service

GOLDEN = Path(__file__).parents[1] / "evals" / "golden_qa.yaml"


def test_percentile_nearest_rank() -> None:
    """REQ-NFR-08: p50/p95 use nearest rank; empty -> None."""
    xs = [float(x) for x in range(1, 101)]
    assert percentile(xs, 50) == 50 and percentile(xs, 95) == 95 and percentile(xs, 100) == 100
    assert percentile([7.0], 95) == 7.0 and percentile([], 50) is None


def test_run_query_measures_latency_ttft_and_cost(settings: Settings) -> None:
    """REQ-NFR-02, REQ-NFR-03: one measured turn through the real service path."""
    r = run_query(fake_service(settings), "qa", "What is an index fund?")
    assert r.ok and r.agents == ["Finance Q&A"]
    assert r.latency_ms > 0 and r.ttft_ms is not None and r.ttft_ms <= r.latency_ms


def test_run_query_records_failures(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-NFR-08: a failing query is recorded, not fatal to the benchmark."""
    svc = fake_service(settings)

    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("down")

    monkeypatch.setattr(svc, "chat_events", boom)
    r = run_query(svc, "qa", "x")
    assert not r.ok and r.error == "RuntimeError: down"


def test_summarize_groups_and_overall() -> None:
    """REQ-NFR-08: per-agent and overall rows; errors excluded from latency stats."""
    rs = [
        QueryResult("qa", "a", True, latency_ms=100, ttft_ms=50, cost_usd=0.001, llm_calls=2),
        QueryResult("qa", "b", True, latency_ms=300, ttft_ms=70, cost_usd=0.003, llm_calls=4),
        QueryResult("tax", "c", False, error="x"),
    ]
    s = summarize(rs)
    assert s["qa"]["p50_ms"] == 100 and s["qa"]["p95_ms"] == 300
    assert s["qa"]["mean_cost_usd"] == pytest.approx(0.002) and s["qa"]["mean_llm_calls"] == 3
    assert s["tax"]["errors"] == 1 and s["tax"]["p50_ms"] is None
    assert s["all"]["n"] == 3


def test_query_set_covers_every_agent() -> None:
    """REQ-NFR-08: latency is reported per agent."""
    assert set(QUERIES) == {"qa", "tax", "market", "news", "portfolio", "goals", "multi"}
    assert all(len(v) >= 2 for v in QUERIES.values())


async def test_concurrent_sessions_are_isolated(settings: Settings) -> None:
    """REQ-NFR-09: 20 concurrent two-turn sessions, no errors, no cross-session leakage."""
    report = await run_concurrent(fake_service(settings), sessions=20)
    assert report.turns == 40 and report.errors == [] and report.leaks == []
    assert report.p95_turn_ms is not None


def test_leak_detector_catches_shared_threads(settings: Settings) -> None:
    """REQ-NFR-09: the isolation check itself works (two markers in one thread = leak)."""
    from src.core.services import ChatRequest

    svc = fake_service(settings)
    svc.chat(ChatRequest(message="group AlphaQ: what is an index fund?", thread_id="t"))
    svc.chat(ChatRequest(message="group BetaQ: what is a bond?", thread_id="t"))
    leaks = find_leaks(svc, {"AlphaQ": "t", "BetaQ": "t"})
    assert "AlphaQ: contains BetaQ" in leaks and "BetaQ: contains AlphaQ" in leaks


def test_router_accuracy_counts_misses() -> None:
    """REQ-NFR-08: router accuracy on labeled cases (keyword router as a stand-in)."""

    def router(state: dict[str, Any]) -> dict[str, Any]:
        return {"intents": keyword_route(state["messages"][-1].text).intents}

    cases = [
        {"text": "What's the price of AAPL?", "expected": ["market"]},
        {"text": "Tell me a joke", "expected": ["tax"]},
    ]
    acc, misses = router_accuracy(router, cases)
    assert acc == 0.5 and len(misses) == 1 and "joke" in misses[0]


def test_golden_set_shape() -> None:
    """REQ-NFR-11: ~30 golden Q&As with key points, covering grounded, data, and safety cases."""
    cases = yaml.safe_load(GOLDEN.read_text())
    assert 25 <= len(cases) <= 40
    assert {c["kind"] for c in cases} == {"grounded", "data", "safety"}
    assert all(c["q"] and c["key_points"] for c in cases)


def _judge(score: JudgeScore | Exception) -> Any:
    def grade(_: Any) -> JudgeScore:
        if isinstance(score, Exception):
            raise score
        return score

    return RunnableLambda(grade)


def test_judge_sees_cited_source_text(settings: Settings) -> None:
    """REQ-NFR-11: groundedness is judged against the cited passages' actual text."""
    seen: list[str] = []
    svc = fake_service(settings)

    def grade(prompt: str) -> JudgeScore:
        seen.append(prompt)
        return JudgeScore(
            groundedness=5, citation_faithfulness=4, compliance=5, clarity=4, rationale="ok"
        )

    results = run_judge(
        svc,
        RunnableLambda(grade),
        [{"q": "What is an index fund?", "kind": "grounded", "key_points": ["tracks an index"]}],
    )
    assert results[0].score is not None and results[0].agents == ["Finance Q&A"]
    assert "[1] " in seen[0] and QA_REPLY in seen[0] and "- tracks an index" in seen[0]
    assert "(text unavailable)" not in seen[0]


def test_cited_sources_without_citations(settings: Settings) -> None:
    """REQ-NFR-11: answers without citations show '(none)' to the judge."""
    from src.core.services import ChatResponse

    resp = ChatResponse(
        thread_id="t",
        request_id="r",
        answer="x",
        agents=[],
        citations=[],
        data={},
        learn_next=[],
        llm_calls=0,
        cost_usd=0,
        latency_ms=0,
    )
    assert cited_sources(fake_service(settings), resp) == "(none)"


def test_judge_failures_counted_and_averaged(settings: Settings) -> None:
    """REQ-NFR-11: averages per kind and overall; failed grades are counted, not averaged."""
    svc = fake_service(settings)
    good = JudgeScore(
        groundedness=4, citation_faithfulness=2, compliance=5, clarity=3, rationale=""
    )
    case = {"q": "What is an index fund?", "kind": "grounded", "key_points": ["x"]}
    ok = run_judge(svc, _judge(good), [case, {**case, "kind": "safety"}])
    failed = run_judge(svc, _judge(ValueError("bad json")), [case])
    avg = averages([*ok, *failed])
    assert avg["all"]["n"] == 3 and avg["all"]["failed"] == 1
    assert avg["grounded"]["citation_faithfulness"] == 2 and avg["safety"]["compliance"] == 5
    assert set(CRITERIA) <= set(avg["all"])
    assert failed[0].error == "ValueError"
    assert averages(failed)["all"]["clarity"] is None  # nothing graded -> no score, not 0


def test_report_sections_replace_in_place(tmp_path: Path) -> None:
    """REQ-NFR-08: regenerated sections replace old ones; hand-written text is kept."""
    md = "# Report\n\nIntro.\n"
    md = replace_section(md, "perf", "old")
    md = replace_section(md, "judge", "J")
    md = replace_section(md, "perf", "new")
    assert md.count("BEGIN perf") == 1 and "new" in md and "old" not in md
    assert md.startswith("# Report\n\nIntro.") and md.index("perf") < md.index("judge")
    path = tmp_path / "docs" / "b.md"
    update_report(path, "perf", "x")
    assert "x" in path.read_text()


def test_table_and_fmt() -> None:
    """REQ-NFR-08: Markdown table rendering."""
    assert table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |"
    assert fmt(None) == "n/a" and fmt(1234.5, ",.0f", " ms") == "1,234 ms"
