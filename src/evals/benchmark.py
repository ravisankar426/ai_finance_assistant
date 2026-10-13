"""Performance benchmark (REQ-NFR-02, -03, -08, -09).

Measures, through the same :class:`AssistantService` the API uses:
- per-agent latency p50/p95, time to first token, LLM calls and cost per query;
- market-data cache hit rate;
- router accuracy on the labeled case set;
- N concurrent sessions (2 turns each): errors and cross-session state leakage.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.core.services import AssistantService, ChatRequest, ChatResponse

# Single-agent queries per agent, plus multi-intent ones. Goal queries carry every input, so
# the agent plans instead of asking a follow-up question.
QUERIES: dict[str, list[str]] = {
    "qa": [
        "What is an index fund?",
        "How does compound interest work?",
        "What does an expense ratio mean?",
        "Why does diversification matter?",
        "How big should an emergency fund be?",
    ],
    "tax": [
        "What's the difference between a Roth IRA and a traditional IRA?",
        "How are long-term capital gains taxed?",
        "What is the wash sale rule?",
        "Can I use my HSA for retirement?",
        "What is the standard deduction?",
    ],
    "market": [
        "What's the current price of AAPL?",
        "How is the S&P 500 doing?",
        "What's the trend for MSFT?",
        "Give me a quote for NVDA",
        "How is AAPL doing this year?",
    ],
    "news": [
        "Any news about Apple?",
        "What are today's top financial headlines?",
        "Any news on Nvidia?",
        "What's the latest on Microsoft?",
        "Any news about Tesla?",
    ],
    "portfolio": [
        "Analyze my portfolio: 10 AAPL, 5 MSFT and 20 VTI",
        "I own 50 NVDA and 10 AAPL. Am I too concentrated?",
        "Analyze 30 VTI, 40 BND and 10 VXUS",
        "How diversified is 100 shares of TSLA?",
        "Review my holdings: 15 JPM, 10 XOM, 25 SPY",
    ],
    "goals": [
        "I want $60,000 for a house down payment in 5 years. I have $10,000 saved and my risk "
        "tolerance is moderate.",
        "I need $30,000 for a car in 3 years, starting from $5,000, conservative risk.",
        "Can I reach $1,000,000 for retirement in 30 years with $20,000 saved? Aggressive risk.",
        "I want $100,000 for college in 15 years, I have nothing saved, moderate risk.",
        "Help me save $20,000 for a wedding in 2 years with $2,000 saved, conservative risk.",
    ],
    "multi": [
        "What's NVDA trading at, and any news on it?",
        "Explain Roth IRAs and show me how AAPL is doing",
    ],
}


@dataclass
class QueryResult:
    """One measured turn."""

    group: str
    query: str
    ok: bool
    agents: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    llm_calls: int = 0
    cost_usd: float = 0.0
    error: str | None = None


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile (p in 0-100); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def run_query(service: AssistantService, group: str, query: str) -> QueryResult:
    """Run one turn on a fresh thread, timing the first token and the full answer."""
    start = time.perf_counter()
    ttft: float | None = None
    final: ChatResponse | None = None
    try:
        for kind, payload in service.chat_events(ChatRequest(message=query)):
            if kind == "token" and ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            elif kind == "final":
                final = payload
    except Exception as exc:  # a benchmark records failures instead of stopping
        return QueryResult(group, query, ok=False, error=f"{type(exc).__name__}: {exc}")
    if final is None:
        return QueryResult(group, query, ok=False, error="stream ended without a final event")
    latency = (time.perf_counter() - start) * 1000
    return QueryResult(
        group,
        query,
        ok=bool(final.answer),
        agents=final.agents,
        latency_ms=latency,
        ttft_ms=ttft if ttft is not None else latency,  # nothing streamed: first output = answer
        llm_calls=final.llm_calls,
        cost_usd=final.cost_usd,
    )


def summarize(results: Sequence[QueryResult]) -> dict[str, dict[str, Any]]:
    """Per-group and overall stats: count, errors, p50/p95 latency and TTFT, mean cost/calls."""
    groups: dict[str, list[QueryResult]] = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)
    groups["all"] = list(results)
    out: dict[str, dict[str, Any]] = {}
    for name, rs in groups.items():
        ok = [r for r in rs if r.ok]
        lat = [r.latency_ms for r in ok]
        ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
        out[name] = {
            "n": len(rs),
            "errors": len(rs) - len(ok),
            "p50_ms": percentile(lat, 50),
            "p95_ms": percentile(lat, 95),
            "ttft_p50_ms": percentile(ttft, 50),
            "ttft_p95_ms": percentile(ttft, 95),
            "mean_cost_usd": sum(r.cost_usd for r in ok) / len(ok) if ok else None,
            "mean_llm_calls": sum(r.llm_calls for r in ok) / len(ok) if ok else None,
        }
    return out


@dataclass
class ConcurrencyReport:
    """Outcome of the concurrent-sessions run (REQ-NFR-09)."""

    sessions: int
    turns: int
    errors: list[str]
    leaks: list[str]
    wall_s: float
    p95_turn_ms: float | None


async def _session(
    service: AssistantService, marker: str, timings: list[float], errors: list[str]
) -> str:
    thread = uuid.uuid4().hex
    for text in (
        f"My study group is called {marker}. What is an index fund?",
        "And how is that different from an ETF?",
    ):
        start = time.perf_counter()
        try:
            async for _ in service.chat_stream(ChatRequest(message=text, thread_id=thread)):
                pass
            timings.append((time.perf_counter() - start) * 1000)
        except Exception as exc:
            errors.append(f"{marker}: {type(exc).__name__}: {exc}")
    return thread


def find_leaks(service: AssistantService, threads: dict[str, str]) -> list[str]:
    """List leaks: a thread whose saved history lacks its marker or mentions another's."""
    leaks = []
    for marker, thread in threads.items():
        state = service.graph.get_state({"configurable": {"thread_id": thread}})
        text = " ".join(str(m.content) for m in state.values.get("messages", []))
        if marker not in text:
            leaks.append(f"{marker}: own marker missing from its thread")
        leaks += [
            f"{marker}: contains {other}" for other in threads if other != marker and other in text
        ]
    return leaks


async def run_concurrent(service: AssistantService, sessions: int = 20) -> ConcurrencyReport:
    """Run ``sessions`` two-turn conversations at once; check errors and isolation."""
    markers = [f"Falcon{i:02d}Q" for i in range(sessions)]
    timings: list[float] = []
    errors: list[str] = []
    start = time.perf_counter()
    threads = await asyncio.gather(*(_session(service, m, timings, errors) for m in markers))
    wall = time.perf_counter() - start
    leaks = find_leaks(service, dict(zip(markers, threads, strict=True)))
    return ConcurrencyReport(
        sessions=sessions,
        turns=2 * sessions,
        errors=errors,
        leaks=leaks,
        wall_s=wall,
        p95_turn_ms=percentile(timings, 95),
    )


def router_accuracy(router: Any, cases: Sequence[dict[str, Any]]) -> tuple[float, list[str]]:
    """Share of labeled cases whose expected intents are all predicted (and none forbidden)."""
    from langchain_core.messages import HumanMessage

    misses = []
    for case in cases:
        state = {"messages": [HumanMessage(case["text"])], "request_id": "bench"}
        intents = set(router(state)["intents"])
        if not set(case["expected"]) <= intents or set(case.get("forbidden", [])) & intents:
            misses.append(f"{case['text']} -> {sorted(intents)} (expected {case['expected']})")
    return 1 - len(misses) / len(cases), misses
