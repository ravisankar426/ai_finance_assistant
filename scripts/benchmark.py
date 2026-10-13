"""Live performance benchmark -> docs/benchmarks.md (REQ-NFR-02, -03, -08, -09).

uv run python -m scripts.benchmark            # full run (~70 LLM-backed turns, about $0.05)
uv run python -m scripts.benchmark --quick    # 1 query per agent, 5 concurrent sessions
"""

from __future__ import annotations

import argparse
import asyncio
import warnings
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scripts.eval_retrieval import CASES as RETRIEVAL_CASES
from scripts.eval_retrieval import evaluate
from src.core.config import PROJECT_ROOT, get_settings
from src.core.llm import get_chat_model
from src.core.services import build_service
from src.evals.benchmark import QUERIES, router_accuracy, run_concurrent, run_query, summarize
from src.evals.report import fmt, table, update_report
from src.utils.logging import configure_logging
from src.workflow.router import RouteDecision, Router

REPORT = PROJECT_ROOT / "docs" / "benchmarks.md"
ROUTER_CASES = PROJECT_ROOT / "tests" / "fixtures" / "router_cases.yaml"
LABELS = {
    "qa": "Finance Q&A",
    "tax": "Tax Education",
    "market": "Market Analysis",
    "news": "News Synthesizer",
    "portfolio": "Portfolio Analysis",
    "goals": "Goal Planning",
    "multi": "Multi-intent (2 agents)",
    "all": "**All queries**",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(level="ERROR", fmt=settings.app.log_format)
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")  # LangChain noise
    service = build_service(settings)
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    roles = settings.llm.roles
    setup = (
        f"_Run {when} · router/guard `{roles['router'].model}`, agents/synthesizer "
        f"`{roles['agent'].model}` · market providers {settings.market.providers} · "
        f"{'quick' if args.quick else 'full'} run._"
    )

    # 1. Per-agent latency and cost (sequential, fresh thread per query). One unmeasured warm-up
    #    query first, so connection setup and lazy imports don't land on the first agent.
    run_query(service, "warmup", "What is a stock?")
    results = []
    for group, queries in QUERIES.items():
        for q in queries[:1] if args.quick else queries:
            r = run_query(service, group, q)
            results.append(r)
            status = "ok " if r.ok else "ERR"
            print(f"[{status}] {group:<9} {r.latency_ms:7.0f} ms  ${r.cost_usd:.5f}  {q[:60]}")
    stats = summarize(results)
    rows = [
        [
            LABELS[g],
            str(s["n"]),
            str(s["errors"]),
            fmt(s["ttft_p50_ms"], ",.0f"),
            fmt(s["ttft_p95_ms"], ",.0f"),
            fmt(s["p50_ms"], ",.0f"),
            fmt(s["p95_ms"], ",.0f"),
            fmt(s["mean_llm_calls"], ".1f"),
            fmt(s["mean_cost_usd"], ".5f", "") if s["mean_cost_usd"] is not None else "n/a",
        ]
        for g, s in stats.items()
    ]
    single = [r for r in results if r.ok and r.group != "multi"]
    single_stats = summarize(single)["all"]
    ttft95, full95 = single_stats["ttft_p95_ms"] or 0, single_stats["p95_ms"] or 0
    cost = stats["all"]["mean_cost_usd"] or 0
    verdict = table(
        ["Target", "Measured", "Met"],
        [
            [
                "p95 time to first token ≤ 3 s (single-agent)",
                f"{ttft95 / 1000:.2f} s",
                _ok(ttft95 <= 3000),
            ],
            [
                "p95 full response ≤ 12 s (single-agent)",
                f"{full95 / 1000:.2f} s",
                _ok(full95 <= 12000),
            ],
            ["Mean LLM cost per query ≤ $0.01", f"${cost:.5f}", _ok(cost <= 0.01)],
        ],
    )
    perf = "\n\n".join(
        [
            setup,
            table(
                [
                    "Agent",
                    "Queries",
                    "Errors",
                    "TTFT p50 (ms)",
                    "TTFT p95 (ms)",
                    "p50 (ms)",
                    "p95 (ms)",
                    "LLM calls",
                    "Cost / query ($)",
                ],
                rows,
            ),
            verdict,
        ]
    )
    update_report(args.report, "performance", perf)

    # 2. Cache hit rate, router accuracy, retrieval recall.
    hit_rate = service.market.cache.hit_rate
    router = Router(get_chat_model("router", structured_output=RouteDecision))
    router_cases = yaml.safe_load(ROUTER_CASES.read_text())
    accuracy, misses = router_accuracy(router, router_cases)
    retrieval = evaluate(service.retriever, yaml.safe_load(RETRIEVAL_CASES.read_text()))
    quality = table(
        ["Metric", "Value", "How measured"],
        [
            [
                "Market-data cache hit rate",
                fmt(hit_rate * 100 if hit_rate is not None else None, ".0f", "%"),
                f"fresh-cache reads during the {len(results)} benchmark queries (TTL: quotes "
                f"{settings.market.ttl_quote_s:.0f}s, "
                f"history {settings.market.ttl_history_s:.0f}s)",
            ],
            [
                "Router accuracy",
                f"{accuracy:.0%} ({len(router_cases) - len(misses)}/{len(router_cases)})",
                "labeled cases in `tests/fixtures/router_cases.yaml`",
            ],
            [
                "Retrieval recall@5",
                f"{retrieval['recall@5']:.3f}",
                "hybrid retriever, `tests/evals/retrieval_set.yaml`",
            ],
            ["Retrieval MRR", f"{retrieval['mrr']:.3f}", "same set"],
            [
                "Off-topic queries rejected",
                f"{retrieval['off_topic_rejected']:.0%}",
                "no passage returned",
            ],
        ],
    )
    if misses:
        quality += "\n\nRouter misses:\n" + "\n".join(f"- {m}" for m in misses)
    update_report(args.report, "quality", quality)

    # 3. Concurrent sessions (async, same process, in-memory checkpointer).
    n = 5 if args.quick else 20
    conc = asyncio.run(run_concurrent(service, sessions=n))
    concurrency = table(
        ["Sessions", "Turns", "Errors", "Leaks", "Wall time", "p95 turn"],
        [
            [
                str(conc.sessions),
                str(conc.turns),
                str(len(conc.errors)),
                str(len(conc.leaks)),
                f"{conc.wall_s:.1f} s",
                fmt(conc.p95_turn_ms, ",.0f", " ms"),
            ]
        ],
    )
    if conc.errors or conc.leaks:
        concurrency += "\n\n" + "\n".join(f"- {e}" for e in [*conc.errors, *conc.leaks])
    update_report(args.report, "concurrency", concurrency)
    print(
        f"\nrouter {accuracy:.0%} · cache hit {hit_rate} · recall@5 {retrieval['recall@5']:.3f} · "
        f"concurrency errors={len(conc.errors)} leaks={len(conc.leaks)} · report: {args.report}"
    )


def _ok(flag: bool) -> str:
    return "✅" if flag else "❌"


if __name__ == "__main__":
    main()
