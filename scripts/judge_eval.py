"""Opt-in LLM-as-judge eval on Gemini -> docs/benchmarks.md (REQ-NFR-11, REQ-LLM-06).

uv run python -m scripts.judge_eval            # 30 golden questions (~$0.03 OpenAI + Gemini)
uv run python -m scripts.judge_eval --limit 5  # smoke run
"""

from __future__ import annotations

import argparse
import warnings
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.core.config import PROJECT_ROOT, get_settings
from src.core.llm import get_chat_model
from src.core.services import build_service
from src.evals.judge import CRITERIA, JudgeScore, averages, run_judge
from src.evals.report import fmt, table, update_report
from src.utils.logging import configure_logging

REPORT = PROJECT_ROOT / "docs" / "benchmarks.md"
GOLDEN = PROJECT_ROOT / "tests" / "evals" / "golden_qa.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between cases")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(level="ERROR", fmt=settings.app.log_format)
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")  # LangChain noise
    cases = yaml.safe_load(GOLDEN.read_text())[: args.limit]
    service = build_service(settings)
    judge = get_chat_model("judge", structured_output=JudgeScore)
    results = run_judge(service, judge, cases, pause_s=args.pause)
    for r in results:
        s = r.score
        marks = " ".join(f"{getattr(s, c)}" for c in CRITERIA) if s else f"FAILED ({r.error})"
        print(f"{r.kind:<9} {marks:<12} {r.question[:70]}")

    avg = averages(results)
    failed, total = int(avg["all"]["failed"] or 0), len(results)
    if failed > total // 5:  # don't publish averages built on a fraction of the set
        errors = sorted({r.error for r in results if r.error})
        raise SystemExit(f"{failed}/{total} grades failed ({errors}); report not updated.")
    judge_cfg = settings.llm.roles["judge"]
    rows = [
        [kind, str(row["n"]), str(row["failed"]), *(fmt(row[c], ".2f") for c in CRITERIA)]
        for kind, row in avg.items()
    ]
    lowest = sorted(
        (r for r in results if r.score),
        key=lambda r: min(getattr(r.score, c) for c in CRITERIA),
    )[:5]
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    body = "\n\n".join(
        [
            f"_Run {when} · judge `{judge_cfg.provider}:{judge_cfg.model}` · answers by "
            f"`{settings.llm.roles['agent'].model}` · {len(results)} golden questions "
            "(`tests/evals/golden_qa.yaml`). Scores 1 to 5._",
            table(["Kind", "Cases", "Failed", *(c.replace("_", " ") for c in CRITERIA)], rows),
            "Lowest-scoring answers (judge's rationale):\n"
            + "\n".join(
                f"- *{r.question}* — {r.score.rationale}" for r in lowest if r.score is not None
            ),
        ]
    )
    update_report(args.report, "judge", body)
    print(
        "\noverall:",
        {c: round(float(avg["all"][c]), 2) for c in CRITERIA},
        "· report:",
        args.report,
    )


if __name__ == "__main__":
    main()
