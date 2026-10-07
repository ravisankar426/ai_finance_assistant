"""Chat with the assistant in the terminal, using the real models (needs keys in .env).

uv run python scripts/chat_cli.py                       # interactive
uv run python scripts/chat_cli.py -q "What is an ETF?" -q "and its fees?"   # scripted turns
"""

from __future__ import annotations

import argparse
import time
import uuid

from langchain_core.messages import HumanMessage

from src.core.llm import track_usage
from src.utils.logging import configure_logging, request_context
from src.workflow.graph import build_default_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--question", action="append", help="question (repeat for turns)")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(level=args.log_level)

    graph = build_default_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    questions = iter(args.question) if args.question else None
    while True:
        question = next(questions, None) if questions else input("\nyou> ").strip()
        if not question or question in {"exit", "quit"}:
            break
        if questions:
            print(f"\nyou> {question}")
        start = time.perf_counter()
        with request_context(), track_usage() as usage:
            out = graph.invoke({"messages": [HumanMessage(question)]}, config)
        print(f"\nassistant> {out['final_answer']}")
        for c in out.get("citations", []):
            print(f"   source: {c.title} — {c.url}")
        print(
            f"   [{', '.join(out['intents'])}] {time.perf_counter() - start:.1f}s, "
            f"{usage.calls} LLM calls, ${usage.cost_usd:.5f}"
        )


if __name__ == "__main__":
    main()
