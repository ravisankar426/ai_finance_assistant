"""Live check of which models your keys can use and whether every configured role answers.

Usage:  uv run python scripts/check_models.py            # smoke-test every role + embeddings
        uv run python scripts/check_models.py --list     # also list chat models per provider

Costs a fraction of a cent. Requires OPENAI_API_KEY and GOOGLE_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import time

from src.core.config import ALL_ROLES, get_settings
from src.core.llm import _build_model, get_chat_model, get_embeddings, track_usage
from src.utils.logging import configure_logging


def list_models() -> None:
    """Print chat-capable model ids visible to each key (scripts may import vendor SDKs)."""
    settings = get_settings()
    if settings.openai_api_key:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        ids = sorted(m.id for m in client.models.list() if m.id.startswith(("gpt-", "o")))
        print("OpenAI models:", ", ".join(ids))
    if settings.google_api_key:
        from google import genai

        gclient = genai.Client(api_key=settings.google_api_key.get_secret_value())
        names = sorted(
            (m.name or "").removeprefix("models/")
            for m in gclient.models.list()
            if "gemini" in (m.name or "")
        )
        print("Gemini models:", ", ".join(names))


def smoke_test() -> int:
    """Call every role once; return the number of failures."""
    failures = 0
    for role in ALL_ROLES:
        cfg = get_settings().llm.roles[role]
        start = time.perf_counter()
        try:
            with track_usage() as usage:
                reply = get_chat_model(role).invoke("Reply with exactly: OK")
            ms = (time.perf_counter() - start) * 1000
            print(
                f"[ok]   {role:<12} {cfg.provider}:{cfg.model:<22} {ms:6.0f} ms  "
                f"tokens={usage.input_tokens}+{usage.output_tokens}  "
                f"cost=${usage.cost_usd:.6f}  fallback={usage.fallbacks}  reply={reply.text!r}"
            )
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {role:<12} {cfg.provider}:{cfg.model:<22} {type(exc).__name__}: {exc}")

    fb = get_settings().llm.fallback
    if fb:  # call the fallback directly: in normal use it only runs when the primary fails
        try:
            model = _build_model(fb, role="fallback-check", settings=get_settings())
            reply = model.invoke("Reply with exactly: OK")
            print(f"[ok]   fallback     {fb.provider}:{fb.model:<22} reply={reply.text!r}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] fallback     {fb.provider}:{fb.model:<22} {type(exc).__name__}: {exc}")

    try:
        vec = get_embeddings().embed_query("index fund")
        print(f"[ok]   embeddings   dim={len(vec)}")
    except Exception as exc:
        failures += 1
        print(f"[FAIL] embeddings   {type(exc).__name__}: {exc}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list available models per provider")
    args = parser.parse_args()
    configure_logging(level="WARNING")
    if args.list:
        list_models()
    raise SystemExit(1 if smoke_test() else 0)


if __name__ == "__main__":
    main()
