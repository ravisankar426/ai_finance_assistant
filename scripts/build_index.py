"""Rebuild the knowledge-base index from scratch (embedding cache still avoids re-paying).

uv run python scripts/build_index.py              # always rebuild
uv run python scripts/build_index.py --if-needed  # reuse a matching index (docker-compose)
"""

from __future__ import annotations

import argparse
from collections import Counter

from src.core.config import get_settings
from src.core.llm import get_embeddings
from src.rag.index import build_index, load_or_build
from src.rag.knowledge_base import load_articles
from src.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-needed", action="store_true", help="reuse a matching index")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(level=settings.app.log_level, fmt=settings.app.log_format)
    articles = load_articles(settings.rag.knowledge_base_dir)
    build = load_or_build if args.if_needed else build_index
    store = build(articles, get_embeddings(settings), settings.llm.embeddings.model, settings.rag)
    print(f"{len(articles)} articles -> {len(store.chunks)} chunks ({store.dimension}-dim)")
    print("by category:", dict(Counter(a.category for a in articles)))
    print("saved to:", settings.rag.index_dir)


if __name__ == "__main__":
    main()
