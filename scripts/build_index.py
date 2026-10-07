"""Rebuild the knowledge-base index from scratch (embedding cache still avoids re-paying).

uv run python scripts/build_index.py
"""

from __future__ import annotations

from collections import Counter

from src.core.config import get_settings
from src.core.llm import get_embeddings
from src.rag.index import build_index
from src.rag.knowledge_base import load_articles
from src.utils.logging import configure_logging


def main() -> None:
    configure_logging(level="INFO")
    settings = get_settings()
    articles = load_articles(settings.rag.knowledge_base_dir)
    store = build_index(
        articles, get_embeddings(settings), settings.llm.embeddings.model, settings.rag
    )
    print(f"{len(articles)} articles -> {len(store.chunks)} chunks ({store.dimension}-dim)")
    print("by category:", dict(Counter(a.category for a in articles)))
    print("saved to:", settings.rag.index_dir)


if __name__ == "__main__":
    main()
