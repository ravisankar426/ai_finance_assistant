"""Load knowledge-base articles: Markdown files with YAML front-matter (REQ-RAG-01).

File layout: ``<knowledge_base_dir>/<category>/<slug>.md``::

    ---
    id: investing-index-funds
    title: Index Funds
    category: investing
    difficulty: beginner
    source_name: Investor.gov (U.S. SEC)
    source_url: https://www.investor.gov/...
    last_reviewed: 2026-10-07
    ---
    # Index Funds
    ...
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from src.core.errors import KnowledgeBaseError

Category = Literal["basics", "investing", "portfolio", "retirement", "tax", "markets"]


class Article(BaseModel):
    """One knowledge-base article."""

    id: str
    title: str
    category: Category
    difficulty: Literal["beginner", "intermediate", "advanced"]
    source_name: str
    source_url: str
    last_reviewed: date
    tax_year: int | None = None
    body: str


def parse_article(path: Path) -> Article:
    """Parse one Markdown file with front-matter into an :class:`Article`."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise KnowledgeBaseError(f"{path}: missing YAML front-matter")
    try:
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
        article = Article(**meta, body=body.strip())
    except (ValueError, yaml.YAMLError, ValidationError) as exc:
        raise KnowledgeBaseError(f"{path}: invalid article ({exc})") from exc
    if article.category != path.parent.name:
        raise KnowledgeBaseError(
            f"{path}: category '{article.category}' doesn't match folder '{path.parent.name}'"
        )
    return article


def load_articles(root: Path) -> list[Article]:
    """Load every article under ``root``; fail on duplicate ids."""
    if not root.is_dir():
        raise KnowledgeBaseError(f"Knowledge base folder not found: {root}")
    articles = [parse_article(p) for p in sorted(root.glob("*/*.md"))]
    ids = [a.id for a in articles]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise KnowledgeBaseError(f"Duplicate article ids: {duplicates}")
    return articles
