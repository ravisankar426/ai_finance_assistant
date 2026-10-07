from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import get_settings
from src.core.errors import KnowledgeBaseError
from src.rag.knowledge_base import load_articles, parse_article

VALID = """---
id: basics-x
title: X
category: basics
difficulty: beginner
source_name: Test
source_url: https://example.com/x
last_reviewed: 2026-10-07
---
# X
Body text.
"""


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_real_knowledge_base_loads() -> None:
    """REQ-RAG-01: every shipped article has valid front-matter and a source URL."""
    articles = load_articles(get_settings().rag.knowledge_base_dir)
    assert len(articles) >= 9
    for a in articles:
        assert a.source_url.startswith("https://")
        assert a.body.startswith("# ")
        assert a.id.startswith(a.category + "-")


def test_parse_valid_article(tmp_path: Path) -> None:
    """REQ-RAG-01: front-matter fields and body are parsed."""
    article = parse_article(_write(tmp_path, "basics/x.md", VALID))
    assert article.title == "X"
    assert article.category == "basics"
    assert article.body == "# X\nBody text."
    assert article.tax_year is None


@pytest.mark.parametrize(
    ("rel", "text", "match"),
    [
        ("basics/a.md", "# no front matter", "front-matter"),
        ("basics/b.md", "---\nid: [unclosed\n---\nbody", "invalid"),
        ("basics/c.md", VALID.replace("difficulty: beginner\n", ""), "invalid"),
        ("investing/d.md", VALID, "doesn't match folder"),
    ],
)
def test_invalid_articles_fail_loudly(tmp_path: Path, rel: str, text: str, match: str) -> None:
    """REQ-RAG-01: a malformed article stops the load with a clear message."""
    with pytest.raises(KnowledgeBaseError, match=match):
        parse_article(_write(tmp_path, rel, text))


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    """REQ-RAG-01: article ids are unique."""
    _write(tmp_path, "basics/one.md", VALID)
    _write(tmp_path, "basics/two.md", VALID)
    with pytest.raises(KnowledgeBaseError, match="Duplicate"):
        load_articles(tmp_path)


def test_missing_folder(tmp_path: Path) -> None:
    """REQ-RAG-01: a missing knowledge-base folder is a clear error."""
    with pytest.raises(KnowledgeBaseError, match="not found"):
        load_articles(tmp_path / "nope")
