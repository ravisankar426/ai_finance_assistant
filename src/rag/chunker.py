"""Split articles into retrievable chunks (design section 9).

Strategy: one chunk per ``## `` section (sections are topical units, so they make coherent
chunks); a section longer than ``max_words`` is split into overlapping word windows. Each
chunk's *embedding text* is prefixed with "Article title — Section", which gives short
chunks the context they need to be found ("Risks" alone means nothing; "Bonds — The main
risks" does).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from src.rag.knowledge_base import Article

_SECTION = re.compile(r"^## +(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of an article."""

    chunk_id: str
    article_id: str
    title: str
    category: str
    url: str
    section: str
    text: str  # what the agent reads and the user may see
    difficulty: str = "beginner"
    tax_year: int | None = None  # set for articles with year-specific figures (REQ-TX-02)

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model and BM25: section text with its context prefix."""
        return f"{self.title} — {self.section}\n{self.text}"

    @property
    def content_hash(self) -> str:
        """Stable hash of the embedding text (embedding-cache key component)."""
        return hashlib.sha256(self.embedding_text.encode("utf-8")).hexdigest()


def _sections(body: str, title: str) -> list[tuple[str, str]]:
    """Return (heading, text) pairs; text before the first ``##`` is the 'Overview'."""
    matches = list(_SECTION.finditer(body))
    intro = body[: matches[0].start()] if matches else body
    intro = re.sub(r"^# .+$", "", intro, count=1, flags=re.MULTILINE).strip()
    sections = [("Overview", intro)] if intro else []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[m.end() : end].strip()
        if text:
            sections.append((m.group(1).strip(), text))
    return sections or [(title, body.strip())]


def _windows(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max(max_words - overlap, 1)
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words) - overlap, step)]


def chunk_article(
    article: Article, *, max_words: int = 250, overlap_words: int = 40
) -> list[Chunk]:
    """Split one article into chunks with stable ids ``<article_id>#<n>``."""
    chunks: list[Chunk] = []
    for heading, text in _sections(article.body, article.title):
        for piece in _windows(text, max_words, overlap_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{article.id}#{len(chunks)}",
                    article_id=article.id,
                    title=article.title,
                    category=article.category,
                    url=article.source_url,
                    section=heading,
                    text=piece,
                    difficulty=article.difficulty,
                    tax_year=article.tax_year,
                )
            )
    return chunks
