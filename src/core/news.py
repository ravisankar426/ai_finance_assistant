"""Deterministic headline processing for the News agent (REQ-NW-01, REQ-NW-02).

Free-tier sources return headlines only, often tagged with many tickers (a "Dow futures"
roundup was the first "AAPL" result when probed). So, in code before any LLM call:
1. drop headlines older than the lookback window;
2. for ticker requests, keep only headlines that list the ticker, and rank focused stories
   (few related tickers) above roundups;
3. remove duplicates: same URL, same normalized title, or near-identical titles
   (word-set Jaccard similarity >= 0.8) syndicated by different outlets.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta

from src.data.models import NewsItem

_WORD = re.compile(r"[a-z0-9]+")


def _words(title: str) -> set[str]:
    return set(_WORD.findall(title.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Word-set overlap in [0, 1]."""
    return len(a & b) / len(a | b) if a or b else 1.0


def dedupe(items: Sequence[NewsItem], threshold: float = 0.8) -> list[NewsItem]:
    """Keep the first of any duplicate group (callers sort by priority first)."""
    kept: list[NewsItem] = []
    seen_urls: set[str] = set()
    kept_words: list[set[str]] = []
    for item in items:
        words = _words(item.title)
        if item.url in seen_urls or any(jaccard(words, w) >= threshold for w in kept_words):
            continue
        seen_urls.add(item.url)
        kept_words.append(words)
        kept.append(item)
    return kept


def select_headlines(
    items: Sequence[NewsItem],
    *,
    ticker: str | None,
    now: datetime,
    lookback_days: int,
    limit: int,
) -> list[NewsItem]:
    """Filter, rank, and deduplicate headlines for one subject."""
    cutoff = now - timedelta(days=lookback_days)
    recent = [i for i in items if i.published_at >= cutoff]
    if ticker:
        recent = [i for i in recent if ticker in i.related_tickers]
        # Focused stories first (fewer tagged tickers), then newest.
        recent.sort(key=lambda i: (len(i.related_tickers), -i.published_at.timestamp()))
    else:
        recent.sort(key=lambda i: -i.published_at.timestamp())
    return dedupe(recent)[:limit]
