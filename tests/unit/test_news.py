"""Headline filtering/ranking/dedupe, yfinance news mapping, and fixture-backed news."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.config import get_settings
from src.core.news import dedupe, jaccard, select_headlines
from src.data.models import NewsItem
from src.data.providers.fixture_provider import FixtureProvider
from src.data.providers.yfinance_provider import YFinanceProvider

NOW = datetime(2026, 10, 7, 22, 0, tzinfo=UTC)


def _item(
    title: str, *, hours_ago: float = 1, tickers: list[str] | None = None, url: str | None = None
) -> NewsItem:
    return NewsItem(
        title=title,
        url=url or f"https://news.example/{abs(hash(title))}",
        publisher="Wire",
        published_at=NOW - timedelta(hours=hours_ago),
        related_tickers=tickers or [],
    )


def test_jaccard() -> None:
    """REQ-NW-02: word-set similarity used for near-duplicate detection."""
    assert jaccard({"a", "b"}, {"a", "b"}) == 1
    assert jaccard({"a"}, {"b"}) == 0
    assert jaccard(set(), set()) == 1


def test_dedupe_removes_same_url_and_syndicated_titles() -> None:
    """REQ-NW-02: the same story from several outlets appears once."""
    items = [
        _item("Apple unveils new smart home hub", url="https://a/1"),
        _item("Apple unveils new smart home hub", url="https://b/2"),  # same title elsewhere
        _item("Apple Unveils New Smart-Home Hub!", url="https://c/3"),  # punctuation/case only
        _item("Different story entirely", url="https://a/1"),  # same URL
        _item("Microsoft earnings beat estimates", url="https://d/4"),
    ]
    assert [i.url for i in dedupe(items)] == ["https://a/1", "https://d/4"]


def test_select_filters_lookback_and_relevance_and_ranks_focus() -> None:
    """REQ-NW-01: old and unrelated headlines dropped; focused stories outrank roundups."""
    items = [
        _item("Dow futures roundup", tickers=["AAPL", "MSFT", "NVDA", "^DJI"], hours_ago=1),
        _item("Apple-only story", tickers=["AAPL"], hours_ago=5),
        _item("Old Apple story", tickers=["AAPL"], hours_ago=24 * 10),
        _item("Nvidia story", tickers=["NVDA"], hours_ago=1),
    ]
    picked = select_headlines(items, ticker="AAPL", now=NOW, lookback_days=7, limit=5)
    assert [i.title for i in picked] == ["Apple-only story", "Dow futures roundup"]


def test_select_topic_orders_by_recency_and_limits() -> None:
    """REQ-NW-01: market-topic news keeps everything recent, newest first."""
    items = [_item(f"Story {n}", hours_ago=n) for n in (3, 1, 2)]
    picked = select_headlines(items, ticker=None, now=NOW, lookback_days=7, limit=2)
    assert [i.title for i in picked] == ["Story 1", "Story 2"]


class FakeSearch:
    def __init__(self, news: list[dict[str, Any]]) -> None:
        self.news = news


def test_yfinance_news_mapping_skips_incomplete_items() -> None:
    """REQ-NW-01: Yahoo search results become NewsItems; malformed entries are skipped."""
    raw = [
        {
            "title": "T1",
            "link": "https://x/1",
            "publisher": "P",
            "providerPublishTime": 1791412200,
            "relatedTickers": ["AAPL"],
        },
        {"title": "no link", "providerPublishTime": 1791412200},
        {"title": "T2", "link": "https://x/2", "providerPublishTime": 1791412200},
    ]
    provider = YFinanceProvider(search_factory=lambda *a, **k: FakeSearch(raw), clock=lambda: NOW)
    feed = provider.get_news("AAPL", 10)
    assert [i.title for i in feed.items] == ["T1", "T2"]
    assert feed.items[0].related_tickers == ["AAPL"] and feed.items[1].publisher == "Unknown"
    assert feed.items[0].published_at.tzinfo is not None


def test_recorded_news_fixtures_load() -> None:
    """REQ-DEL-02: recorded headlines ship with the repo and are labeled as recorded."""
    feed = FixtureProvider(get_settings().market.fixture_dir).get_news("AAPL", 5)
    assert 1 <= len(feed.items) <= 5
    assert feed.source.startswith("recorded ")
    market = FixtureProvider(get_settings().market.fixture_dir).get_news("stock market", 10)
    assert market.items
