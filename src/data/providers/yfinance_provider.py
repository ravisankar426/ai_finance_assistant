"""yfinance provider (primary; free, no key, unofficial — ADR-08).

Behaviors observed live (2026-10-07) that this adapter normalizes:
- an unknown ticker returns an *empty* history (and ``fast_info`` raises ``KeyError``), so
  "empty" maps to ``NoDataError`` — Yahoo outages look the same, so the service asks the next
  provider before calling a ticker invalid;
- Yahoo sector names differ from GICS ("Technology" vs "Information Technology"), so sectors
  are mapped to GICS to match the knowledge base and portfolio analysis;
- ETFs have no sector.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import yfinance as yf
from yfinance.exceptions import (
    YFPricesMissingError,
    YFRateLimitError,
    YFTickerMissingError,
)

from src.data.models import (
    AssetProfile,
    AssetType,
    NewsFeed,
    NewsItem,
    PriceBar,
    PriceHistory,
    Quote,
)
from src.data.providers.base import NoDataError, RateLimitedError, TransientProviderError

YAHOO_TO_GICS: dict[str, str] = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}

_QUOTE_TYPES: dict[str, AssetType] = {
    "EQUITY": "equity",
    "ETF": "etf",
    "MUTUALFUND": "fund",
    "INDEX": "index",
}


def _now() -> datetime:
    return datetime.now(UTC)


class YFinanceProvider:
    """Market data from Yahoo Finance via the ``yfinance`` package."""

    name = "yfinance"

    def __init__(
        self,
        *,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        search_factory: Callable[..., Any] = yf.Search,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.ticker_factory = ticker_factory
        self.search_factory = search_factory
        self.clock = clock

    def _call(self, fn: Callable[[], Any]) -> Any:
        """Run a yfinance call, translating its errors into provider error categories."""
        try:
            return fn()
        except YFRateLimitError as exc:
            raise RateLimitedError(str(exc)) from exc
        except (YFTickerMissingError, YFPricesMissingError, KeyError, IndexError) as exc:
            raise NoDataError(str(exc)) from exc
        except Exception as exc:  # network, timeouts, Yahoo-side changes
            raise TransientProviderError(f"{type(exc).__name__}: {exc}") from exc

    def get_quote(self, ticker: str) -> Quote:
        """Latest close (live price during market hours) vs. the previous close."""
        t = self.ticker_factory(ticker)
        hist = self._call(lambda: t.history(period="5d", interval="1d", auto_adjust=True))
        if hist is None or hist.empty:
            raise NoDataError(f"no recent prices for {ticker}")
        last = hist.iloc[-1]
        price = float(last["Close"])
        prev = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else None
        change = price - prev if prev else 0.0
        metadata = getattr(t, "history_metadata", None) or {}
        return Quote(
            ticker=ticker,
            price=round(price, 4),
            previous_close=round(prev, 4) if prev else None,
            change=round(change, 4),
            change_pct=round(change / prev * 100, 4) if prev else 0.0,
            volume=int(last["Volume"]) if last.get("Volume") == last.get("Volume") else None,
            currency=str(metadata.get("currency") or "USD"),
            market_date=hist.index[-1].date(),
            as_of=self.clock(),
            source=self.name,
        )

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        """Split/dividend-adjusted daily closes for the last ``days`` calendar days."""
        t = self.ticker_factory(ticker)
        start: date = self.clock().date() - timedelta(days=days)
        hist = self._call(
            lambda: t.history(start=start.isoformat(), interval="1d", auto_adjust=True)
        )
        if hist is None or hist.empty:
            raise NoDataError(f"no price history for {ticker}")
        bars = [
            PriceBar(
                day=idx.date(),
                close=round(float(row["Close"]), 4),
                volume=int(row["Volume"]) if row["Volume"] == row["Volume"] else None,
            )
            for idx, row in hist.iterrows()
            if row["Close"] == row["Close"]  # skip NaN rows
        ]
        return PriceHistory(
            ticker=ticker, bars=bars, adjusted=True, as_of=self.clock(), source=self.name
        )

    def get_profile(self, ticker: str) -> AssetProfile:
        """Company/fund name, asset type, and GICS-mapped sector."""
        t = self.ticker_factory(ticker)
        info: dict[str, Any] = self._call(lambda: t.info) or {}
        name = info.get("longName") or info.get("shortName")
        if not name:
            raise NoDataError(f"no profile for {ticker}")
        sector = info.get("sector")
        return AssetProfile(
            ticker=ticker,
            name=str(name),
            asset_type=_QUOTE_TYPES.get(str(info.get("quoteType", "")).upper(), "other"),
            sector=YAHOO_TO_GICS.get(sector, sector) if sector else None,
            industry=info.get("industry"),
            currency=str(info.get("currency") or "USD"),
            as_of=self.clock(),
            source=self.name,
        )

    def get_news(self, subject: str, limit: int) -> NewsFeed:
        """Headlines via Yahoo search (``Ticker.news`` returned nothing when probed 2026-10-07)."""
        search = self._call(lambda: self.search_factory(subject, news_count=limit, max_results=0))
        raw: list[dict[str, Any]] = getattr(search, "news", None) or []
        items = [
            NewsItem(
                title=str(n["title"]),
                url=str(n["link"]),
                publisher=str(n.get("publisher") or "Unknown"),
                published_at=datetime.fromtimestamp(int(n["providerPublishTime"]), UTC),
                related_tickers=[str(t) for t in n.get("relatedTickers") or []],
            )
            for n in raw
            if n.get("title") and n.get("link") and n.get("providerPublishTime")
        ]
        return NewsFeed(subject=subject, items=items, as_of=self.clock(), source=self.name)
