"""Normalized market-data models: every provider returns these shapes (REQ-MD-01)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

AssetType = Literal["equity", "etf", "fund", "index", "other"]


class Quote(BaseModel):
    """Latest price snapshot (REQ-MK-01)."""

    ticker: str
    price: float
    previous_close: float | None = None
    change: float = 0.0
    change_pct: float = 0.0
    volume: int | None = None
    currency: str = "USD"
    market_date: date | None = None  # trading day the price belongs to
    as_of: datetime  # when we fetched it (UTC) — drives freshness (REQ-MK-04)
    source: str
    stale: bool = False


class PriceBar(BaseModel):
    """One daily bar."""

    day: date
    close: float
    volume: int | None = None


class PriceHistory(BaseModel):
    """Daily closing prices, oldest first."""

    ticker: str
    bars: list[PriceBar] = Field(default_factory=list)
    adjusted: bool = True  # adjusted for splits/dividends (yfinance) or raw (Alpha Vantage free)
    as_of: datetime
    source: str
    stale: bool = False

    @property
    def closes(self) -> list[float]:
        """Closing prices, oldest first."""
        return [b.close for b in self.bars]


class AssetProfile(BaseModel):
    """What a ticker is: name, type, and GICS sector (for portfolio allocation)."""

    ticker: str
    name: str
    asset_type: AssetType = "other"
    sector: str | None = None  # GICS sector name; None for ETFs/funds
    industry: str | None = None
    currency: str = "USD"
    as_of: datetime
    source: str
    stale: bool = False


class NewsItem(BaseModel):
    """One headline (providers on the free tier give headlines, not article text)."""

    title: str
    url: str
    publisher: str
    published_at: datetime
    related_tickers: list[str] = Field(default_factory=list)


class NewsFeed(BaseModel):
    """Recent headlines for a ticker or a market topic."""

    subject: str  # ticker (e.g. "AAPL") or topic (e.g. "stock market")
    items: list[NewsItem] = Field(default_factory=list)
    as_of: datetime
    source: str
    stale: bool = False
