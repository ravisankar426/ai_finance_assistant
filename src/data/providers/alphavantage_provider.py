"""Alpha Vantage provider (fallback; official API, small free tier — ADR-08).

Free-tier behaviors observed live (2026-10-07) that this adapter normalizes:
- throttling comes back as **HTTP 200** with an ``"Information"`` (or ``"Note"``) body:
  "...1 request per second... 25 requests per day" -> ``RateLimitedError``;
- premium-only features also come back as HTTP 200 + ``"Information"`` mentioning "premium"
  (``OVERVIEW``, ``outputsize=full``) -> ``UnsupportedOperationError``;
- an unknown symbol returns ``{"Global Quote": {}}`` -> ``NoDataError``;
- daily history is limited to the last 100 trading days and is not split-adjusted.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from pydantic import SecretStr

from src.core.errors import ConfigurationError
from src.data.models import AssetProfile, PriceBar, PriceHistory, Quote
from src.data.providers.base import (
    NoDataError,
    RateLimitedError,
    TransientProviderError,
    UnsupportedOperationError,
)

BASE_URL = "https://www.alphavantage.co/query"


def _now() -> datetime:
    return datetime.now(UTC)


def _num(value: Any) -> float:
    return float(str(value).rstrip("%"))


class AlphaVantageProvider:
    """Market data from the Alpha Vantage REST API."""

    name = "alphavantage"

    def __init__(
        self,
        api_key: SecretStr,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 10,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._api_key = api_key
        self.client = client or httpx.Client(timeout=timeout_s)
        self.clock = clock

    def _get(self, **params: str) -> dict[str, Any]:
        """GET the API and classify every failure mode, including errors sent with HTTP 200."""
        try:
            resp = self.client.get(
                BASE_URL, params={**params, "apikey": self._api_key.get_secret_value()}
            )
        except httpx.HTTPError as exc:  # connection errors, timeouts
            raise TransientProviderError(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code == 429:
            raise RateLimitedError("HTTP 429")
        if resp.status_code >= 500:
            raise TransientProviderError(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise NoDataError(f"HTTP {resp.status_code}")
        try:
            body: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise TransientProviderError("non-JSON response") from exc
        notice = str(body.get("Information") or body.get("Note") or "")
        if notice:
            text = notice.lower()
            # Order matters: throttling messages ALSO mention "premium plans", so check the
            # rate-limit wording first; only specific phrases mean "premium-only feature".
            if any(s in text for s in ("per second", "per day", "rate limit", "spreading out")):
                raise RateLimitedError(notice[:160])
            if "premium endpoint" in text or "premium feature" in text:
                raise UnsupportedOperationError(notice[:160])
            raise RateLimitedError(notice[:160])  # unknown notice: assume transient, retry
        error = str(body.get("Error Message") or "")
        if error:
            if "apikey" in error.lower():
                raise ConfigurationError(f"Alpha Vantage rejected the API key: {error[:120]}")
            raise NoDataError(error[:160])
        return body

    def get_quote(self, ticker: str) -> Quote:
        """GLOBAL_QUOTE -> normalized quote."""
        q = self._get(function="GLOBAL_QUOTE", symbol=ticker).get("Global Quote") or {}
        if not q:
            raise NoDataError(f"no quote for {ticker}")
        return Quote(
            ticker=ticker,
            price=_num(q["05. price"]),
            previous_close=_num(q["08. previous close"]),
            change=_num(q["09. change"]),
            change_pct=_num(q["10. change percent"]),
            volume=int(q["06. volume"]),
            market_date=date.fromisoformat(q["07. latest trading day"]),
            as_of=self.clock(),
            source=self.name,
        )

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        """TIME_SERIES_DAILY (compact = last 100 trading days; unadjusted on the free tier)."""
        body = self._get(function="TIME_SERIES_DAILY", symbol=ticker, outputsize="compact")
        series: dict[str, dict[str, str]] = body.get("Time Series (Daily)") or {}
        if not series:
            raise NoDataError(f"no price history for {ticker}")
        cutoff = self.clock().date() - timedelta(days=days)
        bars = sorted(
            (
                PriceBar(
                    day=date.fromisoformat(d), close=_num(v["4. close"]), volume=int(v["5. volume"])
                )
                for d, v in series.items()
                if date.fromisoformat(d) >= cutoff
            ),
            key=lambda b: b.day,
        )
        return PriceHistory(
            ticker=ticker, bars=bars, adjusted=False, as_of=self.clock(), source=self.name
        )

    def get_profile(self, ticker: str) -> AssetProfile:
        """OVERVIEW is premium-only on the free key (verified live), so this always declines."""
        raise UnsupportedOperationError("Alpha Vantage OVERVIEW requires a premium key")
