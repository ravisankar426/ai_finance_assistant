"""Provider interface and the error categories the market service reasons about.

The service's behavior depends only on *which kind* of error a provider raises:

| Error                       | Retried? | Counts toward breaker? | Next provider? |
|-----------------------------|----------|------------------------|----------------|
| TransientProviderError      | yes      | yes (after retries)    | yes            |
| RateLimitedError            | yes      | yes (after retries)    | yes            |
| NoDataError                 | no       | no                     | yes            |
| UnsupportedOperationError   | no       | no                     | yes            |
| InvalidTickerError (format) | —        | —                      | no: raised before any call |
| ConfigurationError          | no       | no                     | no: raised (fail loud)     |
"""

from __future__ import annotations

import re
from typing import Protocol

from src.core.errors import InvalidTickerError
from src.data.models import AssetProfile, PriceHistory, Quote


class ProviderError(Exception):
    """Base class for provider failures."""


class TransientProviderError(ProviderError):
    """Network trouble, timeouts, 5xx — worth retrying."""


class RateLimitedError(TransientProviderError):
    """The provider throttled us — worth retrying after a pause."""


class NoDataError(ProviderError):
    """Raised when a provider has no data for the ticker (it may exist elsewhere)."""


class UnsupportedOperationError(ProviderError):
    """Raised when a provider can't do an operation (e.g. a premium-only endpoint)."""


class MarketDataProvider(Protocol):
    """One source of market data, returning normalized models."""

    name: str

    def get_quote(self, ticker: str) -> Quote:
        """Latest price snapshot."""
        ...

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        """Daily closes covering roughly the last ``days`` calendar days."""
        ...

    def get_profile(self, ticker: str) -> AssetProfile:
        """Name, asset type, and sector."""
        ...


_TICKER = re.compile(r"^[A-Z]{1,5}(?:-[A-Z]{1,2})?$")


def normalize_ticker(raw: str) -> str:
    """Validate and normalize a ticker: '$aapl ' -> 'AAPL', 'brk.b' -> 'BRK-B' (REQ-MD-07)."""
    ticker = raw.strip().lstrip("$").upper().replace(".", "-")
    if not _TICKER.match(ticker):
        raise InvalidTickerError(
            f"invalid ticker format: {raw!r}",
            user_message=f"“{raw.strip()}” doesn't look like a valid US ticker symbol.",
        )
    return ticker
