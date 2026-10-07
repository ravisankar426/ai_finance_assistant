"""Provider adapters against recorded responses (no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from pydantic import SecretStr
from yfinance.exceptions import YFRateLimitError

from src.core.errors import ConfigurationError
from src.data.providers.alphavantage_provider import AlphaVantageProvider
from src.data.providers.base import (
    NoDataError,
    RateLimitedError,
    TransientProviderError,
    UnsupportedOperationError,
)
from src.data.providers.yfinance_provider import YFinanceProvider

FIXTURES = Path(__file__).parents[1] / "fixtures" / "alphavantage"
NOW = datetime(2026, 10, 7, 21, 0, tzinfo=UTC)


# --- Alpha Vantage ---------------------------------------------------------------------------


def _av(responder: Any) -> AlphaVantageProvider:
    client = httpx.Client(transport=httpx.MockTransport(responder))
    return AlphaVantageProvider(SecretStr("test-key"), client=client, clock=lambda: NOW)


def _fixture(name: str, status: int = 200) -> Any:
    body = (FIXTURES / name).read_text()
    return lambda request: httpx.Response(status, text=body)


def test_av_quote_parses_recorded_response() -> None:
    """REQ-MK-01: Alpha Vantage GLOBAL_QUOTE is normalized (price, change %, date, source)."""
    seen: list[httpx.Request] = []

    def responder(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=(FIXTURES / "global_quote_aapl.json").read_text())

    q = _av(responder).get_quote("AAPL")
    assert (q.price, q.previous_close, q.change, q.volume) == (336.67, 333.63, 3.04, 34108778)
    assert q.change_pct == pytest.approx(0.9112)
    assert q.market_date.isoformat() == "2026-10-07"  # type: ignore[union-attr]
    assert q.source == "alphavantage" and q.as_of == NOW
    params = dict(seen[0].url.params)
    assert params["function"] == "GLOBAL_QUOTE" and params["symbol"] == "AAPL"


def test_av_history_sorted_filtered_unadjusted() -> None:
    """REQ-MK-02: daily series sorted oldest-first, limited to the window, flagged unadjusted."""
    h = _av(_fixture("daily_aapl.json")).get_history("AAPL", 30)
    assert [b.day.isoformat() for b in h.bars] == ["2026-10-05", "2026-10-06", "2026-10-07"]
    assert h.closes == [331.0, 333.63, 336.67]
    assert h.adjusted is False


@pytest.mark.parametrize(
    ("fixture", "status", "error"),
    [
        ("global_quote_unknown.json", 200, NoDataError),
        ("rate_limited.json", 200, RateLimitedError),  # throttling arrives as HTTP 200
        ("premium_overview.json", 200, UnsupportedOperationError),
        ("premium_outputsize.json", 200, UnsupportedOperationError),
        ("invalid_call.json", 200, NoDataError),
        ("invalid_apikey.json", 200, ConfigurationError),
    ],
)
def test_av_error_bodies_with_http_200(fixture: str, status: int, error: type[Exception]) -> None:
    """REQ-MD-04: errors hidden inside HTTP 200 bodies are classified correctly."""
    with pytest.raises(error):
        _av(_fixture(fixture, status)).get_quote("AAPL")


@pytest.mark.parametrize(
    ("status", "error"),
    [(429, RateLimitedError), (503, TransientProviderError), (404, NoDataError)],
)
def test_av_http_status_errors(status: int, error: type[Exception]) -> None:
    """REQ-MD-04: HTTP status codes map to retryable vs. permanent errors."""
    with pytest.raises(error):
        _av(lambda r: httpx.Response(status, text="{}")).get_quote("AAPL")


def test_av_network_and_garbage_are_transient() -> None:
    """REQ-MD-04: timeouts and non-JSON responses are retryable."""

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(TransientProviderError):
        _av(timeout).get_quote("AAPL")
    with pytest.raises(TransientProviderError):
        _av(lambda r: httpx.Response(200, text="<html>oops</html>")).get_quote("AAPL")


def test_av_profile_is_premium_only() -> None:
    """REQ-MD-01: OVERVIEW is unsupported on the free tier (verified live)."""
    with pytest.raises(UnsupportedOperationError):
        _av(_fixture("global_quote_aapl.json")).get_profile("AAPL")


def test_av_history_empty_series_is_no_data() -> None:
    """REQ-MD-07: a response without a time series is 'no data'."""
    with pytest.raises(NoDataError):
        _av(lambda r: httpx.Response(200, text=json.dumps({"Meta Data": {}}))).get_history("X", 30)


# --- yfinance --------------------------------------------------------------------------------


class FakeTicker:
    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        info: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.frame = frame if frame is not None else pd.DataFrame()
        self.info_data = info or {}
        self.error = error
        self.history_metadata = {"currency": "USD"}

    def history(self, **kwargs: Any) -> pd.DataFrame:
        if self.error:
            raise self.error
        return self.frame

    @property
    def info(self) -> dict[str, Any]:
        if self.error:
            raise self.error
        return self.info_data


def _frame(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range(end="2026-10-07", periods=len(closes), freq="B", tz="America/New_York")
    return pd.DataFrame({"Close": closes, "Volume": [1000] * len(closes)}, index=idx)


def _yf(ticker: FakeTicker) -> YFinanceProvider:
    return YFinanceProvider(ticker_factory=lambda t: ticker, clock=lambda: NOW)


def test_yf_quote_from_recent_history() -> None:
    """REQ-MK-01: price = latest close, change vs. previous close."""
    q = _yf(FakeTicker(_frame([100.0, 110.0]))).get_quote("AAPL")
    assert (q.price, q.previous_close, q.change) == (110.0, 100.0, 10.0)
    assert q.change_pct == pytest.approx(10.0)
    assert q.market_date.isoformat() == "2026-10-07"  # type: ignore[union-attr]
    assert q.volume == 1000 and q.currency == "USD"


def test_yf_history_adjusted_and_skips_nan() -> None:
    """REQ-MK-02: adjusted closes, NaN rows dropped."""
    h = _yf(FakeTicker(_frame([1.0, float("nan"), 3.0]))).get_history("AAPL", 30)
    assert h.closes == [1.0, 3.0] and h.adjusted is True


def test_yf_empty_is_no_data() -> None:
    """REQ-MD-07: Yahoo's 'empty frame' for unknown tickers becomes NoDataError."""
    with pytest.raises(NoDataError):
        _yf(FakeTicker()).get_quote("ZZQXW")
    with pytest.raises(NoDataError):
        _yf(FakeTicker()).get_history("ZZQXW", 30)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (YFRateLimitError(), RateLimitedError),
        (KeyError("currentTradingPeriod"), NoDataError),
        (ConnectionError("reset"), TransientProviderError),
    ],
)
def test_yf_error_mapping(error: Exception, expected: type[Exception]) -> None:
    """REQ-MD-04: yfinance errors map to retryable vs. permanent categories."""
    with pytest.raises(expected):
        _yf(FakeTicker(error=error)).get_quote("AAPL")


@pytest.mark.parametrize(
    ("info", "asset_type", "sector"),
    [
        (
            {"longName": "Apple Inc.", "quoteType": "EQUITY", "sector": "Technology"},
            "equity",
            "Information Technology",
        ),
        (
            {"shortName": "JNJ", "quoteType": "EQUITY", "sector": "Healthcare"},
            "equity",
            "Health Care",
        ),
        ({"longName": "SPDR S&P 500", "quoteType": "ETF"}, "etf", None),
        (
            {"longName": "Fund", "quoteType": "MUTUALFUND", "sector": "Something New"},
            "fund",
            "Something New",
        ),
    ],
)
def test_yf_profile_maps_to_gics(info: dict[str, Any], asset_type: str, sector: str | None) -> None:
    """REQ-PF-02: Yahoo sector names are mapped to GICS; ETFs have no sector."""
    p = _yf(FakeTicker(info=info)).get_profile("X")
    assert (p.asset_type, p.sector) == (asset_type, sector)


def test_yf_profile_without_name_is_no_data() -> None:
    """REQ-MD-07: an empty info dict means no profile."""
    with pytest.raises(NoDataError):
        _yf(FakeTicker(info={})).get_profile("X")
