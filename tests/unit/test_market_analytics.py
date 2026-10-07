"""Deterministic market analytics — example-based and property-based tests."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.market_analytics import (
    annualized_volatility,
    classify_trend,
    describe_snapshot,
    fmt_pct,
    period_return,
    simple_moving_average,
    trend_snapshot,
)
from src.data.models import PriceBar, PriceHistory

prices = st.lists(
    st.floats(min_value=1, max_value=10_000, allow_nan=False), min_size=1, max_size=400
)


def _history(closes: list[float], *, adjusted: bool = True, stale: bool = False) -> PriceHistory:
    start = date(2025, 1, 1)
    bars = [PriceBar(day=start + timedelta(days=i), close=c) for i, c in enumerate(closes)]
    return PriceHistory(
        ticker="TEST",
        bars=bars,
        adjusted=adjusted,
        as_of=datetime(2026, 10, 7, tzinfo=UTC),
        source="unit",
        stale=stale,
    )


# --- examples ----------------------------------------------------------------------------------


def test_sma_and_returns_examples() -> None:
    """REQ-MK-02: SMA = mean of the last n closes; return = last / start - 1."""
    closes = [10.0, 11.0, 12.0, 13.0]
    assert simple_moving_average(closes, 2) == 12.5
    assert simple_moving_average(closes, 5) is None
    assert period_return(closes, 3) == pytest.approx(0.3)
    assert period_return(closes, 4) is None  # needs a start price 4 days back


def test_volatility_of_constant_growth_is_zero_and_needs_a_month() -> None:
    """REQ-MK-02: steady compounding has zero volatility; <21 days isn't estimated."""
    steady = [100 * 1.001**i for i in range(60)]
    assert annualized_volatility(steady) == pytest.approx(0, abs=1e-9)
    assert annualized_volatility(steady[:10]) is None


def test_volatility_known_value() -> None:
    """REQ-MK-02: alternating ±1% log moves -> daily sd ~1%, annualized ~sqrt(252)%."""
    closes = [100.0]
    for i in range(100):
        closes.append(closes[-1] * math.exp(0.01 if i % 2 == 0 else -0.01))
    vol = annualized_volatility(closes)
    assert vol == pytest.approx(0.01 * math.sqrt(252), rel=0.02)


@pytest.mark.parametrize(
    ("price", "sma50", "sma200", "expected"),
    [
        (110, 100, 90, "uptrend"),
        (80, 90, 100, "downtrend"),
        (95, 100, 90, "mixed"),
        (100, None, 90, "insufficient data"),
    ],
)
def test_classify_trend(
    price: float, sma50: float | None, sma200: float | None, expected: str
) -> None:
    """REQ-MK-02: uptrend = price > 50-day > 200-day; downtrend the reverse."""
    assert classify_trend(price, sma50, sma200) == expected


def test_snapshot_on_rising_series() -> None:
    """REQ-MK-02: a steadily rising 300-day series is an uptrend at its 52-week high."""
    snap = trend_snapshot(_history([100 + i for i in range(300)]))
    assert snap.trend == "uptrend"
    assert snap.high_52w == snap.last_close == 399
    assert snap.pct_below_52w_high == 0
    assert snap.return_1y == pytest.approx(399 / 147 - 1)
    assert snap.sma_200 is not None and snap.pct_vs_sma_200 is not None and snap.pct_vs_sma_200 > 0


def test_snapshot_short_history_marks_missing_values() -> None:
    """REQ-MK-02, REQ-MD-01: a 100-day fallback history has no 200-day SMA or 1y return."""
    snap = trend_snapshot(_history([float(x) for x in range(1, 101)], adjusted=False, stale=True))
    assert snap.sma_200 is None and snap.return_1y is None
    assert snap.trend == "insufficient data"
    text = describe_snapshot(snap, "Test Corp")
    assert "UNADJUSTED" in text and "STALE" in text and "n/a" in text and "Test Corp" in text


def test_snapshot_rejects_empty_history() -> None:
    """REQ-MK-02: no prices is a programming error, not a silent zero."""
    with pytest.raises(ValueError, match="no prices"):
        trend_snapshot(_history([]))


def test_fmt_pct() -> None:
    """REQ-MK-02: consistent percent formatting for the fact block."""
    assert fmt_pct(0.0512) == "+5.1%"
    assert fmt_pct(-0.2) == "-20.0%"
    assert fmt_pct(0.25, signed=False) == "25.0%"
    assert fmt_pct(None) == "n/a"


# --- properties --------------------------------------------------------------------------------


@given(prices, st.integers(min_value=1, max_value=250))
def test_sma_lies_within_window_range(closes: list[float], window: int) -> None:
    """REQ-MK-02 (property): an average is between the window's min and max."""
    sma = simple_moving_average(closes, window)
    if len(closes) < window:
        assert sma is None
    else:
        recent = closes[-window:]
        assert sma is not None
        assert min(recent) - 1e-6 <= sma <= max(recent) + 1e-6


@given(prices)
def test_snapshot_invariants(closes: list[float]) -> None:
    """REQ-MK-02 (property): range brackets the price; distance from the high is never positive."""
    snap = trend_snapshot(_history(closes))
    assert snap.low_52w <= snap.last_close <= snap.high_52w
    assert snap.pct_below_52w_high <= 1e-12
    assert snap.volatility is None or snap.volatility >= 0
    assert snap.bars == len(closes)


@given(prices, st.floats(min_value=0.01, max_value=100))
def test_returns_are_scale_invariant(closes: list[float], factor: float) -> None:
    """REQ-MK-02 (property): rescaling all prices (e.g. a split) doesn't change returns."""
    a = period_return(closes, 21)
    b = period_return([c * factor for c in closes], 21)
    assert (a is None and b is None) or (
        a is not None and b == pytest.approx(a, rel=1e-9, abs=1e-12)
    )
