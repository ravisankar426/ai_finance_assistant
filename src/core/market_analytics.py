"""Deterministic market analytics (constitution P2: numbers come from code).

The Market agent's LLM only *explains* what these functions compute (REQ-MK-02, REQ-MK-03).
Windows are in trading days: ~21 per month, ~63 per quarter, ~252 per year.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel

from src.data.models import PriceHistory

TRADING_DAYS = {"1m": 21, "3m": 63, "1y": 252}

INDEX_ETFS: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "DIA": "Dow Jones Industrial Average",
    "IWM": "Russell 2000 (small caps)",
}

# The 11 Select Sector SPDR ETFs, one per GICS sector (REQ-MK-03).
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Information Technology",
    "XLV": "Health Care",
    "XLF": "Financials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

Trend = Literal["uptrend", "downtrend", "mixed", "insufficient data"]


def simple_moving_average(closes: Sequence[float], window: int) -> float | None:
    """Mean of the last ``window`` closes; None if there aren't enough."""
    if window <= 0 or len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def period_return(closes: Sequence[float], days: int) -> float | None:
    """Return over the last ``days`` trading days as a fraction (0.05 = +5%)."""
    if days <= 0 or len(closes) <= days or closes[-1 - days] == 0:
        return None
    return closes[-1] / closes[-1 - days] - 1


def annualized_volatility(closes: Sequence[float], window: int = 252) -> float | None:
    """Sample std. dev. of daily log returns over ``window`` days, times sqrt(252)."""
    recent = [c for c in closes[-(window + 1) :] if c > 0]
    if len(recent) < 21:  # less than a month of data isn't a meaningful estimate
        return None
    rets = [math.log(b / a) for a, b in pairwise(recent)]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def classify_trend(price: float, sma_50: float | None, sma_200: float | None) -> Trend:
    """Uptrend: price > 50-day > 200-day; downtrend: the reverse; otherwise mixed."""
    if sma_50 is None or sma_200 is None:
        return "insufficient data"
    if price > sma_50 > sma_200:
        return "uptrend"
    if price < sma_50 < sma_200:
        return "downtrend"
    return "mixed"


class TrendSnapshot(BaseModel):
    """Everything the Market agent may say about a ticker's recent behavior."""

    ticker: str
    last_close: float
    last_date: date
    bars: int
    adjusted: bool
    sma_50: float | None
    sma_200: float | None
    pct_vs_sma_50: float | None
    pct_vs_sma_200: float | None
    return_1m: float | None
    return_3m: float | None
    return_1y: float | None
    high_52w: float
    low_52w: float
    pct_below_52w_high: float
    volatility: float | None
    trend: Trend
    source: str
    stale: bool


def _pct_vs(price: float, ref: float | None) -> float | None:
    return None if not ref else price / ref - 1


def trend_snapshot(history: PriceHistory) -> TrendSnapshot:
    """Compute trend indicators from daily closes (REQ-MK-02)."""
    closes = history.closes
    if not closes:
        raise ValueError(f"no prices for {history.ticker}")
    price = closes[-1]
    year = closes[-TRADING_DAYS["1y"] :]
    sma_50 = simple_moving_average(closes, 50)
    sma_200 = simple_moving_average(closes, 200)
    high = max(year)
    return TrendSnapshot(
        ticker=history.ticker,
        last_close=price,
        last_date=history.bars[-1].day,
        bars=len(closes),
        adjusted=history.adjusted,
        sma_50=sma_50,
        sma_200=sma_200,
        pct_vs_sma_50=_pct_vs(price, sma_50),
        pct_vs_sma_200=_pct_vs(price, sma_200),
        return_1m=period_return(closes, TRADING_DAYS["1m"]),
        return_3m=period_return(closes, TRADING_DAYS["3m"]),
        return_1y=period_return(closes, TRADING_DAYS["1y"]),
        high_52w=high,
        low_52w=min(year),
        pct_below_52w_high=price / high - 1,
        volatility=annualized_volatility(closes),
        trend=classify_trend(price, sma_50, sma_200),
        source=history.source,
        stale=history.stale,
    )


def fmt_pct(value: float | None, signed: bool = True) -> str:
    """Format a fraction as a percentage ('n/a' when missing)."""
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}%" if signed else f"{value * 100:.1f}%"


def describe_snapshot(s: TrendSnapshot, name: str | None = None) -> str:
    """Render a snapshot as the fact block the LLM explains (it must not compute new numbers)."""
    header = f"{s.ticker}" + (f" — {name}" if name else "")
    basis = "adjusted daily closes" if s.adjusted else "UNADJUSTED daily closes (fallback source)"
    sma50 = f"${s.sma_50:,.2f}" if s.sma_50 else "n/a"
    sma200 = f"${s.sma_200:,.2f}" if s.sma_200 else "n/a"
    lines = [
        f"{header} [source: {s.source}; {basis}; {s.bars} trading days; last {s.last_date}"
        f"{'; STALE' if s.stale else ''}]",
        f"  last close ${s.last_close:,.2f}",
        f"  50-day SMA {sma50} (price {fmt_pct(s.pct_vs_sma_50)} vs it) | "
        f"200-day SMA {sma200} (price {fmt_pct(s.pct_vs_sma_200)} vs it)",
        f"  trend classification: {s.trend}",
        f"  returns: 1 month {fmt_pct(s.return_1m)} | 3 months {fmt_pct(s.return_3m)} | "
        f"1 year {fmt_pct(s.return_1y)}",
        f"  52-week range ${s.low_52w:,.2f} to ${s.high_52w:,.2f} "
        f"(price {fmt_pct(s.pct_below_52w_high)} vs the high)",
        f"  annualized volatility {fmt_pct(s.volatility, signed=False)}",
    ]
    return "\n".join(lines)
