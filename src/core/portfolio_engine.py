"""Portfolio analysis engine — every number the Portfolio agent explains (REQ-PF-02..05).

Pure computation over market data; the LLM never calculates. Steps:
1. price each holding (skip tickers without data -> ``skipped``, REQ-PF-05);
2. weights, asset-class mix, stock-sector mix, concentration (HHI);
3. history-based risk metrics on a buy-and-hold series of the *current* holdings:
   annualized volatility, max drawdown, Sharpe ratio, beta vs. the benchmark, 1-year return;
4. unrealized gain/loss where a cost basis was given;
5. flags: a holding or stock sector above its configured limit (REQ-PF-03).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, Field

from src.core.config import PortfolioConfig
from src.core.errors import AppError
from src.core.models import Holding
from src.data.market_service import MarketDataService
from src.data.models import AssetProfile, PriceHistory

AssetClass = Literal["Stocks", "Bonds", "Cash", "Other"]

_BOND_WORDS = ("bond", "treasury", "fixed income", "aggregate", "municipal", "tips")
_CASH_WORDS = ("money market", "ultra short", "t-bill", "treasury bill")


def asset_class(profile: AssetProfile) -> AssetClass:
    """Classify a holding as stocks, bonds, or cash (fund names carry the signal)."""
    if profile.asset_type == "equity":
        return "Stocks"
    name = profile.name.lower()
    if any(w in name for w in _CASH_WORDS):
        return "Cash"
    if any(w in name for w in _BOND_WORDS):
        return "Bonds"
    if profile.asset_type in ("etf", "fund", "index"):
        return "Stocks"
    return "Other"


def sector_label(profile: AssetProfile, cls: AssetClass) -> str:
    """GICS sector for individual stocks; funds are grouped, not split by sector."""
    if cls == "Stocks" and profile.asset_type == "equity":
        return profile.sector or "Unknown sector"
    if cls == "Stocks":
        return "Diversified stock funds"
    return cls


class HoldingRow(BaseModel):
    """One analyzed position."""

    ticker: str
    name: str
    quantity: float
    price: float
    value: float
    weight: float
    asset_class: AssetClass
    sector: str
    cost_basis: float | None = None
    unrealized_gain: float | None = None
    unrealized_gain_pct: float | None = None
    stale: bool = False


class RiskMetrics(BaseModel):
    """History-based metrics for the current holdings held over the lookback period."""

    start: date
    end: date
    trading_days: int
    annual_return: float | None
    volatility: float | None
    max_drawdown: float
    sharpe: float | None
    beta: float | None


class Flag(BaseModel):
    """An educational observation triggered by a threshold (REQ-PF-03)."""

    kind: Literal["single_holding", "sector", "few_holdings", "no_bonds"]
    subject: str
    value: float
    limit: float
    message: str


class PortfolioAnalysis(BaseModel):
    """The engine's full output; also the payload for UI charts."""

    total_value: float
    holdings: list[HoldingRow]
    asset_classes: dict[str, float]  # weight by asset class
    sectors: dict[str, float]  # weight by sector label
    hhi: float
    effective_holdings: float
    risk: RiskMetrics | None
    total_cost: float | None = None
    total_unrealized_gain: float | None = None
    flags: list[Flag] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


# --- risk math --------------------------------------------------------------------------------


def daily_returns(values: Sequence[float]) -> list[float]:
    """Return simple daily returns of a value series."""
    return [b / a - 1 for a, b in pairwise(values) if a > 0]


def max_drawdown(values: Sequence[float]) -> float:
    """Largest peak-to-trough fall as a negative fraction (0 if it never fell)."""
    peak, worst = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, v / peak - 1)
    return worst


def _stdev(xs: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def beta(asset: Sequence[float], market: Sequence[float]) -> float | None:
    """cov(asset, market) / var(market) over aligned daily returns."""
    n = min(len(asset), len(market))
    if n < 21:
        return None
    a, m = asset[-n:], market[-n:]
    ma, mm = sum(a) / n, sum(m) / n
    var_m = sum((x - mm) ** 2 for x in m) / (n - 1)
    if var_m == 0:
        return None
    cov = sum((x - ma) * (y - mm) for x, y in zip(a, m, strict=True)) / (n - 1)
    return cov / var_m


def portfolio_series(
    quantities: dict[str, float], histories: dict[str, PriceHistory]
) -> tuple[list[date], list[float]]:
    """Daily value of holding today's quantities on the dates every history shares."""
    closes = {t: {b.day: b.close for b in h.bars} for t, h in histories.items()}
    common = sorted(set.intersection(*(set(c) for c in closes.values()))) if closes else []
    return common, [sum(quantities[t] * closes[t][d] for t in closes) for d in common]


def risk_metrics(
    days: list[date], values: list[float], bench: PriceHistory | None, risk_free: float
) -> RiskMetrics | None:
    """Volatility, drawdown, Sharpe, beta, and return over the shared window."""
    if len(values) < 22:
        return None
    rets = daily_returns(values)
    sd = _stdev(rets)
    vol = sd * math.sqrt(252) if sd is not None else None
    years = len(rets) / 252
    annual = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 and values[0] > 0 else None
    sharpe = (annual - risk_free) / vol if annual is not None and vol else None
    b = None
    if bench is not None:
        bench_close = {bar.day: bar.close for bar in bench.bars}
        aligned = [d for d in days if d in bench_close]
        if len(aligned) == len(days):
            b = beta(rets, daily_returns([bench_close[d] for d in days]))
    return RiskMetrics(
        start=days[0],
        end=days[-1],
        trading_days=len(days),
        annual_return=annual,
        volatility=vol,
        max_drawdown=max_drawdown(values),
        sharpe=sharpe,
        beta=b,
    )


# --- main entry point -------------------------------------------------------------------------


def analyze_portfolio(
    holdings: Sequence[Holding], market: MarketDataService, cfg: PortfolioConfig
) -> PortfolioAnalysis:
    """Price, measure, and flag a portfolio. Tickers without data are skipped, not fatal."""
    merged: dict[str, Holding] = {}
    for h in holdings[: cfg.max_holdings]:
        t = h.ticker.strip().upper().lstrip("$")
        if t in merged:  # same ticker twice: add quantities; cost basis becomes weighted
            prev = merged[t]
            qty = prev.quantity + h.quantity
            cost = None
            if prev.cost_basis and h.cost_basis:
                cost = (prev.cost_basis * prev.quantity + h.cost_basis * h.quantity) / qty
            merged[t] = Holding(ticker=t, quantity=qty, cost_basis=cost)
        else:
            merged[t] = h.model_copy(update={"ticker": t})

    rows: list[HoldingRow] = []
    histories: dict[str, PriceHistory] = {}
    skipped: list[str] = []
    notes: list[str] = []
    sources: set[str] = set()
    for t, h in merged.items():
        try:
            quote = market.get_quote(t)
        except AppError:
            skipped.append(t)
            continue
        try:
            profile = market.get_profile(t)
        except AppError:
            profile = AssetProfile(ticker=t, name=t, as_of=quote.as_of, source=quote.source)
            notes.append(f"No profile for {t}; classified as 'Other'.")
        try:
            histories[t] = market.get_history(t)
        except AppError:
            notes.append(f"No price history for {t}; risk metrics exclude it.")
        cls = asset_class(profile)
        value = h.quantity * quote.price
        gain = (quote.price - h.cost_basis) * h.quantity if h.cost_basis else None
        rows.append(
            HoldingRow(
                ticker=t,
                name=profile.name,
                quantity=h.quantity,
                price=quote.price,
                value=value,
                weight=0.0,
                asset_class=cls,
                sector=sector_label(profile, cls),
                cost_basis=h.cost_basis,
                unrealized_gain=gain,
                unrealized_gain_pct=(quote.price / h.cost_basis - 1) if h.cost_basis else None,
                stale=quote.stale,
            )
        )
        sources.add(quote.source)

    total = sum(r.value for r in rows)
    if not rows or total <= 0:
        return PortfolioAnalysis(
            total_value=0,
            holdings=[],
            asset_classes={},
            sectors={},
            hhi=0,
            effective_holdings=0,
            risk=None,
            skipped=skipped,
            notes=notes,
            sources=sorted(sources),
        )
    for r in rows:
        r.weight = r.value / total
    rows.sort(key=lambda r: r.value, reverse=True)

    classes: dict[str, float] = defaultdict(float)
    sectors: dict[str, float] = defaultdict(float)
    for r in rows:
        classes[r.asset_class] += r.weight
        sectors[r.sector] += r.weight
    hhi = sum(r.weight**2 for r in rows)

    risk = None
    priced = {r.ticker: r.quantity for r in rows if r.ticker in histories}
    if priced:
        days, values = portfolio_series(priced, {t: histories[t] for t in priced})
        bench = None
        try:
            bench = market.get_history(cfg.benchmark)
        except AppError:
            notes.append(f"Benchmark {cfg.benchmark} unavailable; beta not computed.")
        risk = risk_metrics(days, values, bench, cfg.risk_free_rate)
        if any(not histories[t].adjusted for t in priced):
            notes.append(
                "Some histories are unadjusted (fallback source); returns may be distorted."
            )
        if len(priced) < len(rows):
            notes.append("Risk metrics cover only holdings with price history.")

    costs = [(r.cost_basis * r.quantity) for r in rows if r.cost_basis is not None]
    with_cost = [r for r in rows if r.cost_basis is not None]
    total_cost = sum(costs) if costs else None
    total_gain = (
        sum(r.unrealized_gain for r in with_cost if r.unrealized_gain is not None)
        if with_cost
        else None
    )

    return PortfolioAnalysis(
        total_value=total,
        holdings=rows,
        asset_classes=dict(sorted(classes.items(), key=lambda kv: -kv[1])),
        sectors=dict(sorted(sectors.items(), key=lambda kv: -kv[1])),
        hhi=hhi,
        effective_holdings=1 / hhi,
        risk=risk,
        total_cost=total_cost,
        total_unrealized_gain=total_gain,
        flags=_flags(rows, sectors, classes, cfg),
        skipped=skipped,
        notes=notes,
        sources=sorted(sources),
    )


def _flags(
    rows: list[HoldingRow],
    sectors: dict[str, float],
    classes: dict[str, float],
    cfg: PortfolioConfig,
) -> list[Flag]:
    flags = [
        Flag(
            kind="single_holding",
            subject=r.ticker,
            value=r.weight,
            limit=cfg.max_single_weight,
            message=f"{r.ticker} is {r.weight:.0%} of the portfolio "
            f"(above {cfg.max_single_weight:.0%}).",
        )
        for r in rows
        if r.weight > cfg.max_single_weight
    ]
    for sector, w in sectors.items():
        if sector in ("Diversified stock funds", "Bonds", "Cash", "Other"):
            continue  # concentration checks apply to individual-stock sectors
        if w > cfg.max_sector_weight:
            flags.append(
                Flag(
                    kind="sector",
                    subject=sector,
                    value=w,
                    limit=cfg.max_sector_weight,
                    message=f"{sector} stocks are {w:.0%} of the portfolio "
                    f"(above {cfg.max_sector_weight:.0%}).",
                )
            )
    individual = [r for r in rows if r.sector not in ("Diversified stock funds", "Bonds", "Cash")]
    if rows and len(individual) == len(rows) and len(rows) < 5:
        flags.append(
            Flag(
                kind="few_holdings",
                subject="portfolio",
                value=len(rows),
                limit=5,
                message=f"Only {len(rows)} individual stocks and no diversified funds.",
            )
        )
    if classes.get("Bonds", 0) == 0 and classes.get("Cash", 0) == 0:
        flags.append(
            Flag(
                kind="no_bonds",
                subject="portfolio",
                value=0,
                limit=0,
                message="No bonds or cash: the portfolio is 100% stocks.",
            )
        )
    return flags


def _fmt(x: float | None, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def describe_analysis(a: PortfolioAnalysis) -> str:
    """Fact block for the LLM (it must not compute new numbers)."""
    lines = [f"PORTFOLIO (total ${a.total_value:,.2f}; data: {', '.join(a.sources)})"]
    for row in a.holdings:
        gain = ""
        if row.unrealized_gain is not None and row.unrealized_gain_pct is not None:
            word = "gain" if row.unrealized_gain >= 0 else "loss"
            gain = (
                f"; unrealized {word} ${abs(row.unrealized_gain):,.2f} "
                f"({row.unrealized_gain_pct:+.1%})"
            )
        lines.append(
            f"  {row.ticker} ({row.name}): {row.quantity:g} x ${row.price:,.2f} = "
            f"${row.value:,.2f}, {row.weight:.1%}; {row.asset_class}; {row.sector}{gain}"
            f"{'; STALE' if row.stale else ''}"
        )
    lines.append("Asset classes: " + ", ".join(f"{k} {v:.1%}" for k, v in a.asset_classes.items()))
    lines.append("Sectors: " + ", ".join(f"{k} {v:.1%}" for k, v in a.sectors.items()))
    lines.append(
        f"Concentration: HHI {a.hhi:.3f} (equivalent to {a.effective_holdings:.1f} equally "
        "weighted holdings)"
    )
    if a.risk:
        risk = a.risk
        lines.append(
            f"Risk over {risk.start} to {risk.end} ({risk.trading_days} trading days, holding "
            f"today's quantities): annualized return {_fmt(risk.annual_return, '+.1%')}, "
            f"annualized volatility {_fmt(risk.volatility, '.1%')}, max drawdown "
            f"{risk.max_drawdown:.1%}, Sharpe ratio {_fmt(risk.sharpe, '.2f')}, "
            f"beta vs SPY {_fmt(risk.beta, '.2f')}"
        )
    if a.total_unrealized_gain is not None and a.total_cost:
        lines.append(
            f"Unrealized {'gain' if a.total_unrealized_gain >= 0 else 'loss'} on positions with a "
            f"cost basis: ${abs(a.total_unrealized_gain):,.2f} on ${a.total_cost:,.2f} invested"
        )
    lines += [f"FLAG: {f.message}" for f in a.flags]
    if a.skipped:
        lines.append(f"Skipped (no market data): {', '.join(a.skipped)}")
    lines += [f"NOTE: {n}" for n in a.notes]
    return "\n".join(lines)
