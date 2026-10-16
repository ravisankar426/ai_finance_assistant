"""Portfolio engine: examples on recorded data + property-based invariants."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.config import MarketConfig, PortfolioConfig, get_settings
from src.core.models import Holding
from src.core.portfolio_engine import (
    analyze_portfolio,
    asset_class,
    beta,
    daily_returns,
    describe_analysis,
    max_drawdown,
    portfolio_series,
)
from src.data.market_service import MarketDataService
from src.data.models import AssetProfile, PriceBar, PriceHistory
from src.data.providers.fixture_provider import FixtureProvider

NOW = datetime(2026, 10, 7, tzinfo=UTC)


@pytest.fixture(scope="module")
def market() -> MarketDataService:
    return MarketDataService(
        [FixtureProvider(get_settings().market.fixture_dir)], MarketConfig(rate_limits={})
    )


CFG = PortfolioConfig()


def _profile(name: str, asset_type: str, sector: str | None = None) -> AssetProfile:
    return AssetProfile(
        ticker="X", name=name, asset_type=asset_type, sector=sector, as_of=NOW, source="t"
    )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (_profile("Apple Inc.", "equity", "Information Technology"), "Stocks"),
        (_profile("Vanguard Total Bond Market ETF", "etf"), "Bonds"),
        (_profile("iShares 0-3 Month Treasury Bond ETF", "etf"), "Bonds"),
        (_profile("Fidelity Money Market Fund", "fund"), "Cash"),
        (_profile("Vanguard Total Stock Market ETF", "etf"), "Stocks"),
        (_profile("Mystery", "other"), "Other"),
    ],
)
def test_asset_class(profile: AssetProfile, expected: str) -> None:
    """REQ-PF-02: funds are classified by name; stocks by type."""
    assert asset_class(profile) == expected


def test_concentrated_tech_portfolio(market: MarketDataService) -> None:
    """REQ-PF-02, REQ-PF-03: weights, sectors, risk metrics, gains, and flags on real data."""
    holdings = [
        Holding(ticker="NVDA", quantity=100),
        Holding(ticker="AAPL", quantity=50, cost_basis=150),
        Holding(ticker="MSFT", quantity=10),
        Holding(ticker="VTI", quantity=20),
        Holding(ticker="BND", quantity=30),
    ]
    a = analyze_portfolio(holdings, market, CFG)
    assert sum(r.weight for r in a.holdings) == pytest.approx(1)
    assert a.total_value == pytest.approx(sum(r.value for r in a.holdings))
    assert a.holdings[0].ticker == "NVDA"  # sorted by value
    assert set(a.asset_classes) == {"Stocks", "Bonds"}
    assert a.sectors["Information Technology"] > 0.8
    aapl = next(r for r in a.holdings if r.ticker == "AAPL")
    assert aapl.unrealized_gain == pytest.approx((aapl.price - 150) * 50)
    assert a.total_cost == pytest.approx(7500)
    kinds = {(f.kind, f.subject) for f in a.flags}
    assert ("single_holding", "NVDA") in kinds and ("sector", "Information Technology") in kinds
    assert ("no_bonds", "portfolio") not in kinds  # BND present
    assert a.risk is not None and a.risk.trading_days > 200
    assert a.risk.beta is not None and 0.5 < a.risk.beta < 2.5
    assert a.risk.volatility is not None and a.risk.max_drawdown <= 0
    text = describe_analysis(a)
    assert "FLAG: NVDA is" in text and "Sharpe ratio" in text and "unrealized gain" in text


def test_invalid_tickers_skipped_not_fatal(market: MarketDataService) -> None:
    """REQ-PF-05: unknown tickers are listed; the rest is analyzed."""
    a = analyze_portfolio(
        [Holding(ticker="AAPL", quantity=1), Holding(ticker="ZZQXW", quantity=5)], market, CFG
    )
    assert [r.ticker for r in a.holdings] == ["AAPL"] and a.skipped == ["ZZQXW"]
    assert "Skipped (no market data): ZZQXW" in describe_analysis(a)


def test_all_invalid_returns_empty_analysis(market: MarketDataService) -> None:
    """REQ-PF-05: nothing analyzable -> empty result, no crash."""
    a = analyze_portfolio([Holding(ticker="ZZQXW", quantity=1)], market, CFG)
    assert a.holdings == [] and a.total_value == 0 and a.skipped == ["ZZQXW"]


def test_duplicate_tickers_are_merged_with_weighted_cost(market: MarketDataService) -> None:
    """REQ-PF-01: two lots of the same ticker become one position."""
    a = analyze_portfolio(
        [
            Holding(ticker="aapl", quantity=10, cost_basis=100),
            Holding(ticker="$AAPL", quantity=30, cost_basis=200),
        ],
        market,
        CFG,
    )
    (row,) = a.holdings
    assert row.quantity == 40 and row.cost_basis == pytest.approx(175)


def test_few_stocks_no_bonds_flags(market: MarketDataService) -> None:
    """REQ-PF-03: few individual stocks and 100% stocks are flagged."""
    a = analyze_portfolio(
        [Holding(ticker="JNJ", quantity=10), Holding(ticker="XOM", quantity=10)], market, CFG
    )
    assert {f.kind for f in a.flags} >= {"few_holdings", "no_bonds"}


def test_thresholds_come_from_config(market: MarketDataService) -> None:
    """REQ-PF-03: limits are configurable."""
    holdings = [Holding(ticker="VTI", quantity=10), Holding(ticker="BND", quantity=10)]
    strict = analyze_portfolio(holdings, market, PortfolioConfig(max_single_weight=0.5))
    loose = analyze_portfolio(holdings, market, PortfolioConfig(max_single_weight=0.99))
    assert any(f.kind == "single_holding" for f in strict.flags)
    assert not any(f.kind == "single_holding" for f in loose.flags)


def test_max_drawdown_examples() -> None:
    """REQ-PF-02: peak-to-trough fall."""
    assert max_drawdown([100, 120, 90, 130]) == pytest.approx(-0.25)
    assert max_drawdown([1, 2, 3]) == 0


def test_portfolio_series_uses_shared_dates() -> None:
    """REQ-PF-02: values only on dates every holding has a price."""
    d0 = date(2026, 1, 1)

    def hist(closes: dict[int, float]) -> PriceHistory:
        bars = [PriceBar(day=d0 + timedelta(days=k), close=v) for k, v in closes.items()]
        return PriceHistory(ticker="X", bars=bars, as_of=NOW, source="t")

    days, values = portfolio_series(
        {"A": 2, "B": 1}, {"A": hist({0: 10, 1: 11, 2: 12}), "B": hist({1: 100, 2: 90})}
    )
    assert days == [d0 + timedelta(days=1), d0 + timedelta(days=2)]
    assert values == [122, 114]


returns_lists = st.lists(st.floats(min_value=-0.2, max_value=0.2), min_size=25, max_size=200)


@given(returns_lists)
def test_beta_of_series_with_itself_is_one(rets: list[float]) -> None:
    """REQ-PF-02 (property): beta of the market against itself is 1 (when it varies)."""
    b = beta(rets, rets)
    assert b is None or b == pytest.approx(1)


@given(st.lists(st.floats(min_value=1, max_value=1000), min_size=1, max_size=300))
def test_drawdown_bounds(values: list[float]) -> None:
    """REQ-PF-02 (property): drawdown is in [-1, 0]."""
    assert -1 <= max_drawdown(values) <= 0


@given(st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=1, max_size=20))
def test_weights_and_hhi_invariants(values: list[float]) -> None:
    """REQ-PF-02 (property): weights sum to 1; HHI between 1/n and 1."""
    total = sum(values)
    weights = [v / total for v in values]
    hhi = sum(w * w for w in weights)
    assert sum(weights) == pytest.approx(1)
    assert 1 / len(values) - 1e-9 <= hhi <= 1 + 1e-9


def test_daily_returns_skip_zero() -> None:
    """REQ-PF-02: no division by zero on bad data."""
    assert daily_returns([0, 10, 11]) == [pytest.approx(0.1)]


# --- cost basis from the purchase date ------------------------------------------------------


def _close(market: MarketDataService, ticker: str, day: str) -> float:
    from datetime import date

    return next(
        b.close for b in market.get_history(ticker).bars if b.day == date.fromisoformat(day)
    )


def test_cost_basis_from_purchase_date(market: MarketDataService) -> None:
    """REQ-PF-02: a purchase date with no price uses that day's real closing price."""
    from datetime import date

    a = analyze_portfolio(
        [Holding(ticker="NVDA", quantity=10, purchase_date=date(2025, 12, 1))], market, CFG
    )
    row = a.holdings[0]
    paid = _close(market, "NVDA", "2025-12-01")
    assert row.cost_basis == pytest.approx(paid) and row.cost_basis_date == date(2025, 12, 1)
    assert row.unrealized_gain == pytest.approx((row.price - paid) * 10)
    assert any("that day's closing price" in n for n in a.notes)


def test_weekend_purchase_uses_previous_trading_day(market: MarketDataService) -> None:
    """REQ-PF-02: Saturday 2026-03-07 -> Friday 2026-03-06's close."""
    from datetime import date

    a = analyze_portfolio(
        [Holding(ticker="QQQ", quantity=1, purchase_date=date(2026, 3, 7))], market, CFG
    )
    assert a.holdings[0].cost_basis_date == date(2026, 3, 6)
    assert a.holdings[0].cost_basis == pytest.approx(_close(market, "QQQ", "2026-03-06"))


def test_entered_cost_basis_wins_over_date(market: MarketDataService) -> None:
    """REQ-PF-02: an explicit price paid is never overwritten by the historical close."""
    from datetime import date

    a = analyze_portfolio(
        [Holding(ticker="AAPL", quantity=1, cost_basis=150, purchase_date=date(2026, 1, 15))],
        market,
        CFG,
    )
    assert a.holdings[0].cost_basis == 150 and a.holdings[0].cost_basis_date is None


def test_purchase_before_history_is_explained_not_guessed(market: MarketDataService) -> None:
    """REQ-PF-02: a date older than the price history leaves the gain blank, with a note."""
    from datetime import date

    a = analyze_portfolio(
        [Holding(ticker="AAPL", quantity=1, purchase_date=date(2020, 1, 2))], market, CFG
    )
    assert a.holdings[0].cost_basis is None and a.holdings[0].unrealized_gain is None
    assert any("No closing price for AAPL on or before 2020-01-02" in n for n in a.notes)


def test_two_dated_lots_are_cost_weighted(market: MarketDataService) -> None:
    """REQ-PF-02: two purchases of one ticker merge with a quantity-weighted cost."""
    from datetime import date

    a = analyze_portfolio(
        [
            Holding(ticker="AAPL", quantity=10, purchase_date=date(2026, 1, 15)),
            Holding(ticker="AAPL", quantity=30, purchase_date=date(2026, 6, 15)),
        ],
        market,
        CFG,
    )
    c1, c2 = _close(market, "AAPL", "2026-01-15"), _close(market, "AAPL", "2026-06-15")
    assert a.holdings[0].quantity == 40
    assert a.holdings[0].cost_basis == pytest.approx((10 * c1 + 30 * c2) / 40)


def test_future_purchase_date_rejected() -> None:
    """REQ-PF-01: a purchase date after today is invalid input."""
    from datetime import date, timedelta

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="future"):
        Holding(ticker="AAPL", quantity=1, purchase_date=date.today() + timedelta(days=1))
