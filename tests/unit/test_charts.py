"""Chart builders, the performance series, and holdings-table parsing (REQ-UI-03..05)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.config import PlanningConfig
from src.core.goal_engine import plan_goal
from src.core.portfolio_engine import PerformancePoint, performance_series
from src.data.models import PriceBar, PriceHistory
from src.data.overview import OverviewRow
from src.web_app.ui import charts
from src.web_app.ui.tabs import frame_to_holdings, holdings_frame, sample_names

NOW = datetime(2026, 10, 14, tzinfo=UTC)
DAYS = [date(2026, 1, 1) + timedelta(days=i) for i in range(5)]


def history(closes: list[float], ticker: str = "SPY") -> PriceHistory:
    bars = [PriceBar(day=d, close=c) for d, c in zip(DAYS, closes, strict=False)]
    return PriceHistory(ticker=ticker, bars=bars, as_of=NOW, source="test")


def test_performance_series_indexes_both_lines_to_100() -> None:
    """REQ-UI-03: growth of 100 for the portfolio and the benchmark on shared dates."""
    pts = performance_series(DAYS, [50, 55, 60, 45, 50], history([400, 440, 400, 380, 420]))
    assert [p.portfolio for p in pts] == [100, 110, 120, 90, 100]
    assert pts[1].benchmark == pytest.approx(110) and pts[-1].benchmark == pytest.approx(105)


def test_performance_series_without_benchmark_or_values() -> None:
    """REQ-UI-03: missing benchmark -> portfolio line only; no data -> empty."""
    assert all(p.benchmark is None for p in performance_series(DAYS, [1, 2, 3, 4, 5], None))
    assert performance_series([], [], None) == []


@given(st.lists(st.floats(0.01, 1e6), min_size=1, max_size=60), st.integers(1, 20))
def test_rolling_mean_matches_definition(values: list[float], window: int) -> None:
    """REQ-UI-04: SMA is None before a full window, then the window's mean."""
    out = charts.rolling_mean(values, window)
    for i, v in enumerate(out):
        if i < window - 1:
            assert v is None
        else:
            assert v == pytest.approx(sum(values[i - window + 1 : i + 1]) / window, rel=1e-9)


def test_allocation_and_sector_charts() -> None:
    """REQ-UI-03: donut labels are readable; sector bars are sorted with the largest on top."""
    donut = charts.allocation_donut({"us_equity": 0.7, "bonds": 0.3})
    assert list(donut.data[0].labels) == ["Us Equity", "Bonds"] and donut.data[0].hole == 0.55
    bars = charts.sector_bar({"Tech": 0.6, "Energy": 0.1, "Health": 0.3})
    assert list(bars.data[0].y) == ["Energy", "Health", "Tech"]


def test_performance_chart_adds_benchmark_only_when_present() -> None:
    """REQ-UI-03: one line without a benchmark, two with."""
    with_b = [PerformancePoint(day=DAYS[0], portfolio=100, benchmark=100)]
    without = [PerformancePoint(day=DAYS[0], portfolio=100)]
    assert len(charts.performance_chart(with_b).data) == 2
    assert len(charts.performance_chart(without).data) == 1


def test_sector_heatmap_colors_are_symmetric_around_zero() -> None:
    """REQ-UI-04: diverging colors centered on 0 so red/green mean down/up."""
    rows = [
        OverviewRow(ticker="XLK", label="Technology", price=1, change_pct=2.0, source="t"),
        OverviewRow(ticker="XLE", label="Energy", price=1, change_pct=-0.5, source="t"),
    ]
    marker = charts.sector_heatmap(rows).data[0].marker
    assert (marker.cmin, marker.cmax) == (-2.0, 2.0)
    assert "+2.00%" in charts.sector_heatmap(rows).data[0].labels[0]


def test_price_chart_has_close_and_smas() -> None:
    """REQ-UI-04: close plus one trace per moving-average window."""
    fig = charts.price_chart(history([1, 2, 3, 4, 5]), windows=(2, 3))
    assert [t.name for t in fig.data] == ["Close", "2-day avg", "3-day avg"]
    assert list(fig.data[1].y) == [None, 1.5, 2.5, 3.5, 4.5]


def test_goal_projection_band_and_target() -> None:
    """REQ-UI-05: year 0 = current savings; band between pessimistic and optimistic; target line."""
    plan = plan_goal(
        goal="car",
        target_today=20000,
        years=3,
        current_savings=1000,
        planned_monthly=300,
        risk_profile="moderate",
        cfg=PlanningConfig(),
    )
    fig = charts.goal_projection(plan)
    opt, pess, base = fig.data
    assert list(base.x) == [0, 1, 2, 3] and base.y[0] == 1000
    assert pess.fill == "tonexty" and all(o >= p for o, p in zip(opt.y, pess.y, strict=True))
    assert fig.layout.shapes[0].y0 == pytest.approx(plan.target_nominal)


def test_holdings_table_round_trip_and_validation() -> None:
    """REQ-UI-03, REQ-PF-01: edited rows use the CSV parser; blank rows are ignored."""
    df = holdings_frame("ticker,quantity\naapl,10\nMSFT,-1\n")
    assert list(df.columns) == ["ticker", "quantity", "cost_basis", "purchase_date"]
    df = pd.concat([df, pd.DataFrame([{"ticker": None, "quantity": None}])], ignore_index=True)
    holdings, errors = frame_to_holdings(df)
    assert [h.ticker for h in holdings] == ["AAPL"] and len(errors) == 1


def test_holdings_table_keeps_purchase_dates() -> None:
    """REQ-UI-03, REQ-PF-01: dates survive the table round trip; a bad date is reported."""
    from datetime import date

    df = holdings_frame("ticker,quantity,purchase_date\nNVDA,5,2025-12-01\nAAPL,1,12/01/2025\n")
    assert df.loc[0, "purchase_date"] == "2025-12-01"  # text, so typos stay visible
    holdings, errors = frame_to_holdings(df)
    assert holdings[0].purchase_date == date(2025, 12, 1)
    assert errors == ["Row 3 (AAPL): purchase date must look like 2026-01-15."]


def test_samples_are_bundled() -> None:
    """REQ-UI-03: sample portfolios are offered in the Portfolio tab."""
    assert {"balanced_funds", "concentrated_tech", "with_errors"} <= set(sample_names())
