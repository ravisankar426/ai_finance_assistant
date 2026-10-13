"""Plotly figures for the UI tabs (REQ-UI-03..05).

Pure functions: data in, figure out. No Streamlit and no I/O, so every chart is unit-tested.
Colors come from Plotly's defaults plus a diverging red/green scale for daily moves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import plotly.graph_objects as go

from src.core.goal_engine import GoalPlan
from src.core.portfolio_engine import PerformancePoint
from src.data.models import PriceHistory
from src.data.overview import OverviewRow

UP_DOWN = [[0.0, "#d64545"], [0.5, "#f2f2f2"], [1.0, "#2e9e5b"]]  # red ← 0 → green
LAYOUT = {"margin": {"l": 10, "r": 10, "t": 40, "b": 10}, "height": 360}


def _layout(fig: go.Figure, title: str, **extra: object) -> go.Figure:
    fig.update_layout(title=title, **LAYOUT, **extra)
    return fig


def allocation_donut(weights: dict[str, float], title: str = "Allocation") -> go.Figure:
    """Donut of weights (fractions that sum to ~1)."""
    fig = go.Figure(
        go.Pie(
            labels=[k.replace("_", " ").title() for k in weights],
            values=list(weights.values()),
            hole=0.55,
            sort=False,
            textinfo="label+percent",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    return _layout(fig, title, showlegend=False)


def sector_bar(sectors: dict[str, float], title: str = "By sector") -> go.Figure:
    """Horizontal bars, largest on top."""
    items = sorted(sectors.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[w for _, w in items],
            y=[s for s, _ in items],
            orientation="h",
            text=[f"{w:.0%}" for _, w in items],
            textposition="auto",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    return _layout(fig, title, xaxis={"tickformat": ".0%"})


def performance_chart(points: Sequence[PerformancePoint], benchmark: str = "SPY") -> go.Figure:
    """Growth of 100: the portfolio (today's quantities, held throughout) vs the benchmark."""
    fig = go.Figure(
        go.Scatter(
            x=[p.day for p in points], y=[p.portfolio for p in points], name="Your portfolio"
        )
    )
    if any(p.benchmark is not None for p in points):
        fig.add_trace(
            go.Scatter(
                x=[p.day for p in points],
                y=[p.benchmark for p in points],
                name=benchmark,
                line={"dash": "dot"},
            )
        )
    return _layout(fig, f"Growth of 100 vs {benchmark}", legend={"orientation": "h", "y": -0.15})


def sector_heatmap(rows: Sequence[OverviewRow], title: str = "Sectors today") -> go.Figure:
    """Equal-size tiles colored by the day's % change (diverging around 0)."""
    span = max((abs(r.change_pct) for r in rows), default=1.0) or 1.0
    fig = go.Figure(
        go.Treemap(
            labels=[f"{r.label}<br>{r.change_pct:+.2f}%" for r in rows],
            parents=[""] * len(rows),
            values=[1] * len(rows),
            marker={
                "colors": [r.change_pct for r in rows],
                "colorscale": UP_DOWN,
                "cmin": -span,
                "cmax": span,
            },
            customdata=[r.ticker for r in rows],
            hovertemplate="%{label} (%{customdata})<extra></extra>",
            textfont={"size": 14},
        )
    )
    return _layout(fig, title)


def rolling_mean(values: Sequence[float], window: int) -> list[float | None]:
    """Trailing simple moving average; None until a full window is available.

    Each window is summed exactly (``math.fsum``) rather than with a running total, which
    accumulates rounding error after large values leave the window.
    """
    return [
        math.fsum(values[i - window + 1 : i + 1]) / window if i >= window - 1 else None
        for i in range(len(values))
    ]


def price_chart(history: PriceHistory, windows: Sequence[int] = (50, 200)) -> go.Figure:
    """Daily closes with moving averages (REQ-UI-04)."""
    days = [b.day for b in history.bars]
    closes = [b.close for b in history.bars]
    fig = go.Figure(go.Scatter(x=days, y=closes, name="Close"))
    for w in windows:
        fig.add_trace(
            go.Scatter(x=days, y=rolling_mean(closes, w), name=f"{w}-day avg", line={"width": 1})
        )
    return _layout(
        fig,
        f"{history.ticker} — daily close",
        legend={"orientation": "h", "y": -0.15},
        yaxis={"tickprefix": "$"},
    )


def goal_projection(plan: GoalPlan) -> go.Figure:
    """Pessimistic-to-optimistic band, the base path, and the (inflation-adjusted) target."""
    by_name = {s.name: s for s in plan.scenarios}
    years = list(range(len(by_name["base"].yearly_balances) + 1))

    def path(name: str) -> list[float]:
        return [plan.current_savings, *by_name[name].yearly_balances]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=path("optimistic"), name="Optimistic", line={"width": 0}))
    fig.add_trace(
        go.Scatter(
            x=years,
            y=path("pessimistic"),
            name="Pessimistic to optimistic range",
            fill="tonexty",
            fillcolor="rgba(99,110,250,0.18)",
            line={"width": 0},
        )
    )
    fig.add_trace(go.Scatter(x=years, y=path("base"), name="Base case", line={"width": 3}))
    fig.add_hline(
        y=plan.target_nominal,
        line_dash="dash",
        annotation_text=f"Target ${plan.target_nominal:,.0f}",
        annotation_position="top left",
    )
    return _layout(
        fig,
        "Projected balance",
        xaxis={"title": "Years", "dtick": 1 if len(years) <= 15 else 5},
        yaxis={"tickprefix": "$", "tickformat": ",.0f"},
        legend={"orientation": "h", "y": -0.2},
    )
