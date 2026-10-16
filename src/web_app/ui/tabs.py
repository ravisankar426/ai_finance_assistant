"""Portfolio, Market, Goals, and Learn tabs (REQ-UI-03..06).

Each tab calls the shared :class:`AssistantService` (the same facade the API uses) and renders
charts from :mod:`charts`. Expected failures show the error's user-safe message.
"""

from __future__ import annotations

from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import cast, get_args

import pandas as pd
import streamlit as st
from langchain_core.exceptions import ModelError

from src.core.config import PROJECT_ROOT
from src.core.errors import AppError
from src.core.goal_engine import QUESTIONNAIRE, RiskProfile
from src.core.holdings_io import parse_holdings_csv
from src.core.market_analytics import fmt_pct
from src.core.models import Holding
from src.core.portfolio_engine import PortfolioAnalysis
from src.core.services import AssistantService, GoalPlanRequest
from src.web_app.ui import charts

SAMPLES_DIR = PROJECT_ROOT / "src" / "data" / "sample_portfolios"
HOLDING_COLUMNS = ["ticker", "quantity", "cost_basis", "purchase_date"]
RISK_PROFILES: list[RiskProfile] = list(get_args(RiskProfile))


def safely[T](fn: Callable[[], T]) -> T | None:
    """Run a service call; show a safe message instead of a traceback on failure."""
    try:
        return fn()
    except AppError as exc:
        st.error(exc.user_message)
    except ModelError:
        st.error("The AI service is misconfigured or unavailable. Details are in the server logs.")
    return None


def _money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def _num(x: float | None, spec: str = ".2f") -> str:
    return "n/a" if x is None else format(x, spec)


# -- Portfolio ---------------------------------------------------------------------------------


def sample_names() -> list[str]:
    """Bundled sample portfolios (file stems)."""
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.csv"))


def sample_path(name: str) -> Path:
    """Path of a bundled sample portfolio."""
    return SAMPLES_DIR / f"{name}.csv"


def holdings_frame(csv_text: str) -> pd.DataFrame:
    """Editable table from CSV text (invalid rows are kept so the user can fix them)."""
    df = pd.read_csv(StringIO(csv_text), dtype={"ticker": str, "purchase_date": str})
    for col in HOLDING_COLUMNS:
        if col not in df:
            df[col] = None
    return df[HOLDING_COLUMNS]


def frame_to_holdings(df: pd.DataFrame) -> tuple[list[Holding], list[str]]:
    """Validate the edited table with the same parser as CSV uploads."""
    rows = df.dropna(subset=["ticker"])
    rows = rows[rows["ticker"].astype(str).str.strip() != ""]
    return parse_holdings_csv(rows[HOLDING_COLUMNS].to_csv(index=False))


def portfolio_metrics(a: PortfolioAnalysis) -> None:
    """Metric cards (columns stack on narrow screens)."""
    r = a.risk
    cards = [
        ("Total value", _money(a.total_value), None),
        (
            "Unrealized gain",
            _money(a.total_unrealized_gain),
            fmt_pct(a.total_unrealized_gain / a.total_cost)
            if a.total_unrealized_gain is not None and a.total_cost
            else None,
        ),
        ("Effective holdings", f"{a.effective_holdings:.1f}", None),
        ("Annual return (past)", fmt_pct(r.annual_return) if r else "n/a", None),
        ("Volatility", fmt_pct(r.volatility, signed=False) if r else "n/a", None),
        ("Max drawdown", fmt_pct(r.max_drawdown) if r else "n/a", None),
        ("Sharpe ratio", _num(r.sharpe) if r else "n/a", None),
        ("Beta vs SPY", _num(r.beta) if r else "n/a", None),
    ]
    for start in (0, 4):
        for col, (label, value, delta) in zip(st.columns(4), cards[start : start + 4], strict=True):
            col.metric(label, value, delta)


def portfolio_tab(service: AssistantService) -> None:
    """Render holdings entry (table or CSV), metric cards, allocation charts, and performance."""
    st.subheader("Analyze a portfolio")
    st.caption(
        "Enter holdings in the table, upload a CSV (`ticker,quantity,cost_basis,purchase_date`), "
        "or start from a sample. Give either the price you paid or the date you bought: a date "
        "uses that day's closing price as your cost. Analyzed holdings are also shared with "
        "the Chat tab."
    )
    left, right = st.columns(2)
    sample = left.selectbox("Start from a sample", sample_names(), key="pf_sample")
    upload = right.file_uploader("…or upload a CSV", type=["csv"], key="pf_upload")
    source = upload.getvalue().decode("utf-8-sig") if upload else sample_path(sample).read_text()
    edited = st.data_editor(
        holdings_frame(source),
        num_rows="dynamic",
        width="stretch",
        key=f"pf_editor_{upload.name if upload else sample}",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            "quantity": st.column_config.NumberColumn("Shares", min_value=0.0),
            "cost_basis": st.column_config.NumberColumn(
                "Cost / share ($)", min_value=0.0, help="Optional: the price you paid per share"
            ),
            "purchase_date": st.column_config.TextColumn(
                "Bought on",
                help="Optional, YYYY-MM-DD. If Cost / share is blank, that day's closing price "
                "is used.",
                validate=r"^\d{4}-\d{2}-\d{2}$",
            ),
        },
    )
    if st.button("Analyze portfolio", type="primary", key="pf_analyze"):
        holdings, errors = frame_to_holdings(edited)
        for e in errors:
            st.warning(e)
        if holdings:
            with st.spinner("Fetching prices…"):
                analysis = safely(lambda: service.analyze_portfolio(holdings))
            if analysis is not None:
                st.session_state.analysis = analysis
                st.session_state.portfolio = holdings
    analysis = st.session_state.get("analysis")
    if analysis is None:
        return
    render_analysis(analysis, service.settings.portfolio.benchmark)


def render_analysis(a: PortfolioAnalysis, benchmark: str) -> None:
    """Results: warnings, cards, charts, table, provenance."""
    for t in a.skipped:
        st.warning(f"No market data for {t}; it was left out.")
    if not a.holdings:
        return
    portfolio_metrics(a)
    for f in a.flags:
        st.warning(f.message, icon="⚠️")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.allocation_donut(a.asset_classes, "By asset class"), width="stretch")
    c2.plotly_chart(charts.sector_bar(a.sectors), width="stretch")
    if a.performance:
        st.plotly_chart(charts.performance_chart(a.performance, benchmark), width="stretch")
        st.caption(
            "Holds today's quantities over the whole window (no trades, no dividends unless "
            "prices are adjusted). Past performance doesn't predict future results."
        )
    st.dataframe(
        pd.DataFrame(
            {
                "Ticker": h.ticker,
                "Name": h.name,
                "Shares": h.quantity,
                "Price": h.price,
                "Value": h.value,
                "Weight": h.weight,
                "Sector": h.sector,
                "Paid / share": h.cost_basis,
                "Cost from": (
                    f"close {h.cost_basis_date}"
                    if h.cost_basis_date
                    else ("entered" if h.cost_basis is not None else "")
                ),
                "Gain %": h.unrealized_gain_pct,
            }
            for h in a.holdings
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Value": st.column_config.NumberColumn(format="$%.0f"),
            "Paid / share": st.column_config.NumberColumn(format="$%.2f"),
            "Weight": st.column_config.NumberColumn(format="percent"),
            "Gain %": st.column_config.NumberColumn(format="percent"),
        },
    )
    for n in a.notes:
        st.caption(n)
    st.caption("Prices: " + ", ".join(a.sources))


# -- Market ------------------------------------------------------------------------------------


def market_tab(service: AssistantService) -> None:
    """Index overview, sector heatmap, and a ticker chart with moving averages (REQ-UI-04)."""
    head, button = st.columns([4, 1])
    head.subheader("Market overview")
    if button.button("↻ Refresh", key="mk_refresh", help="Re-fetch data whose cache has expired"):
        st.toast("Refreshed — cached prices are reused until they expire.")
    overview = safely(service.overview)
    if overview is not None:
        cols = st.columns(max(1, len(overview.indices)))
        for col, row in zip(cols, overview.indices, strict=False):
            col.metric(row.label, f"${row.price:,.2f}", f"{row.change_pct:+.2f}%")
        if overview.sectors:
            st.plotly_chart(charts.sector_heatmap(overview.sectors), width="stretch")
        sources = sorted({r.source for r in [*overview.indices, *overview.sectors]})
        st.caption(
            f"As of {overview.as_of:%Y-%m-%d %H:%M} UTC · source: {', '.join(sources)}"
            + (f" · unavailable: {', '.join(overview.unavailable)}" if overview.unavailable else "")
        )

    st.subheader("Look up a ticker")
    ticker = st.text_input("Ticker", value="SPY", max_chars=10, key="mk_ticker").strip().upper()
    if not ticker:
        return
    trend = safely(lambda: service.trend(ticker))
    if trend is None:
        return
    s = trend.snapshot
    for col, (label, value) in zip(
        st.columns(4),
        [
            ("Last close", f"${s.last_close:,.2f}"),
            ("1-year return", fmt_pct(s.return_1y)),
            ("Volatility", fmt_pct(s.volatility, signed=False)),
            ("Trend", s.trend.capitalize()),
        ],
        strict=True,
    ):
        col.metric(label, value)
    history = safely(lambda: service.history(ticker))
    if history is not None:
        st.plotly_chart(charts.price_chart(history), width="stretch")
    st.caption(
        f"Close on {s.last_date} · {s.source}{' (stale)' if s.stale else ''}. "
        "Moving averages smooth daily noise; they describe the past, not the future."
    )


# -- Goals -------------------------------------------------------------------------------------


def risk_questionnaire() -> None:
    """Five questions -> a risk profile stored for the Goals form and the chat (REQ-GP-01)."""
    with (
        st.expander("Not sure about your risk profile? Take the 5-question quiz"),
        st.form("risk_quiz"),
    ):
        answers = {
            str(q["id"]): st.radio(
                str(q["question"]),
                list(cast(dict[str, int], q["options"])),
                key=f"rq_{q['id']}",
            )
            for q in QUESTIONNAIRE
        }
        if st.form_submit_button("Score my answers"):
            profile = AssistantService.risk_profile({k: str(v) for k, v in answers.items()})
            st.session_state.risk_profile = profile
            st.success(f"Your risk profile: **{profile}**")


def goals_tab(service: AssistantService) -> None:
    """Goal inputs -> projection with pessimistic/base/optimistic band (REQ-UI-05)."""
    st.subheader("Plan a goal")
    risk_questionnaire()
    current_risk = st.session_state.get("risk_profile", "moderate")
    with st.form("goal_form"):
        goal = st.text_input("Goal", value="House down payment")
        c1, c2 = st.columns(2)
        target = c1.number_input(
            "Target (today's dollars)", min_value=1.0, value=60000.0, step=1000.0
        )
        years = c2.number_input(
            "Years from now", min_value=0.5, max_value=80.0, value=5.0, step=0.5
        )
        savings = c1.number_input("Already saved", min_value=0.0, value=10000.0, step=500.0)
        monthly = c2.number_input(
            "Planned monthly saving (0 = tell me what I need)",
            min_value=0.0,
            value=600.0,
            step=50.0,
        )
        risk = st.selectbox("Risk profile", RISK_PROFILES, index=RISK_PROFILES.index(current_risk))
        submitted = st.form_submit_button("Project my goal", type="primary")
    if submitted:
        req = GoalPlanRequest(
            goal=goal,
            target_amount=target,
            years=years,
            current_savings=savings,
            monthly_contribution=monthly or None,
            risk_profile=risk,
        )
        st.session_state.plan = safely(lambda: service.plan_goal(req))
    plan = st.session_state.get("plan")
    if plan is None:
        return
    base = next(s for s in plan.scenarios if s.name == "base")
    cols = st.columns(3)
    cols[0].metric("Target in future dollars", _money(plan.target_nominal))
    cols[1].metric("Needed per month (base case)", _money(base.required_monthly))
    status = {True: "On track ✅", False: "Short ⚠️", None: "—"}[plan.on_track]
    cols[2].metric("At your planned saving", status)
    st.plotly_chart(charts.goal_projection(plan), width="stretch")
    if plan.levers:
        lv = plan.levers
        extra = lv.get("extra_monthly_needed")
        yrs = lv.get("years_needed_at_planned")
        st.info(
            "Ways to close the gap: save about "
            f"{_money(extra)} more per month, or allow about "
            f"{'n/a' if yrs is None else f'{yrs:.1f}'} years, or lower the target to about "
            f"{_money(lv.get('reachable_today_dollars_at_planned'))} (today's dollars)."
        )
    st.dataframe(
        pd.DataFrame(
            {
                "Scenario": s.name.capitalize(),
                "Assumed return": s.annual_return,
                "Needed / month": s.required_monthly,
                "Projected at your plan": s.projected_value,
            }
            for s in plan.scenarios
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Assumed return": st.column_config.NumberColumn(format="percent"),
            "Needed / month": st.column_config.NumberColumn(format="$%.0f"),
            "Projected at your plan": st.column_config.NumberColumn(format="$%.0f"),
        },
    )
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(
        charts.allocation_donut(
            {k: v / 100 for k, v in plan.allocation.items()}, "Illustrative mix"
        ),
        width="stretch",
    )
    c2.caption(plan.allocation_note)
    c2.caption(
        f"Assumes {plan.inflation:.1%} inflation and the fixed returns shown above. Real returns "
        "vary year to year; this is an educational illustration, not a recommendation."
    )


# -- Learn -------------------------------------------------------------------------------------


def learn_tab(service: AssistantService) -> None:
    """Browse knowledge-base articles by category (REQ-UI-06)."""
    st.subheader("Learn")
    c1, c2 = st.columns([1, 2])
    category = c1.selectbox(
        "Category",
        ["all", *service.categories()],
        format_func=lambda c: c.replace("_", " ").title(),
    )
    query = c2.text_input("Search titles", key="learn_query").strip().lower()
    articles = [
        a
        for a in service.list_articles(None if category == "all" else category)
        if query in a.title.lower()
    ]
    st.caption(f"{len(articles)} article(s)")
    for a in articles:
        with st.expander(f"{a.title} · {a.difficulty}"):
            st.markdown(a.body)
            year = f" · tax year {a.tax_year}" if a.tax_year else ""
            st.caption(f"Source: [{a.source_name}]({a.source_url}){year}")
