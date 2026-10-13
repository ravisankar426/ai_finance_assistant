"""Goal engine: time-value-of-money math, scenarios, allocation, questionnaire."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.config import PlanningConfig
from src.core.goal_engine import (
    QUESTIONNAIRE,
    describe_plan,
    future_value,
    model_allocation,
    months_to_reach,
    plan_goal,
    required_monthly,
    risk_profile_from_answers,
)

CFG = PlanningConfig()


def test_textbook_future_value() -> None:
    """REQ-GP-02: $1,000 at 5% for 10 years compounds to $1,628.89."""
    assert future_value(1000, 0, 0.05, 120) == pytest.approx(1628.89, abs=0.01)
    assert future_value(1000, 100, 0.0, 12) == 2200  # zero-rate edge case


def test_required_monthly_edge_cases() -> None:
    """REQ-GP-02: already funded -> 0; zero months -> the gap; zero rate -> straight line."""
    assert required_monthly(1000, 5000, 0.05, 60) == 0
    assert required_monthly(1000, 400, 0.05, 0) == 600
    assert required_monthly(1200, 0, 0.0, 12) == pytest.approx(100)


@given(
    st.floats(min_value=1_000, max_value=2_000_000),
    st.floats(min_value=0, max_value=500_000),
    st.floats(min_value=0.0, max_value=0.12),
    st.integers(min_value=1, max_value=480),
)
def test_required_monthly_reaches_target(
    target: float, present: float, rate: float, months: int
) -> None:
    """REQ-GP-02 (property): saving the required amount lands exactly on the target."""
    need = required_monthly(target, present, rate, months)
    fv = future_value(present, need, rate, months)
    assert need >= 0
    assert fv >= target - 0.01 * max(1.0, target / 1e5)


@given(st.floats(min_value=0, max_value=0.1), st.floats(min_value=0.001, max_value=0.05))
def test_higher_return_never_lowers_future_value(rate: float, bump: float) -> None:
    """REQ-GP-02 (property): more return -> more money."""
    assert future_value(1000, 100, rate + bump, 120) >= future_value(1000, 100, rate, 120)


def test_months_to_reach() -> None:
    """REQ-GP-05: time needed at a fixed contribution; None if never within the cap."""
    assert months_to_reach(1200, 0, 100, 0.0) == 12
    assert months_to_reach(100, 200, 0, 0.05) == 0
    assert months_to_reach(1e9, 0, 1, 0.0, max_months=100) is None


def test_plan_off_track_with_levers() -> None:
    """REQ-GP-02, REQ-GP-03, REQ-GP-05: inflation-adjusted target, scenarios, gap levers."""
    p = plan_goal(
        goal="house",
        target_today=60_000,
        years=5,
        current_savings=10_000,
        planned_monthly=600,
        risk_profile="moderate",
        cfg=CFG,
    )
    assert p.target_nominal == pytest.approx(60_000 * 1.03**5)
    needs = [s.required_monthly for s in p.scenarios]
    assert needs == sorted(needs, reverse=True)  # pessimistic needs the most
    assert p.on_track is False
    assert p.levers["extra_monthly_needed"] == pytest.approx(p.scenarios[1].required_monthly - 600)
    assert (
        p.levers["years_needed_at_planned"] is not None and p.levers["years_needed_at_planned"] > 5
    )
    assert len(p.scenarios[1].yearly_balances) == 5
    text = describe_plan(p)
    assert (
        "goal at risk" in text
        and "Levers to close the gap" in text
        and "in today's dollars" in text
    )


def test_plan_on_track_and_without_planned_contribution() -> None:
    """REQ-GP-02: generous saving is on track; no stated budget -> required amounts only."""
    rich = plan_goal(
        goal="car",
        target_today=20_000,
        years=4,
        current_savings=5_000,
        planned_monthly=2_000,
        risk_profile="conservative",
        cfg=CFG,
    )
    assert rich.on_track is True and rich.levers == {}
    open_ended = plan_goal(
        goal="trip",
        target_today=5_000,
        years=2,
        current_savings=0,
        planned_monthly=None,
        risk_profile="aggressive",
        cfg=CFG,
    )
    assert open_ended.on_track is None
    assert all(s.reaches_goal for s in open_ended.scenarios)  # saving the required amount


def test_plan_rejects_nonsense() -> None:
    """REQ-GP-02: non-positive targets/horizons are errors."""
    with pytest.raises(ValueError):
        plan_goal(
            goal="x",
            target_today=0,
            years=5,
            current_savings=0,
            planned_monthly=None,
            risk_profile="moderate",
            cfg=CFG,
        )


@pytest.mark.parametrize(
    ("profile", "years", "stocks"),
    [
        ("aggressive", 30, 85),
        ("aggressive", 5, 70),
        ("aggressive", 2, 10),
        ("conservative", 20, 30),
    ],
)
def test_model_allocation_by_profile_and_horizon(profile: str, years: float, stocks: int) -> None:
    """REQ-GP-04: allocation depends on risk profile and shrinks stocks for short horizons."""
    mix, note = model_allocation(profile, years, CFG)  # type: ignore[arg-type]
    assert mix["stocks"] == stocks and sum(mix.values()) == 100 and note


def _answers(**overrides: str) -> dict[str, str]:
    base = {
        "drop": "Wait it out",
        "horizon": "In more than 10 years",
        "goal": "A balance of growth and stability",
        "income": "Fairly stable",
        "experience": "Some",
    }
    return base | overrides


def test_questionnaire_scoring_and_caps() -> None:
    """REQ-GP-01: score bands, short-horizon override, and 'sell in a drop' cap."""
    assert risk_profile_from_answers(_answers()) == "moderate"
    bold = _answers(
        drop="Invest more",
        goal="Maximizing long-term growth",
        income="Very stable",
        experience="A lot",
    )
    assert risk_profile_from_answers(bold) == "aggressive"
    assert risk_profile_from_answers(bold | {"horizon": "Within 3 years"}) == "conservative"
    assert risk_profile_from_answers(bold | {"drop": "Sell to avoid further losses"}) == "moderate"
    timid = _answers(
        drop="Sell to avoid further losses",
        goal="Protecting what I have",
        income="Not very stable",
        experience="None",
        horizon="In 3-10 years",
    )
    assert risk_profile_from_answers(timid) == "conservative"


def test_questionnaire_validates_answers() -> None:
    """REQ-GP-01: every question needs a valid option."""
    assert len(QUESTIONNAIRE) == 5
    with pytest.raises(ValueError, match="drop"):
        risk_profile_from_answers(_answers(drop="Panic"))
