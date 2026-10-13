"""Goal planning engine (REQ-GP-01..05) — deterministic math the Goals agent explains.

Conventions:
- Targets are entered in **today's dollars**; the nominal target grows with inflation (GP-03).
- Returns are annual, converted to the equivalent monthly rate: r_m = (1 + r)^(1/12) - 1.
- Contributions are made at the end of each month.
- Scenario returns and model allocations come from config and are illustrative, not forecasts.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from src.core.config import PlanningConfig

RiskProfile = Literal["conservative", "moderate", "aggressive"]
SCENARIOS = ("pessimistic", "base", "optimistic")

# --- risk questionnaire (GP-01) -----------------------------------------------------------------

QUESTIONNAIRE: list[dict[str, object]] = [
    {
        "id": "drop",
        "question": "If your investments fell 20% in a few months, what would you most likely do?",
        "options": {"Sell to avoid further losses": 0, "Wait it out": 1, "Invest more": 2},
    },
    {
        "id": "horizon",
        "question": "When will you need most of this money?",
        "options": {"Within 3 years": 0, "In 3-10 years": 1, "In more than 10 years": 2},
    },
    {
        "id": "goal",
        "question": "Which matters more to you?",
        "options": {
            "Protecting what I have": 0,
            "A balance of growth and stability": 1,
            "Maximizing long-term growth": 2,
        },
    },
    {
        "id": "income",
        "question": "How stable is your income and emergency savings?",
        "options": {"Not very stable": 0, "Fairly stable": 1, "Very stable": 2},
    },
    {
        "id": "experience",
        "question": "How much investing experience do you have?",
        "options": {"None": 0, "Some": 1, "A lot": 2},
    },
]


def risk_profile_from_answers(answers: dict[str, str]) -> RiskProfile:
    """Score the questionnaire: 0-3 conservative, 4-7 moderate, 8-10 aggressive.

    A short horizon or a "sell after a drop" answer caps the result at moderate/conservative:
    capacity for risk matters more than appetite.
    """
    score = 0
    for q in QUESTIONNAIRE:
        options: dict[str, int] = q["options"]  # type: ignore[assignment]
        answer = answers.get(str(q["id"]))
        if answer not in options:
            raise ValueError(f"missing or invalid answer for {q['id']!r}")
        score += options[answer]
    profile: RiskProfile = (
        "conservative" if score <= 3 else "moderate" if score <= 7 else "aggressive"
    )
    if answers["horizon"] == "Within 3 years":
        return "conservative"  # money needed soon can't ride out a downturn
    if answers["drop"] == "Sell to avoid further losses" and profile == "aggressive":
        return "moderate"  # stated behavior in a drop caps the profile
    return profile


# --- time value of money ------------------------------------------------------------------------


def monthly_rate(annual: float) -> float:
    """Return the monthly rate equivalent to an annual rate."""
    return float((1 + annual) ** (1 / 12) - 1)


def future_value(present: float, monthly: float, annual_return: float, months: int) -> float:
    """Value after ``months`` of growth plus end-of-month contributions."""
    r = monthly_rate(annual_return)
    if r == 0:
        return present + monthly * months
    growth = (1 + r) ** months
    return present * growth + monthly * (growth - 1) / r


def required_monthly(target: float, present: float, annual_return: float, months: int) -> float:
    """Monthly contribution needed to reach ``target`` (0 if savings alone get there)."""
    if months <= 0:
        return max(target - present, 0.0)
    r = monthly_rate(annual_return)
    growth = (1 + r) ** months
    gap = target - present * growth
    if gap <= 0:
        return 0.0
    return gap / months if r == 0 else gap * r / (growth - 1)


def months_to_reach(
    target: float, present: float, monthly: float, annual_return: float, max_months: int = 1200
) -> int | None:
    """Months until the balance reaches ``target``; None if not within ``max_months``."""
    if present >= target:
        return 0
    r = monthly_rate(annual_return)
    balance = present
    for month in range(1, max_months + 1):
        balance = balance * (1 + r) + monthly
        if balance >= target:
            return month
    return None


# --- plan ---------------------------------------------------------------------------------------


class ScenarioResult(BaseModel):
    """One return scenario."""

    name: str
    annual_return: float
    required_monthly: float
    projected_value: float  # at the planned contribution
    reaches_goal: bool
    yearly_balances: list[float]  # end of each year, for charts


class GoalPlan(BaseModel):
    """Everything the Goals agent may say about a goal (GP-02..05)."""

    goal: str
    target_today: float
    target_nominal: float
    years: float
    months: int
    current_savings: float
    planned_monthly: float | None
    risk_profile: RiskProfile
    inflation: float
    scenarios: list[ScenarioResult]
    allocation: dict[str, int]
    allocation_note: str
    on_track: bool | None
    levers: dict[str, float | None] = Field(default_factory=dict)


def model_allocation(
    profile: RiskProfile, years: float, cfg: PlanningConfig
) -> tuple[dict[str, int], str]:
    """Illustrative stock/bond/cash mix for the profile, adjusted for horizon (GP-04)."""
    mix = dict(cfg.allocations[profile])
    if years < cfg.short_horizon_years:
        return {
            "stocks": min(mix["stocks"], 10),
            "bonds": 50,
            "cash": 100 - 50 - min(mix["stocks"], 10),
        }, (
            f"Short horizon (< {cfg.short_horizon_years} years): money you need soon is usually "
            "kept mostly in cash and short-term bonds, whatever your risk tolerance."
        )
    if years < cfg.medium_horizon_years:
        shift = min(15, mix["stocks"])
        mix["stocks"] -= shift
        mix["bonds"] += shift
        return mix, (
            f"Medium horizon (< {cfg.medium_horizon_years} years): tilted 15 points from stocks to "
            "bonds compared with the long-term mix for this risk profile."
        )
    return mix, "Long horizon: the standard illustrative mix for this risk profile."


def plan_goal(
    *,
    goal: str,
    target_today: float,
    years: float,
    current_savings: float,
    planned_monthly: float | None,
    risk_profile: RiskProfile,
    cfg: PlanningConfig,
) -> GoalPlan:
    """Compute required saving, scenario projections, allocation, and gap levers."""
    if target_today <= 0 or years <= 0:
        raise ValueError("target and years must be positive")
    months = max(1, round(years * 12))
    target_nominal = target_today * (1 + cfg.inflation) ** years
    rates = cfg.returns[risk_profile]
    scenarios: list[ScenarioResult] = []
    for name in SCENARIOS:
        r = getattr(rates, name)
        need = required_monthly(target_nominal, current_savings, r, months)
        contribution = planned_monthly if planned_monthly is not None else need
        balances = [
            future_value(current_savings, contribution, r, min(12 * y, months))
            for y in range(1, math.ceil(months / 12) + 1)
        ]
        projected = future_value(current_savings, contribution, r, months)
        scenarios.append(
            ScenarioResult(
                name=name,
                annual_return=r,
                required_monthly=need,
                projected_value=projected,
                reaches_goal=projected >= target_nominal - 0.01,
                yearly_balances=balances,
            )
        )
    base = scenarios[1]
    allocation, note = model_allocation(risk_profile, years, cfg)
    levers: dict[str, float | None] = {}
    on_track = None
    if planned_monthly is not None:
        on_track = base.reaches_goal
        if not on_track:  # GP-05: what would close the gap
            needed = months_to_reach(
                target_nominal, current_savings, planned_monthly, base.annual_return
            )
            levers = {
                "required_monthly_base": base.required_monthly,
                "extra_monthly_needed": base.required_monthly - planned_monthly,
                "years_needed_at_planned": None if needed is None else needed / 12,
                "reachable_today_dollars_at_planned": base.projected_value
                / (1 + cfg.inflation) ** years,
            }
    return GoalPlan(
        goal=goal,
        target_today=target_today,
        target_nominal=target_nominal,
        years=years,
        months=months,
        current_savings=current_savings,
        planned_monthly=planned_monthly,
        risk_profile=risk_profile,
        inflation=cfg.inflation,
        scenarios=scenarios,
        allocation=allocation,
        allocation_note=note,
        on_track=on_track,
        levers=levers,
    )


def describe_plan(p: GoalPlan) -> str:
    """Fact block for the LLM (it must not compute new numbers)."""
    lines = [
        f"GOAL: {p.goal} — ${p.target_today:,.0f} in today's dollars in {p.years:g} years "
        f"(≈ ${p.target_nominal:,.0f} in future dollars at {p.inflation:.0%} inflation)",
        f"Starting savings ${p.current_savings:,.0f}; risk profile {p.risk_profile}",
        "Required monthly saving by scenario (illustrative annual returns):",
    ]
    for s in p.scenarios:
        lines.append(
            f"  {s.name}: {s.annual_return:.1%} return -> ${s.required_monthly:,.0f}/month"
        )
    if p.planned_monthly is not None:
        lines.append(f"Planned saving ${p.planned_monthly:,.0f}/month projects to:")
        for s in p.scenarios:
            verdict = "reaches" if s.reaches_goal else "falls short of"
            lines.append(f"  {s.name}: ${s.projected_value:,.0f} ({verdict} the goal)")
        lines.append(
            f"On track in the base scenario: {'yes' if p.on_track else 'NO — goal at risk'}"
        )
    if p.levers:
        lv = p.levers
        yrs = lv.get("years_needed_at_planned")
        lines.append(
            "Levers to close the gap (base scenario): save "
            f"${lv['extra_monthly_needed']:,.0f} more per month "
            f"(${lv['required_monthly_base']:,.0f} total); or keep the plan and allow "
            f"{'more than 100' if yrs is None else f'{yrs:.1f}'} years; or keep the plan and "
            f"aim for about ${lv['reachable_today_dollars_at_planned']:,.0f} in today's dollars"
        )
    mix = ", ".join(f"{k} {v}%" for k, v in p.allocation.items())
    lines.append(
        f"Illustrative model allocation (educational, not a recommendation): {mix}. "
        f"{p.allocation_note}"
    )
    return "\n".join(lines)
