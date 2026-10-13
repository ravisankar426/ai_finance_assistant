"""Portfolio and Goals agents, CSV holdings, router portfolio guard, and memory across turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import ModelAPIError
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.goals import ASK_GOAL, ASK_RISK, GoalRequest, GoalsAgent
from src.agents.portfolio import FOLLOW_UP, HoldingsRequest, PortfolioAgent, regex_holdings
from src.core.config import MarketConfig, PlanningConfig, PortfolioConfig, get_settings
from src.core.holdings_io import parse_holdings_csv
from src.core.models import Holding, UserProfile
from src.data.market_service import MarketDataService
from src.data.providers.fixture_provider import FixtureProvider
from src.workflow.router import apply_portfolio_guard
from src.workflow.state import AgentInput
from tests.fakes import ScriptedChatModel, seen_text
from tests.kb import offline_retriever

SAMPLES = Path(get_settings().market.fixture_dir).parents[1] / "sample_portfolios"


@pytest.fixture(scope="module")
def market() -> MarketDataService:
    return MarketDataService(
        [FixtureProvider(get_settings().market.fixture_dir)], MarketConfig(rate_limits={})
    )


def _input(
    text: str,
    *,
    portfolio: list[Holding] | None = None,
    risk: str | None = None,
    history: list[Any] | None = None,
) -> AgentInput:
    return {
        "intent": "portfolio",
        "question": text,
        "messages": [*(history or []), HumanMessage(text)],
        "profile": UserProfile(risk_tolerance=risk),  # type: ignore[arg-type]
        "request_id": "r",
        "portfolio": portfolio,
    }


def _structured(model_cls: type, **fields: Any) -> Any:
    return ScriptedChatModel(reply=model_cls(**fields).model_dump_json()).with_structured_output(
        model_cls
    )


# --- CSV & regex parsing ------------------------------------------------------------------------


def test_sample_portfolios_parse() -> None:
    """REQ-PF-01, REQ-DEL-02: shipped sample CSVs are valid; the error sample reports bad rows."""
    good = {p.name: parse_holdings_csv(p.read_text()) for p in SAMPLES.glob("*.csv")}
    assert good["concentrated_tech.csv"][0][0].ticker == "NVDA"
    assert good["balanced_funds.csv"][1] == []
    holdings, errors = good["with_errors.csv"]
    assert [h.ticker for h in holdings] == ["AAPL", "ZZQXW"] and len(errors) == 2


@pytest.mark.parametrize(
    ("text", "problem"),
    [("", "empty"), ("name,amount\nAAPL,3", "needs 'ticker' and 'quantity'")],
)
def test_csv_structural_errors(text: str, problem: str) -> None:
    """REQ-PF-01: unusable files get a clear message."""
    assert problem in parse_holdings_csv(text)[1][0]


def test_csv_aliases() -> None:
    """REQ-PF-01: 'symbol'/'shares' headers and '$' prefixes are accepted."""
    holdings, errors = parse_holdings_csv("Symbol,Shares\n$msft,3\n")
    assert errors == [] and holdings == [Holding(ticker="MSFT", quantity=3)]


def test_regex_holdings_fallback() -> None:
    """REQ-PF-01: '10 AAPL' and '5 shares of MSFT' parse without the LLM."""
    assert regex_holdings("I have 10 AAPL and 5 shares of MSFT") == [
        Holding(ticker="AAPL", quantity=10),
        Holding(ticker="MSFT", quantity=5),
    ]


# --- portfolio agent ---------------------------------------------------------------------------


def _portfolio_agent(
    market: MarketDataService,
    holdings: list[Holding] | None = None,
    reply: str = "Your portfolio is concentrated [1].",
) -> tuple[PortfolioAgent, ScriptedChatModel]:
    model = ScriptedChatModel(reply=reply)
    extractor = _structured(HoldingsRequest, holdings=holdings or [])
    return PortfolioAgent(model, extractor, market, offline_retriever(), PortfolioConfig()), model


def test_portfolio_agent_asks_when_no_holdings(market: MarketDataService) -> None:
    """REQ-WF-10: nothing to analyze -> ask for holdings, no LLM explanation call."""
    agent, model = _portfolio_agent(market)
    result = agent.run(_input("Analyze my portfolio"))
    assert result.follow_up_question == FOLLOW_UP and model.calls == 0


def test_portfolio_agent_analyzes_and_remembers(market: MarketDataService) -> None:
    """REQ-PF-01, REQ-PF-04, REQ-WF-09: parse, analyze, educational advice only, remember."""
    holdings = [
        Holding(ticker="NVDA", quantity=100),
        Holding(ticker="AAPL", quantity=50, cost_basis=150),
    ]
    agent, model = _portfolio_agent(market, holdings)
    result = agent.run(_input("I own 100 NVDA and 50 AAPL at $150"))
    prompt = seen_text(model)
    assert "FLAG:" in prompt and "NEVER tell the user to buy, sell, or hold" in prompt
    assert result.state_updates == {"portfolio": holdings}
    assert result.data["analysis"]["holdings"][0]["ticker"] == "NVDA"
    assert all(c.category == "portfolio" for c in result.citations)
    assert "Metrics use past data" in result.answer


def test_portfolio_agent_uses_remembered_holdings(market: MarketDataService) -> None:
    """REQ-WF-04: a follow-up without holdings analyzes the remembered portfolio."""
    agent, _ = _portfolio_agent(market)
    result = agent.run(
        _input("How diversified is it?", portfolio=[Holding(ticker="VTI", quantity=5)])
    )
    assert result.data["analysis"]["holdings"][0]["ticker"] == "VTI"
    assert result.state_updates == {}  # nothing new to remember


def test_portfolio_agent_all_invalid(market: MarketDataService) -> None:
    """REQ-PF-05: only unknown tickers -> clear message, no LLM call."""
    agent, model = _portfolio_agent(market, [Holding(ticker="ZZQXW", quantity=1)])
    result = agent.run(_input("I own 1 ZZQXW"))
    assert "ZZQXW" in result.answer and model.calls == 0


def test_portfolio_agent_regex_fallback(market: MarketDataService) -> None:
    """REQ-WF-03 pattern: extraction failure falls back to regex parsing."""
    failing = ScriptedChatModel(fail=True, fail_with=ModelAPIError).with_structured_output(
        HoldingsRequest
    )
    agent = PortfolioAgent(
        ScriptedChatModel(reply="ok"), failing, market, offline_retriever(), PortfolioConfig()
    )
    result = agent.run(_input("I have 10 AAPL"))
    assert result.state_updates["portfolio"] == [Holding(ticker="AAPL", quantity=10)]


# --- goals agent -------------------------------------------------------------------------------


def _goals_agent(**fields: Any) -> tuple[GoalsAgent, ScriptedChatModel]:
    model = ScriptedChatModel(reply="Here's your plan [1].")
    return GoalsAgent(
        model, _structured(GoalRequest, **fields), offline_retriever(), PlanningConfig()
    ), model


def test_goals_agent_asks_for_goal_then_risk() -> None:
    """REQ-WF-10, REQ-GP-01: missing amount/horizon -> ask; missing risk profile -> ask."""
    agent, model = _goals_agent(goal="house")
    assert agent.run(_input("I want a house")).follow_up_question == ASK_GOAL
    agent, model = _goals_agent(goal="house", target_amount=60000, years=5)
    assert agent.run(_input("$60k in 5 years")).follow_up_question == ASK_RISK
    assert model.calls == 0


def test_goals_agent_plans_and_remembers_risk() -> None:
    """REQ-GP-01..05, REQ-WF-08: computed plan explained; stated risk saved to the profile."""
    agent, model = _goals_agent(
        goal="house",
        target_amount=60000,
        years=5,
        current_savings=10000,
        monthly_contribution=600,
        risk_tolerance="moderate",
    )
    result = agent.run(_input("$60k in 5 years, $10k saved, $600/month, moderate"))
    prompt = seen_text(model)
    assert "goal at risk" in prompt and "Illustrative model allocation" in prompt
    assert "ASSUMPTION" not in prompt  # savings were stated
    assert result.state_updates["profile"].risk_tolerance == "moderate"
    assert "house" in result.state_updates["profile"].goals
    assert result.data["plan"]["on_track"] is False
    assert "illustrative assumptions" in result.answer


def test_goals_agent_uses_profile_risk_and_flags_assumed_savings() -> None:
    """REQ-WF-08: a remembered risk profile is used; unstated savings are flagged as assumed $0."""
    agent, model = _goals_agent(goal="retirement", target_amount=1_000_000, years=30)
    result = agent.run(_input("I want $1M for retirement in 30 years", risk="aggressive"))
    assert result.data["plan"]["risk_profile"] == "aggressive"
    assert "ASSUMPTION: starting savings not stated" in seen_text(model)


def test_goals_agent_extraction_failure_asks() -> None:
    """REQ-WF-10: if extraction fails, ask rather than guess."""
    failing = ScriptedChatModel(fail=True, fail_with=ModelAPIError).with_structured_output(
        GoalRequest
    )
    agent = GoalsAgent(ScriptedChatModel(), failing, offline_retriever(), PlanningConfig())
    history = [HumanMessage("earlier"), AIMessage("reply")]
    assert agent.run(_input("plan it", history=history)).follow_up_question == ASK_GOAL


# --- router guard ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intents", "text", "has", "expected"),
    [
        (["qa"], "How diversified is it really?", True, ["portfolio"]),
        (["market"], "is my portfolio too risky vs the market?", True, ["market", "portfolio"]),
        (["qa"], "What is beta?", True, ["qa"]),  # concept question, no reference
        (["qa"], "How diversified is it?", False, ["qa"]),  # nothing remembered
        (["portfolio"], "is it risky?", True, ["portfolio"]),
    ],
)
def test_portfolio_guard(intents: list[str], text: str, has: bool, expected: list[str]) -> None:
    """REQ-WF-04: follow-ups about a remembered portfolio reach the Portfolio agent."""
    assert apply_portfolio_guard(intents, text, has_portfolio=has)[0] == expected  # type: ignore[arg-type]
