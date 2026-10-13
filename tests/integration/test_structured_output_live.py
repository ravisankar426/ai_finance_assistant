"""Structured-output contract on both providers (REQ-LLM-06, REQ-NFR-08). Live: needs keys.

Every Pydantic schema the app asks an LLM to fill is sent to the primary (OpenAI) model and to
the fallback (Gemini) model directly. If a provider changes how it handles a schema, this fails
before users see a broken router or extractor after a fallback.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from src.agents.entities import TickerRequest
from src.agents.goals import GoalRequest
from src.agents.portfolio import HoldingsRequest
from src.core.config import ModelConfig, get_settings
from src.core.llm import _build_model
from src.evals.judge import JudgeScore
from src.workflow.router import RouteDecision

pytestmark = pytest.mark.live

CONTRACTS: list[tuple[type[BaseModel], str, Any]] = [
    (
        RouteDecision,
        "Classify the finance question into intents (qa, portfolio, market, goals, news, tax, "
        "out_of_scope) and restate it as a standalone question: 'What is NVDA trading at?'",
        # shape only: classification quality needs the router's prompt (router accuracy bench)
        lambda r: len(r.intents) >= 1 and r.standalone_question,
    ),
    (
        TickerRequest,
        "Extract the stock tickers mentioned: 'How are Apple and Microsoft doing?'",
        lambda r: {"AAPL", "MSFT"} <= {t.upper() for t in r.tickers},
    ),
    (
        HoldingsRequest,
        "Extract holdings (ticker, quantity, optional cost basis): 'I have 10 AAPL bought at "
        "$150 and 5 shares of VTI'",
        lambda r: {h.ticker.upper() for h in r.holdings} == {"AAPL", "VTI"},
    ),
    (
        GoalRequest,
        "Extract the savings goal: 'I want $60,000 for a house in 5 years, I have $10,000 saved'",
        lambda r: r.target_amount == 60000 and r.years == 5,
    ),
    (
        JudgeScore,
        "Grade this answer 1-5 on groundedness, citation_faithfulness, compliance, clarity. "
        "Question: What is an index fund? Answer: An index fund tracks a market index [1].",
        lambda r: 1 <= r.clarity <= 5,
    ),
]


def _providers() -> list[tuple[str, ModelConfig]]:
    s = get_settings()
    out = [("primary", s.llm.roles["router"])]
    if s.llm.fallback is not None:
        out.append(("fallback", s.llm.fallback))
    return out


@pytest.mark.parametrize(("label", "cfg"), _providers(), ids=lambda x: str(x)[:20])
@pytest.mark.parametrize(
    ("schema", "prompt", "check"), CONTRACTS, ids=[c[0].__name__ for c in CONTRACTS]
)
def test_schema_contract(
    label: str, cfg: ModelConfig, schema: type[BaseModel], prompt: str, check: Any
) -> None:
    """REQ-LLM-06: each provider returns a valid instance of each schema."""
    model = _build_model(cfg, role=f"contract-{label}", settings=get_settings())
    result = model.with_structured_output(schema).invoke(prompt)
    assert isinstance(result, schema), f"{cfg.provider}:{cfg.model} returned {type(result)}"
    assert check(result), f"{cfg.provider}:{cfg.model} -> {result!r}"
