"""A real AssistantService wired to fakes: scripted models, recorded market data, offline index."""

from __future__ import annotations

from src.agents.placeholder import PlaceholderAgent
from src.agents.qa import QAAgent
from src.core.config import MarketConfig, Settings, get_settings
from src.core.models import AGENT_NAMES
from src.core.services import AssistantService
from src.data.market_service import MarketDataService
from src.data.providers.fixture_provider import FixtureProvider
from src.rag.knowledge_base import load_articles
from src.workflow.graph import build_graph
from src.workflow.router import RouteDecision, Router
from tests.fakes import ScriptedChatModel, route_json
from tests.kb import offline_retriever

QA_REPLY = "Index funds track an index [1]."


def fake_service(settings: Settings, intents: list[str] | None = None) -> AssistantService:
    """Q&A answers from a scripted model; every other agent is a placeholder."""
    market = MarketDataService(
        [FixtureProvider(get_settings().market.fixture_dir)], MarketConfig(rate_limits={})
    )
    graph = build_graph(
        router=Router(
            ScriptedChatModel(reply=route_json(intents or ["qa"])).with_structured_output(
                RouteDecision
            )
        ),
        agents={
            **{n: PlaceholderAgent(n) for n in AGENT_NAMES},
            "qa": QAAgent(ScriptedChatModel(reply=QA_REPLY), offline_retriever()),
        },
    )
    articles = load_articles(settings.rag.knowledge_base_dir)
    return AssistantService(graph, market, settings, articles, offline_retriever())
