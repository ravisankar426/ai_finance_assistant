"""Market and News agents over recorded market data (no network)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.exceptions import ModelAPIError, ModelAuthenticationError
from langchain_core.messages import HumanMessage

from src.agents.entities import TickerExtractor, TickerRequest, regex_tickers
from src.agents.market import FOLLOW_UP, MarketAgent
from src.agents.news import NewsAgent
from src.core.config import MarketConfig, get_settings
from src.core.market_analytics import INDEX_ETFS, SECTOR_ETFS
from src.core.models import UserProfile
from src.data.market_service import MarketDataService, build_market_service
from src.data.overview import describe_overview, market_overview
from src.data.providers.fixture_provider import FixtureProvider
from src.workflow.state import AgentInput
from tests.fakes import ScriptedChatModel, seen_text
from tests.kb import offline_retriever


@pytest.fixture(scope="module")
def recorded() -> MarketDataService:
    """Market service backed only by the recorded fixtures."""
    return MarketDataService(
        [FixtureProvider(get_settings().market.fixture_dir)], MarketConfig(rate_limits={})
    )


def _extractor(tickers: list[str] | None = None, overview: bool = False) -> TickerExtractor:
    req = TickerRequest(tickers=tickers or [], wants_market_overview=overview)
    return TickerExtractor(
        ScriptedChatModel(reply=req.model_dump_json()).with_structured_output(TickerRequest)
    )


def _input(question: str, level: str = "beginner") -> AgentInput:
    return {
        "intent": "market",
        "question": question,
        "messages": [HumanMessage(question)],
        "profile": UserProfile(knowledge_level=level),  # type: ignore[arg-type]
        "request_id": "r",
    }


# --- ticker extraction -------------------------------------------------------------------------


def test_extractor_normalizes_and_dedupes() -> None:
    """REQ-MK-01: LLM output is cleaned: upper-case, '$' stripped, invalid shapes dropped."""
    ex = _extractor(["aapl", "$MSFT", "AAPL", "not a ticker"])
    assert ex.extract("compare apple and microsoft").tickers == ["AAPL", "MSFT"]


def test_extractor_falls_back_to_regex_on_llm_failure() -> None:
    """REQ-WF-03 pattern: if the extraction call fails, explicit symbols still work."""
    ex = TickerExtractor(
        ScriptedChatModel(fail=True, fail_with=ModelAPIError).with_structured_output(TickerRequest)
    )
    assert ex.extract("How is $nvda vs MSFT doing? What about the ETF?").tickers == ["NVDA", "MSFT"]


def test_extractor_config_errors_fail_loud() -> None:
    """REQ-LLM-07: a bad key isn't masked by the regex fallback."""
    ex = TickerExtractor(
        ScriptedChatModel(fail=True, fail_with=ModelAuthenticationError).with_structured_output(
            TickerRequest
        )
    )
    with pytest.raises(ModelAuthenticationError):
        ex.extract("AAPL?")


def test_regex_fallback_overview_detection() -> None:
    """REQ-MK-03: market-wide wording without tickers -> overview."""
    assert regex_tickers("how is the market doing?").wants_market_overview
    assert not regex_tickers("How is AAPL doing?").wants_market_overview


# --- overview ----------------------------------------------------------------------------------


def test_overview_from_recorded_data(recorded: MarketDataService) -> None:
    """REQ-MK-03: 4 index ETFs + 11 sector ETFs, sectors sorted best to worst."""
    o = market_overview(recorded)
    assert [r.ticker for r in o.indices] == list(INDEX_ETFS)
    assert {r.ticker for r in o.sectors} == set(SECTOR_ETFS)
    changes = [r.change_pct for r in o.sectors]
    assert changes == sorted(changes, reverse=True)
    assert "Sectors, best to worst" in describe_overview(o)


def test_overview_tolerates_missing_tickers(tmp_path: Any) -> None:
    """REQ-WF-05: tickers without data are listed as unavailable, the rest still shown."""
    import shutil

    src = get_settings().market.fixture_dir
    shutil.copytree(src, tmp_path / "m")
    (tmp_path / "m" / "quote" / "XLK.json").unlink()
    svc = MarketDataService([FixtureProvider(tmp_path / "m")], MarketConfig(rate_limits={}))
    o = market_overview(svc)
    assert o.unavailable == ["XLK"] and len(o.sectors) == 10
    assert "Unavailable right now: XLK" in describe_overview(o)


# --- market agent ------------------------------------------------------------------------------


def test_market_agent_asks_follow_up_without_a_ticker(recorded: MarketDataService) -> None:
    """REQ-WF-10: no ticker and not market-wide -> one targeted question, no data calls."""
    model = ScriptedChatModel()
    agent = MarketAgent(model, _extractor(), recorded, offline_retriever())
    result = agent.run(_input("What's the stock price right now?"))
    assert result.follow_up_question == FOLLOW_UP and result.answer == ""
    assert model.calls == 0


def test_market_agent_explains_computed_facts(recorded: MarketDataService) -> None:
    """REQ-MK-01, REQ-MK-02, REQ-MK-04, REQ-WF-09: computed facts, cited concepts, data footer."""
    model = ScriptedChatModel(reply="AAPL is above its moving averages [1].")
    result = MarketAgent(model, _extractor(["AAPL"]), recorded, offline_retriever()).run(
        _input("How is Apple doing?")
    )
    prompt = seen_text(model)
    assert (
        "AAPL — Apple Inc." in prompt
        and "50-day SMA" in prompt
        and "trend classification" in prompt
    )
    assert "NEVER predict future prices" in prompt
    assert "Market data: recorded" in result.answer  # source label, deterministic
    assert result.data["tickers"] == ["AAPL"] and result.data["snapshots"][0]["ticker"] == "AAPL"
    assert all(c.category in {"markets", "portfolio"} for c in result.citations)


def test_market_agent_overview_includes_market_trend(recorded: MarketDataService) -> None:
    """REQ-MK-03: overview requests also carry SPY's trend, so 'trend' isn't one day's move."""
    model = ScriptedChatModel(reply="Markets mixed today.")
    result = MarketAgent(model, _extractor(overview=True), recorded, offline_retriever()).run(
        _input("How is the market doing?")
    )
    prompt = seen_text(model)
    assert "MARKET OVERVIEW" in prompt and "SPY" in prompt and "trend classification" in prompt
    assert "A single day's change is NOT a trend" in prompt
    assert "overview" in result.data and "SPY" in result.data["tickers"]


def test_market_agent_reports_bad_tickers_without_llm(recorded: MarketDataService) -> None:
    """REQ-MD-07: only unknown tickers -> plain explanation, no LLM call."""
    model = ScriptedChatModel()
    result = MarketAgent(model, _extractor(["ZZQXW"]), recorded, offline_retriever()).run(
        _input("ZZQXW?")
    )
    assert "ZZQXW" in result.answer and model.calls == 0


def test_market_agent_mentions_stale_data(recorded: MarketDataService) -> None:
    """REQ-MK-04: stale values are flagged in the answer."""
    from src.data.models import Quote

    class StaleService:
        def __init__(self, inner: MarketDataService) -> None:
            self.inner = inner

        def get_quote(self, t: str) -> Quote:
            return self.inner.get_quote(t).model_copy(update={"stale": True})

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

    agent = MarketAgent(
        ScriptedChatModel(reply="ok"),
        _extractor(["MSFT"]),
        StaleService(recorded),
        offline_retriever(),
    )  # type: ignore[arg-type]
    assert "from cache" in agent.run(_input("MSFT?")).answer


# --- news agent --------------------------------------------------------------------------------


def _news_agent(
    model: ScriptedChatModel, recorded: MarketDataService, tickers: list[str] | None = None
) -> NewsAgent:
    return NewsAgent(model, _extractor(tickers), recorded, offline_retriever(), lookback_days=7)


def test_news_agent_summarizes_linked_headlines(recorded: MarketDataService) -> None:
    """REQ-NW-01, REQ-NW-02, REQ-NW-03: relevant headlines, each cited headline linked."""
    model = ScriptedChatModel(reply="- Apple rises on AI hopes [1]\n- Smart home push [2]")
    result = _news_agent(model, recorded, ["AAPL"]).run(_input("Any news on Apple?"))
    prompt = seen_text(model)
    assert "ONLY HEADLINES" in prompt and "NEVER predict prices" in prompt
    assert [c.category for c in result.citations[:2]] == ["news", "news"]
    assert all(c.url.startswith("https://") for c in result.citations)
    assert "based on headlines only" in result.answer
    assert result.data["subjects"] == ["AAPL"]


def test_news_agent_market_topic_when_no_ticker(recorded: MarketDataService) -> None:
    """REQ-NW-01: no ticker -> overall market headlines."""
    model = ScriptedChatModel(reply="Markets headline [1].")
    result = _news_agent(model, recorded).run(_input("What's in the financial news today?"))
    assert result.data["subjects"] == ["stock market"] and result.citations


def test_news_agent_says_so_when_nothing_found(recorded: MarketDataService) -> None:
    """REQ-NW-04: no headlines -> an honest message, no LLM call."""
    model = ScriptedChatModel()
    result = _news_agent(model, recorded, ["ZZQXW"]).run(_input("news on ZZQXW?"))
    assert "couldn't find recent news" in result.answer and model.calls == 0


def test_build_market_service_fixture_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-DEL-02: MARKET__PROVIDERS='["fixture"]' gives an offline demo."""
    from src.core.config import Settings

    monkeypatch.setenv("MARKET__PROVIDERS", '["fixture"]')
    svc = build_market_service(Settings(_env_file=None))  # type: ignore[call-arg]
    assert [p.name for p in svc.providers] == ["fixture"]
    assert svc.get_quote("NVDA").source.startswith("recorded")
