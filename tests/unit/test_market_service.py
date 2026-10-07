from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from structlog.testing import capture_logs

from src.core.config import MarketConfig, RateLimit, Settings
from src.core.errors import ConfigurationError, InvalidTickerError, MarketDataUnavailableError
from src.data.market_service import MarketDataService, build_market_service
from src.data.models import AssetProfile, PriceBar, PriceHistory, Quote
from src.data.providers.base import (
    NoDataError,
    TransientProviderError,
    UnsupportedOperationError,
    normalize_ticker,
)
from tests.unit.test_resilience import FakeClock

NOW = datetime(2026, 10, 7, 20, 0, tzinfo=UTC)


class FakeProvider:
    """Scriptable provider: ``behavior`` is 'ok', 'down', 'no_data', 'unsupported', 'config'."""

    def __init__(self, name: str, behavior: str = "ok", price: float = 100.0) -> None:
        self.name = name
        self.behavior = behavior
        self.price = price
        self.calls = 0

    def _maybe_fail(self) -> None:
        self.calls += 1
        errors: dict[str, Exception] = {
            "down": TransientProviderError("down"),
            "no_data": NoDataError("none"),
            "unsupported": UnsupportedOperationError("premium"),
            "config": ConfigurationError("bad key"),
        }
        if self.behavior in errors:
            raise errors[self.behavior]

    def get_quote(self, ticker: str) -> Quote:
        self._maybe_fail()
        return Quote(ticker=ticker, price=self.price, as_of=NOW, source=self.name)

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        self._maybe_fail()
        bars = [PriceBar(day=NOW.date(), close=self.price)]
        return PriceHistory(ticker=ticker, bars=bars, as_of=NOW, source=self.name)

    def get_profile(self, ticker: str) -> AssetProfile:
        self._maybe_fail()
        return AssetProfile(ticker=ticker, name="X Corp", as_of=NOW, source=self.name)


def _service(
    *providers: FakeProvider, clock: FakeClock | None = None, **cfg: Any
) -> MarketDataService:
    config = MarketConfig(
        retry_attempts=2,
        breaker_failure_threshold=2,
        rate_limits={},
        **cfg,
    )
    return MarketDataService(
        list(providers), config, clock=clock or FakeClock(), sleep=lambda _: None
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("aapl", "AAPL"), (" $msft ", "MSFT"), ("brk.b", "BRK-B"), ("BRK-B", "BRK-B")],
)
def test_normalize_ticker(raw: str, expected: str) -> None:
    """REQ-MD-07: tickers are normalized before any provider call."""
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "TOOLONG", "AAPL;DROP", "12345", "A B"])
def test_invalid_ticker_format_rejected_without_calls(raw: str) -> None:
    """REQ-MD-07: malformed tickers never reach a provider."""
    primary = FakeProvider("p")
    with pytest.raises(InvalidTickerError):
        _service(primary).get_quote(raw)
    assert primary.calls == 0


def test_primary_used_and_cached() -> None:
    """REQ-MD-01, REQ-MD-02: the first provider answers; a second call within TTL is cached."""
    primary, backup = FakeProvider("p"), FakeProvider("b")
    svc = _service(primary, backup)
    assert svc.get_quote("aapl").source == "p"
    assert svc.get_quote("AAPL").source == "p"
    assert primary.calls == 1 and backup.calls == 0


def test_cache_expires_after_ttl() -> None:
    """REQ-MD-02: after the TTL a fresh fetch happens."""
    clock = FakeClock()
    primary = FakeProvider("p")
    svc = _service(primary, clock=clock, ttl_quote_s=60)
    svc.get_quote("AAPL")
    clock.advance(61)
    svc.get_quote("AAPL")
    assert primary.calls == 2


def test_falls_back_to_next_provider_on_failure() -> None:
    """REQ-MD-01, REQ-MD-04: a provider failing after retries hands over to the next."""
    primary, backup = FakeProvider("p", "down"), FakeProvider("b", price=99)
    with capture_logs() as logs:
        quote = _service(primary, backup).get_quote("AAPL")
    assert quote.source == "b" and quote.price == 99
    assert primary.calls == 2  # retried once (retry_attempts=2)
    assert any(e["event"] == "market_fetch" and e["fallback"] for e in logs)


def test_breaker_skips_provider_after_threshold() -> None:
    """REQ-MD-05: after N failed requests the provider isn't called until the cool-down ends."""
    clock = FakeClock()
    primary, backup = FakeProvider("p", "down"), FakeProvider("b")
    svc = _service(primary, backup, clock=clock, breaker_reset_s=60, ttl_quote_s=1)
    svc.get_quote("AAA")
    svc.get_quote("BBB")  # second failed request -> breaker opens (threshold 2)
    calls_when_open = primary.calls
    svc.get_quote("CCC")
    assert primary.calls == calls_when_open  # skipped while open
    clock.advance(61)
    primary.behavior = "ok"
    assert svc.get_quote("DDD").source == "p"  # half-open probe succeeds
    assert svc.breakers["p"].state == "closed"


def test_local_rate_limit_skips_to_next_provider() -> None:
    """REQ-MD-03: a provider at its client-side limit is skipped, not waited on."""
    primary, backup = FakeProvider("p"), FakeProvider("b")
    config = MarketConfig(rate_limits={"p": [RateLimit(max_calls=1, period_s=60)]}, ttl_quote_s=1)
    clock = FakeClock()
    svc = MarketDataService([primary, backup], config, clock=clock, sleep=lambda _: None)
    assert svc.get_quote("AAA").source == "p"
    assert svc.get_quote("BBB").source == "b"


def test_unsupported_and_no_data_move_on_without_breaker_penalty() -> None:
    """REQ-MD-05: 'unsupported' / 'no data' aren't the provider failing — no breaker count."""
    primary, backup = FakeProvider("p", "unsupported"), FakeProvider("b")
    svc = _service(primary, backup)
    assert svc.get_profile("AAPL").source == "b"
    assert svc.breakers["p"].failures == 0


def test_stale_cache_served_when_all_providers_fail() -> None:
    """REQ-MD-06, REQ-MK-04: everything down -> last known value, flagged stale."""
    clock = FakeClock()
    primary = FakeProvider("p")
    svc = _service(primary, clock=clock, ttl_quote_s=60)
    svc.get_quote("AAPL")
    clock.advance(3600)
    primary.behavior = "down"
    with capture_logs() as logs:
        quote = svc.get_quote("AAPL")
    assert quote.stale is True and quote.price == 100
    assert any(e["event"] == "market_serving_stale" for e in logs)
    assert svc.get_quote("AAPL").stale is True  # cache itself isn't overwritten with the flag


def test_unavailable_when_nothing_cached() -> None:
    """REQ-MD-06: all down and no cache -> MarketDataUnavailableError with a safe message."""
    with pytest.raises(MarketDataUnavailableError) as err:
        _service(FakeProvider("p", "down"), FakeProvider("b", "down")).get_history("AAPL")
    assert "unavailable" in err.value.user_message.lower()


def test_unknown_ticker_when_every_provider_has_no_data() -> None:
    """REQ-MD-07: only when all answering providers say 'no data' is the ticker called invalid."""
    with pytest.raises(InvalidTickerError, match="no provider"):
        _service(FakeProvider("p", "no_data"), FakeProvider("b", "no_data")).get_quote("ZZQXW")


def test_no_data_from_one_and_outage_from_other_is_unavailable() -> None:
    """REQ-MD-07: Yahoo returning 'empty' during an outage must not be reported as a bad ticker."""
    with pytest.raises(MarketDataUnavailableError):
        _service(FakeProvider("p", "no_data"), FakeProvider("b", "down")).get_quote("AAPL")


def test_configuration_errors_fail_loud() -> None:
    """REQ-LLM-07 (applied to data): a rejected API key is raised, not masked by fallback."""
    with pytest.raises(ConfigurationError):
        _service(FakeProvider("p", "config"), FakeProvider("b")).get_quote("AAPL")


def test_history_and_profile_use_separate_cache_keys() -> None:
    """REQ-MD-02: different data types and history lengths don't collide in the cache."""
    primary = FakeProvider("p")
    svc = _service(primary)
    svc.get_history("AAPL", 365)
    svc.get_history("AAPL", 90)
    svc.get_profile("AAPL")
    assert primary.calls == 3


def test_service_requires_providers() -> None:
    """REQ-MD-01: misconfiguration is caught at construction."""
    with pytest.raises(ConfigurationError):
        MarketDataService([], MarketConfig())


def test_build_market_service_skips_alphavantage_without_key(settings: Settings) -> None:
    """REQ-MD-01: no Alpha Vantage key -> yfinance only, with a warning (not a crash)."""
    settings.alphavantage_api_key = None
    with capture_logs() as logs:
        svc = build_market_service(settings)
    assert [p.name for p in svc.providers] == ["yfinance"]
    assert any(e["event"] == "market_provider_disabled" for e in logs)


def test_build_market_service_with_key(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-MD-01: providers are wired in configured order."""
    from pydantic import SecretStr

    settings.alphavantage_api_key = SecretStr("demo")
    assert [p.name for p in build_market_service(settings).providers] == [
        "yfinance",
        "alphavantage",
    ]


def test_short_rate_limit_waits_instead_of_failing() -> None:
    """REQ-MD-03: a 1-call-per-second provider is waited on briefly, not skipped.

    Found by functional testing: with Yahoo down, quote + history back-to-back failed because
    Alpha Vantage's 1/s limit made the service skip it.
    """
    clock = FakeClock()
    backup = FakeProvider("b")
    config = MarketConfig(
        rate_limits={"b": [RateLimit(max_calls=1, period_s=1.1)]}, rate_limit_max_wait_s=2.0
    )
    svc = MarketDataService(
        [FakeProvider("p", "down"), backup], config, clock=clock, sleep=clock.advance
    )
    assert svc.get_quote("AAPL").source == "b"
    assert svc.get_history("AAPL").source == "b"  # waited ~1.1 s on the fake clock
    assert backup.calls == 2


def test_long_rate_limit_wait_is_skipped() -> None:
    """REQ-MD-03: an exhausted daily quota is not waited on."""
    clock = FakeClock()
    waits: list[float] = []
    config = MarketConfig(
        rate_limits={"p": [RateLimit(max_calls=1, period_s=86400)]}, rate_limit_max_wait_s=2.0
    )
    svc = MarketDataService(
        [FakeProvider("p"), FakeProvider("b")], config, clock=clock, sleep=waits.append
    )
    svc.get_quote("AAA")
    assert svc.get_quote("BBB").source == "b"
    assert waits == []
