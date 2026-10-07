"""Market-data service: one call path with fallback, caching, and resilience (design section 8).

For every request:

    validate ticker -> fresh cache? -> for each provider in order:
        breaker open? skip -> rate limit full? skip -> call with retries
          success            -> cache, close breaker, return
          transient (final)  -> breaker failure, next provider
          no data / unsupported -> next provider (not the provider's fault)
    -> all failed: stale cache (flagged stale) -> else MarketDataUnavailableError
       (or InvalidTickerError if every provider that answered said "no such ticker")
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, TypeVar, cast

from src.core.config import MarketConfig, Settings, get_settings
from src.core.errors import ConfigurationError, InvalidTickerError, MarketDataUnavailableError
from src.data.models import AssetProfile, NewsFeed, PriceHistory, Quote
from src.data.providers.base import (
    MarketDataProvider,
    NoDataError,
    ProviderError,
    UnsupportedOperationError,
    looks_like_ticker,
    normalize_ticker,
)
from src.data.resilience import CircuitBreaker, RateLimiter, TTLCache, retry
from src.utils.logging import get_logger

log = get_logger(__name__)

M = TypeVar("M", Quote, PriceHistory, AssetProfile, NewsFeed)


class MarketDataService:
    """Quotes, history, and profiles from an ordered list of providers."""

    def __init__(
        self,
        providers: Sequence[MarketDataProvider],
        config: MarketConfig,
        *,
        cache: TTLCache | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not providers:
            raise ConfigurationError("No market-data providers configured")
        self.providers = list(providers)
        self.config = config
        self.cache = cache or TTLCache(clock=clock)
        self.sleep = sleep
        self.breakers = {
            p.name: CircuitBreaker(
                failure_threshold=config.breaker_failure_threshold,
                reset_s=config.breaker_reset_s,
                clock=clock,
            )
            for p in self.providers
        }
        self.limiters = {
            p.name: RateLimiter(
                [(r.max_calls, r.period_s) for r in config.rate_limits.get(p.name, [])], clock=clock
            )
            for p in self.providers
        }

    # -- public API ----------------------------------------------------------------------------

    def get_quote(self, ticker: str) -> Quote:
        """Latest quote (REQ-MK-01)."""
        return self._fetch("quote", ticker, self.config.ttl_quote_s, lambda p, t: p.get_quote(t))

    def get_history(self, ticker: str, days: int | None = None) -> PriceHistory:
        """Daily closes for the last ``days`` (default from config)."""
        n = days or self.config.history_days
        return self._fetch(
            f"history:{n}", ticker, self.config.ttl_history_s, lambda p, t: p.get_history(t, n)
        )

    def get_profile(self, ticker: str) -> AssetProfile:
        """Name, asset type, and GICS sector."""
        return self._fetch(
            "profile", ticker, self.config.ttl_profile_s, lambda p, t: p.get_profile(t)
        )

    def get_news(self, subject: str, limit: int | None = None) -> NewsFeed:
        """Recent headlines for a ticker ("AAPL") or a topic ("stock market") (REQ-NW-01)."""
        n = limit or self.config.news_max_items
        return self._fetch(
            f"news:{n}",
            subject,
            self.config.ttl_news_s,
            lambda p, t: p.get_news(t, n),
            validate=looks_like_ticker(subject),
        )

    # -- core ----------------------------------------------------------------------------------

    def _fetch(
        self,
        kind: str,
        raw_ticker: str,
        ttl_s: float,
        call: Callable[[MarketDataProvider, str], M],
        *,
        validate: bool = True,
    ) -> M:
        # REQ-MD-07: validate before any network call (topics like "stock market" skip this).
        ticker = normalize_ticker(raw_ticker) if validate else raw_ticker.strip().lower()
        key = (kind, ticker)
        cached = self.cache.get_fresh(key, ttl_s)
        if cached is not None:
            log.debug("market_cache_hit", kind=kind, ticker=ticker)
            return cast(M, cached)

        outcomes: dict[str, str] = {}
        for provider in self.providers:
            name = provider.name
            breaker = self.breakers[name]
            if not breaker.allow():
                outcomes[name] = "circuit_open"
                continue
            if not self._acquire_slot(name):
                outcomes[name] = "rate_limited_locally"
                continue
            start = time.perf_counter()
            try:
                value = retry(
                    partial(call, provider, ticker),
                    attempts=self.config.retry_attempts,
                    base_delay_s=self.config.retry_base_delay_s,
                    max_delay_s=self.config.retry_max_delay_s,
                    sleep=self.sleep,
                )
            except NoDataError:
                outcomes[name] = "no_data"
            except UnsupportedOperationError:
                outcomes[name] = "unsupported"
            except ProviderError as exc:  # transient, after retries
                breaker.record_failure()
                outcomes[name] = f"failed:{type(exc).__name__}"
                log.warning(
                    "market_provider_failed",
                    provider=name,
                    kind=kind,
                    ticker=ticker,
                    error_type=type(exc).__name__,
                    breaker=breaker.state,
                )
            else:
                breaker.record_success()
                self.cache.set(key, value)
                log.info(
                    "market_fetch",
                    provider=name,
                    kind=kind,
                    ticker=ticker,
                    latency_ms=round((time.perf_counter() - start) * 1000),
                    fallback=name != self.providers[0].name,
                )
                return value
        return cast(M, self._fallback(kind, ticker, key, outcomes))

    def _acquire_slot(self, name: str) -> bool:
        """Take a rate-limit slot, waiting briefly if one frees up soon (REQ-MD-03)."""
        limiter = self.limiters[name]
        if limiter.try_acquire():
            return True
        wait = limiter.wait_time()
        if wait > self.config.rate_limit_max_wait_s:
            return False  # e.g. daily quota used up: don't make the user wait hours
        log.debug("market_rate_limit_wait", provider=name, wait_s=round(wait, 2))
        self.sleep(wait)
        return limiter.try_acquire()

    def _fallback(self, kind: str, ticker: str, key: Any, outcomes: dict[str, str]) -> Any:
        stale = self.cache.get_any(key)
        if stale is not None:
            value, age = stale
            log.warning(
                "market_serving_stale",
                kind=kind,
                ticker=ticker,
                age_s=round(age),
                outcomes=outcomes,
            )
            return value.model_copy(update={"stale": True})  # REQ-MD-06, REQ-MK-04
        attempted = [o for o in outcomes.values() if o not in ("unsupported",)]
        if attempted and all(o == "no_data" for o in attempted):
            raise InvalidTickerError(
                f"no provider has data for {ticker}: {outcomes}",
                user_message=f"I couldn't find market data for “{ticker}”. Check the symbol.",
            )
        log.error("market_data_unavailable", kind=kind, ticker=ticker, outcomes=outcomes)
        raise MarketDataUnavailableError(f"{kind} {ticker}: {outcomes}")


def build_market_service(settings: Settings | None = None) -> MarketDataService:
    """Wire providers in configured order; skip Alpha Vantage (with a warning) if no key."""
    from src.data.providers.alphavantage_provider import AlphaVantageProvider
    from src.data.providers.fixture_provider import FixtureProvider
    from src.data.providers.yfinance_provider import YFinanceProvider

    settings = settings or get_settings()
    cfg = settings.market
    providers: list[MarketDataProvider] = []
    for name in cfg.providers:
        if name == "yfinance":
            providers.append(YFinanceProvider())
        elif name == "fixture":
            providers.append(FixtureProvider(cfg.fixture_dir))
        elif name == "alphavantage":
            if settings.alphavantage_api_key is None:
                log.warning("market_provider_disabled", provider=name, reason="no API key")
                continue
            providers.append(
                AlphaVantageProvider(settings.alphavantage_api_key, timeout_s=cfg.request_timeout_s)
            )
    return MarketDataService(providers, cfg)
