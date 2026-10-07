from __future__ import annotations

import random

import pytest

from src.data.providers.base import NoDataError, RateLimitedError, TransientProviderError
from src.data.resilience import CircuitBreaker, RateLimiter, TTLCache, retry


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- TTL cache -------------------------------------------------------------------------------


def test_cache_fresh_then_stale() -> None:
    """REQ-MD-02, REQ-MD-06: fresh within TTL; afterwards only available as a stale fallback."""
    clock = FakeClock()
    cache = TTLCache(clock=clock)
    cache.set("k", "v")
    assert cache.get_fresh("k", ttl_s=60) == "v"
    clock.advance(61)
    assert cache.get_fresh("k", ttl_s=60) is None
    assert cache.get_any("k") == ("v", 61)
    assert cache.get_any("missing") is None


def test_cache_evicts_least_recently_used() -> None:
    """REQ-MD-02: the cache is bounded."""
    cache = TTLCache(maxsize=2, clock=FakeClock())
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get_fresh("a", 60)  # 'a' becomes most recent
    cache.set("c", 3)
    assert cache.get_any("b") is None
    assert cache.get_any("a") is not None and cache.get_any("c") is not None


# --- rate limiter ----------------------------------------------------------------------------


def test_rate_limiter_enforces_every_window() -> None:
    """REQ-MD-03: 1 call/1.1s AND 3 calls/day are both enforced, without blocking."""
    clock = FakeClock()
    limiter = RateLimiter([(1, 1.1), (3, 86400)], clock=clock)
    assert limiter.try_acquire()
    assert not limiter.try_acquire()  # per-second window full
    clock.advance(1.2)
    assert limiter.try_acquire()
    clock.advance(1.2)
    assert limiter.try_acquire()
    clock.advance(1.2)
    assert not limiter.try_acquire()  # daily window full
    clock.advance(86400)
    assert limiter.try_acquire()


def test_rate_limiter_without_limits_always_allows() -> None:
    """REQ-MD-03: providers with no configured limits aren't throttled."""
    limiter = RateLimiter([], clock=FakeClock())
    assert all(limiter.try_acquire() for _ in range(100))


# --- circuit breaker -------------------------------------------------------------------------


def test_breaker_opens_after_threshold_and_half_opens_after_reset() -> None:
    """REQ-MD-05: N consecutive failures open the breaker; it probes again after the cool-down."""
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, reset_s=60, clock=clock)
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == "closed" and breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open" and not breaker.allow()
    clock.advance(60)
    assert breaker.state == "half_open" and breaker.allow()


def test_breaker_half_open_probe_outcomes() -> None:
    """REQ-MD-05: a failed probe re-opens immediately; a successful one closes."""
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, reset_s=10, clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    clock.advance(10)
    breaker.record_failure()  # probe failed
    assert breaker.state == "open"
    clock.advance(10)
    breaker.record_success()  # probe succeeded
    assert breaker.state == "closed" and breaker.failures == 0


def test_success_resets_failure_count() -> None:
    """REQ-MD-05: only *consecutive* failures count."""
    breaker = CircuitBreaker(failure_threshold=2, clock=FakeClock())
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == "closed"


# --- retry -----------------------------------------------------------------------------------


def test_retry_succeeds_after_transient_errors_with_bounded_jittered_backoff() -> None:
    """REQ-MD-04: transient errors are retried with exponential backoff and full jitter."""
    calls = {"n": 0}
    delays: list[float] = []

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitedError("slow down")
        return "ok"

    result = retry(
        flaky,
        attempts=3,
        base_delay_s=1.0,
        max_delay_s=10,
        sleep=delays.append,
        rng=random.Random(7),  # noqa: S311 — deterministic jitter for the test
    )
    assert result == "ok" and calls["n"] == 3
    assert len(delays) == 2
    assert 0 <= delays[0] <= 1.0 and 0 <= delays[1] <= 2.0  # caps double each attempt


def test_retry_gives_up_after_attempts() -> None:
    """REQ-MD-04: the last transient error propagates after the final attempt."""
    delays: list[float] = []

    def always_down() -> None:
        raise TransientProviderError("down")

    with pytest.raises(TransientProviderError):
        retry(always_down, attempts=3, sleep=delays.append)
    assert len(delays) == 2


def test_retry_does_not_retry_permanent_errors() -> None:
    """REQ-MD-04: 'no data' is not transient — fail immediately, no sleeping."""
    delays: list[float] = []

    def missing() -> None:
        raise NoDataError("unknown ticker")

    with pytest.raises(NoDataError):
        retry(missing, attempts=5, sleep=delays.append)
    assert delays == []


def test_retry_delay_respects_max() -> None:
    """REQ-MD-04: backoff never exceeds max_delay."""
    delays: list[float] = []

    def down() -> None:
        raise TransientProviderError("x")

    with pytest.raises(TransientProviderError):
        retry(down, attempts=6, base_delay_s=1, max_delay_s=2, sleep=delays.append)
    assert max(delays) <= 2


def test_rate_limiter_reports_wait_time() -> None:
    """REQ-MD-03: callers learn how long until a slot frees, per window."""
    clock = FakeClock()
    limiter = RateLimiter([(1, 1.1), (2, 100)], clock=clock)
    assert limiter.wait_time() == 0
    assert limiter.try_acquire()
    assert limiter.wait_time() == pytest.approx(1.1)
    clock.advance(1.2)
    assert limiter.try_acquire()
    assert limiter.wait_time() == pytest.approx(100 - 1.2)  # daily-style window dominates
