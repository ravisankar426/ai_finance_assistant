"""Resilience building blocks for external calls (REQ-MD-02..06).

All take an injectable ``clock`` (and ``sleep``) so tests run instantly and deterministically.
"""

from __future__ import annotations

import random
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.data.providers.base import TransientProviderError

Clock = Callable[[], float]


# --- cache ------------------------------------------------------------------------------------


@dataclass
class CacheEntry[T]:
    """A cached value and when it was stored."""

    value: T
    stored_at: float


class TTLCache:
    """In-process cache with per-read freshness and a stale fallback (REQ-MD-02, REQ-MD-06).

    Entries are never expired on write — a stale copy is still useful when every provider is
    down. Size is bounded with least-recently-used eviction. Behind a small interface so Redis
    can replace it for multi-worker deployments (design 7b).
    """

    def __init__(self, *, maxsize: int = 2048, clock: Clock = time.monotonic) -> None:
        self.maxsize = maxsize
        self.clock = clock
        self._data: OrderedDict[Any, CacheEntry[Any]] = OrderedDict()

    def set(self, key: Any, value: Any) -> None:
        """Store ``value`` now, evicting the least recently used entry if full."""
        self._data[key] = CacheEntry(value, self.clock())
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def get_fresh(self, key: Any, ttl_s: float) -> Any | None:
        """Return the value if younger than ``ttl_s``, else None."""
        entry = self._data.get(key)
        if entry is None or self.clock() - entry.stored_at > ttl_s:
            return None
        self._data.move_to_end(key)
        return entry.value

    def get_any(self, key: Any) -> tuple[Any, float] | None:
        """Return (value, age in seconds) regardless of freshness, or None."""
        entry = self._data.get(key)
        return None if entry is None else (entry.value, self.clock() - entry.stored_at)


# --- rate limiting ----------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window limits, e.g. [1 call / 1.1 s, 25 calls / day] (REQ-MD-03).

    ``try_acquire`` never blocks. ``wait_time`` tells the caller how long until a slot frees, so
    it can choose: wait briefly (1 call/second limits) or move on (daily quota exhausted).
    """

    def __init__(
        self, limits: Sequence[tuple[int, float]], *, clock: Clock = time.monotonic
    ) -> None:
        self.limits = list(limits)
        self.clock = clock
        self._calls: deque[float] = deque()

    def try_acquire(self) -> bool:
        """Record a call and return True if every window has room; otherwise False."""
        now = self.clock()
        longest = max((p for _, p in self.limits), default=0.0)
        while self._calls and now - self._calls[0] >= longest:
            self._calls.popleft()
        for max_calls, period in self.limits:
            if sum(1 for t in self._calls if now - t < period) >= max_calls:
                return False
        self._calls.append(now)
        return True

    def wait_time(self) -> float:
        """Seconds until ``try_acquire`` would succeed (0 if it would succeed now)."""
        now = self.clock()
        wait = 0.0
        for max_calls, period in self.limits:
            in_window = [t for t in self._calls if now - t < period]
            if len(in_window) >= max_calls:
                oldest_blocking = in_window[len(in_window) - max_calls]
                wait = max(wait, period - (now - oldest_blocking))
        return wait


# --- circuit breaker --------------------------------------------------------------------------

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Stop calling a provider that keeps failing; probe again after a cool-down (REQ-MD-05).

    closed --(N consecutive failures)--> open --(reset_s elapsed)--> half_open
    half_open --success--> closed;  half_open --failure--> open
    """

    def __init__(
        self, *, failure_threshold: int = 5, reset_s: float = 60, clock: Clock = time.monotonic
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_s = reset_s
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        """Current state."""
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.reset_s else "open"

    def allow(self) -> bool:
        """Return True if a call may be attempted now."""
        return self.state != "open"

    def record_success(self) -> None:
        """Close the breaker."""
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        """Count a failure; open (or re-open) the breaker at the threshold or after a probe."""
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.opened_at = self.clock()


# --- retry ------------------------------------------------------------------------------------


def retry[T](
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 4.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call ``fn``; on ``TransientProviderError`` retry with exponential backoff + full jitter.

    Delay before retry n (1-based) is uniform(0, min(max_delay, base * 2**(n-1))). Jitter
    spreads retries out so many clients don't hammer a recovering provider in lockstep.
    Non-transient errors propagate immediately (REQ-MD-04).
    """
    rng = rng or random.Random()  # noqa: S311 — jitter, not cryptography
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TransientProviderError:
            if attempt == attempts:
                raise
            sleep(rng.uniform(0, min(max_delay_s, base_delay_s * 2 ** (attempt - 1))))
    raise AssertionError("unreachable")  # pragma: no cover
