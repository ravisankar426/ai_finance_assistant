"""Functional check of the market-data layer (live APIs).

    uv run python scripts/market_check.py AAPL SPY            # quote + history + profile
    uv run python scripts/market_check.py MSFT --yahoo-down   # simulate a Yahoo outage
    uv run python scripts/market_check.py ZZQXW "BAD TICKER!" # error handling

Uses at most a few Alpha Vantage calls (free tier: 25/day).
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from src.core.config import get_settings
from src.core.errors import AppError
from src.data.market_service import MarketDataService, build_market_service
from src.data.providers.alphavantage_provider import AlphaVantageProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.utils.logging import configure_logging


class _YahooDown:
    """Stands in for a yfinance Ticker during an outage."""

    def history(self, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("simulated Yahoo outage")

    @property
    def info(self) -> Any:
        raise ConnectionError("simulated Yahoo outage")


def build(yahoo_down: bool) -> MarketDataService:
    settings = get_settings()
    if not yahoo_down:
        return build_market_service(settings)
    providers: list[Any] = [YFinanceProvider(ticker_factory=lambda _t: _YahooDown())]
    if settings.alphavantage_api_key:
        providers.append(AlphaVantageProvider(settings.alphavantage_api_key))
    return MarketDataService(providers, settings.market)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--yahoo-down", action="store_true", help="simulate a Yahoo outage")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    configure_logging(level=args.log_level)
    svc = build(args.yahoo_down)
    print(
        f"providers: {[p.name for p in svc.providers]}"
        + ("  (Yahoo simulated DOWN)" if args.yahoo_down else "")
    )

    for raw in args.tickers:
        print(f"\n== {raw}")
        steps = (
            ("quote", svc.get_quote),
            ("history", svc.get_history),
            ("profile", svc.get_profile),
        )
        ok = True
        for kind, fn in steps:
            start = time.perf_counter()
            try:
                v = fn(raw)
            except AppError as exc:
                print(f"   {kind:<8} -> {type(exc).__name__}: {exc.user_message}")
                ok = False
                break  # same answer for the other kinds
            ms = (time.perf_counter() - start) * 1000
            if kind == "quote":
                detail = f"${v.price:,.2f} ({v.change_pct:+.2f}%) on {v.market_date}"
            elif kind == "history":
                detail = (
                    f"{len(v.bars)} daily bars {v.bars[0].day} → {v.bars[-1].day}, "
                    f"{'adjusted' if v.adjusted else 'UNadjusted'}"
                )
            else:
                detail = f"{v.name} | {v.asset_type} | sector: {v.sector or '—'}"
            flags = f"source={v.source}{', STALE' if v.stale else ''}, {ms:.0f} ms"
            print(f"   {kind:<8} {detail}  [{flags}]")

        if ok and not args.yahoo_down:
            start = time.perf_counter()
            svc.get_quote(raw)
            ms = (time.perf_counter() - start) * 1000
            print(f"   repeat quote served from cache in {ms:.1f} ms")


if __name__ == "__main__":
    main()
