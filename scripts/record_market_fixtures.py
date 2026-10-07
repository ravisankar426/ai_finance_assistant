"""Record live market data into src/data/fixtures/market/ (REQ-DEL-02).

    uv run python scripts/record_market_fixtures.py

Records quotes, ~400 days of history, and profiles for the demo tickers, the 4 index ETFs and
the 11 sector ETFs, plus headlines for a few tickers and the overall market. Uses yfinance
only (no Alpha Vantage quota).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.core.config import get_settings
from src.core.market_analytics import INDEX_ETFS, SECTOR_ETFS
from src.data.providers.fixture_provider import subject_filename
from src.data.providers.yfinance_provider import YFinanceProvider
from src.utils.logging import configure_logging

DEMO_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "JNJ", "JPM", "XOM", "VTI", "BND"]
NEWS_SUBJECTS = ["AAPL", "MSFT", "NVDA", "TSLA", "stock market"]


def main() -> None:
    configure_logging(level="WARNING")
    cfg = get_settings().market
    root = cfg.fixture_dir
    provider = YFinanceProvider()
    tickers = [*DEMO_TICKERS, *INDEX_ETFS, *SECTOR_ETFS]
    for kind in ("quote", "history", "profile", "news"):
        (root / kind).mkdir(parents=True, exist_ok=True)

    def save(kind: str, subject: str, model: object) -> None:
        path = root / kind / f"{subject_filename(subject)}.json"
        path.write_text(model.model_dump_json(indent=1) + "\n")  # type: ignore[attr-defined]

    for t in tickers:
        save("quote", t, provider.get_quote(t))
        save("history", t, provider.get_history(t, cfg.history_days))
        save("profile", t, provider.get_profile(t))
        print(f"recorded {t}")
    for subject in NEWS_SUBJECTS:
        feed = provider.get_news(subject, cfg.news_max_items)
        save("news", subject, feed)
        print(f"recorded news '{subject}': {len(feed.items)} headlines")
    (root / "meta.json").write_text(
        json.dumps(
            {
                "recorded_on": datetime.now(UTC).date().isoformat(),
                "tickers": tickers,
                "news": NEWS_SUBJECTS,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"saved to {root}")


if __name__ == "__main__":
    main()
