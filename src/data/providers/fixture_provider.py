"""Recorded market data for tests and offline demos (REQ-DEL-02, task T5.4).

Serves normalized models recorded from the live providers by
``scripts/record_market_fixtures.py``. Every value is labeled ``source="recorded <date>"`` so a
demo never passes recorded prices off as live. Enable with
``MARKET__PROVIDERS='["fixture"]' make ui``.

Layout: ``<fixture_dir>/{quote,history,profile,news}/<SUBJECT>.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.data.models import AssetProfile, NewsFeed, PriceHistory, Quote
from src.data.providers.base import NoDataError


def subject_filename(subject: str) -> str:
    """File-safe name for a ticker or topic ('stock market' -> 'stock_market')."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", subject.strip())


class FixtureProvider:
    """Read-only provider backed by JSON files of normalized models."""

    name = "fixture"

    def __init__(self, root: Path) -> None:
        self.root = root
        meta_path = root / "meta.json"
        meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.recorded_on: str = meta.get("recorded_on", "unknown date")

    def _load(self, kind: str, subject: str, model: type[BaseModel]) -> Any:
        path = self.root / kind / f"{subject_filename(subject)}.json"
        if not path.exists():
            raise NoDataError(f"no recorded {kind} for {subject}")
        value = model.model_validate_json(path.read_text())
        return value.model_copy(update={"source": f"recorded {self.recorded_on}"})

    def get_quote(self, ticker: str) -> Quote:
        """Return the recorded quote."""
        return self._load("quote", ticker, Quote)  # type: ignore[no-any-return]

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        """Return the recorded history (whatever length was recorded)."""
        return self._load("history", ticker, PriceHistory)  # type: ignore[no-any-return]

    def get_profile(self, ticker: str) -> AssetProfile:
        """Return the recorded profile."""
        return self._load("profile", ticker, AssetProfile)  # type: ignore[no-any-return]

    def get_news(self, subject: str, limit: int) -> NewsFeed:
        """Return recorded headlines (callers anchor the lookback to the newest one)."""
        feed: NewsFeed = self._load("news", subject, NewsFeed)
        return feed.model_copy(update={"items": feed.items[:limit]})
