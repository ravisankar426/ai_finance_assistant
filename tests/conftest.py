"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.core.config import Settings, get_settings
from src.utils.logging import configure_logging


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Never read the developer's real keys; reset cached settings around each test."""
    for var in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ALPHAVANTAGE_API_KEY", "APP_CONFIG_FILE"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    configure_logging(level="DEBUG", fmt="json")
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Build settings from the real config.yaml with fake API keys and no .env file."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    return Settings(_env_file=None)  # type: ignore[call-arg]
