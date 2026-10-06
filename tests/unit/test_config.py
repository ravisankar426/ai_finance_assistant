from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import ALL_ROLES, PROJECT_ROOT, Settings, get_settings


def test_yaml_maps_every_role_to_a_model(settings: Settings) -> None:
    """REQ-LLM-02: every role resolves to a provider/model from config.yaml."""
    assert set(settings.llm.roles) == set(ALL_ROLES)
    assert settings.llm.roles["agent"].provider == "openai"
    assert settings.llm.roles["judge"].provider == "google_genai"
    assert settings.llm.fallback is not None
    assert settings.llm.fallback.provider == "google_genai"


def test_env_var_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LLM-02: switching a role's model is a config/env change only."""
    monkeypatch.setenv("LLM__ROLES__AGENT__MODEL", "gpt-4.1-nano")
    monkeypatch.setenv("APP__LOG_LEVEL", "DEBUG")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm.roles["agent"].model == "gpt-4.1-nano"
    assert s.llm.roles["agent"].provider == "openai"  # sibling keys from YAML survive
    assert s.app.log_level == "DEBUG"


def test_secrets_never_appear_in_repr(settings: Settings) -> None:
    """REQ-NFR-04: API keys are SecretStr and are masked in repr/logs."""
    assert settings.openai_api_key is not None
    assert "sk-test" not in repr(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-test"


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


def test_missing_pricing_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LLM-04: every configured model must have a price, checked at startup."""
    roles = "\n".join(f"    {r}: {{provider: openai, model: m1}}" for r in ALL_ROLES)
    cfg = _write_config(
        tmp_path,
        f"llm:\n  roles:\n{roles}\n  embeddings: {{provider: openai, model: e1}}\n"
        "  pricing:\n    m1: {input: 1, output: 1}\n",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg))
    with pytest.raises(ValidationError, match="no entry for: \\['e1'\\]"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_missing_role_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LLM-02: config must define a model for every role."""
    cfg = _write_config(
        tmp_path,
        "llm:\n  roles:\n    agent: {provider: openai, model: m1}\n"
        "  embeddings: {provider: openai, model: m1}\n  pricing:\n    m1: {input: 1, output: 1}\n",
    )
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg))
    with pytest.raises(ValidationError, match="missing"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_is_cached() -> None:
    """REQ-LLM-02: settings are loaded once per process."""
    assert get_settings() is get_settings()
    assert (PROJECT_ROOT / "config.yaml").exists()
