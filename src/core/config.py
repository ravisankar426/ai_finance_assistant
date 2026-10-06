"""Typed application settings.

Sources, highest priority first:
    1. explicit constructor arguments (tests)
    2. environment variables (nested with ``__``, e.g. ``LLM__ROLES__AGENT__MODEL``)
    3. ``.env`` file
    4. ``config.yaml`` (path overridable with ``APP_CONFIG_FILE``)

Settings are validated once at startup so misconfiguration fails fast (constitution P7).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Role = Literal["router", "guard", "agent", "synthesizer", "judge"]
ALL_ROLES: tuple[Role, ...] = ("router", "guard", "agent", "synthesizer", "judge")


def _config_file() -> Path:
    return Path(os.environ.get("APP_CONFIG_FILE", PROJECT_ROOT / "config.yaml"))


class AppConfig(BaseModel):
    """General application settings."""

    name: str = "ai-finance-assistant"
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"


class ModelConfig(BaseModel):
    """One chat model: which provider, which model, and how to call it."""

    provider: str
    model: str
    temperature: float | None = None  # None = provider default (some reasoning models reject it)
    timeout_s: float = Field(default=30, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_tokens: int | None = None
    # Provider-specific constructor kwargs (e.g. Gemini thinking settings). Keeps provider
    # quirks in config instead of code (REQ-LLM-02).
    extra: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """Embedding model settings."""

    provider: str
    model: str


class ModelPrice(BaseModel):
    """Price in USD per one million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class LLMConfig(BaseModel):
    """Role-to-model mapping, fallback, embeddings, and pricing."""

    roles: dict[Role, ModelConfig]
    fallback: ModelConfig | None = None
    fallback_exempt_roles: list[Role] = Field(default_factory=list)
    embeddings: EmbeddingConfig
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_complete(self) -> LLMConfig:
        missing_roles = [r for r in ALL_ROLES if r not in self.roles]
        if missing_roles:
            raise ValueError(f"llm.roles is missing: {missing_roles}")
        models = {m.model for m in self.roles.values()} | {self.embeddings.model}
        if self.fallback:
            models.add(self.fallback.model)
        unpriced = sorted(models - self.pricing.keys())
        if unpriced:
            raise ValueError(f"llm.pricing has no entry for: {unpriced} (needed for REQ-LLM-04)")
        return self


class Settings(BaseSettings):
    """Root settings object. Get it via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig

    # Secrets — env/.env only. SecretStr keeps them out of repr() and logs.
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    alphavantage_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order sources so env vars override YAML (12-factor)."""
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=_config_file())
        return (init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached; tests call ``get_settings.cache_clear()``)."""
    return Settings()
