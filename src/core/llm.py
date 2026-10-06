"""LLM gateway — the only module that knows which LLM vendors exist (REQ-LLM-01).

Callers ask for a *role*; ``config.yaml`` decides which provider/model serves it::

    llm = get_chat_model("agent")
    router = get_chat_model("router", structured_output=RouteDecision)
    emb = get_embeddings()

What every returned model already has:
- provider-native retries with backoff and a timeout (from config);
- a cross-provider fallback (OpenAI -> Gemini by default) unless the role is exempt (REQ-LLM-03);
- token + cost accounting for every call (REQ-LLM-04).

Switching provider = edit ``config.yaml`` + set the provider's API key. No code changes
(REQ-LLM-02). The only provider-specific knowledge lives in the two small tables below.

Rule for callers: read reply text with ``message.text``, never ``message.content``.
OpenAI returns ``content`` as a string; Gemini 3 returns a list of content blocks (text +
thought signatures). ``.text`` is LangChain's provider-neutral accessor for both.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, SecretStr

from src.core.config import ModelConfig, ModelPrice, Role, Settings, get_settings
from src.core.errors import ConfigurationError
from src.utils.logging import get_logger

log = get_logger(__name__)

# --- provider-specific knowledge (keep it here, and only here) ----------------------------

# provider -> (constructor kwarg for the key, Settings attribute holding it)
_API_KEYS: dict[str, tuple[str, str]] = {
    "openai": ("api_key", "openai_api_key"),
    "google_genai": ("google_api_key", "google_api_key"),
}

# Extra constructor kwargs some providers need for our features to work.
_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {"stream_usage": True},  # report token usage when streaming too
}

# --- fallback policy (REQ-LLM-03, REQ-LLM-07) -------------------------------------------------
# LangChain maps every provider's errors onto these provider-neutral classes.

# Configuration problems: switching provider would only hide them. Never fall back; fail loud.
CONFIG_ERRORS: tuple[type[BaseException], ...] = (
    ModelAuthenticationError,  # 401: bad/expired key
    ModelPermissionDeniedError,  # 403: key lacks access
    ModelNotFoundError,  # 404: retired or misspelled model id
)

# Provider-side or request-shape problems another provider may not have: fall back.
# This is an allowlist (with_fallbacks only accepts types), so any error type not listed
# here — including ones we never anticipated — is raised loudly instead of being masked.
FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
    ModelRateLimitError,  # 429
    ModelAPIError,  # 5xx
    ModelConnectionError,
    ModelTimeoutError,
    ModelInvalidRequestError,  # 400: e.g. a schema one provider rejects and another accepts
    ContextOverflowError,  # prompt too long for this model's context window
    OutputParserException,  # structured output didn't parse
    TimeoutError,
    ConnectionError,
)


# --- usage & cost accounting ---------------------------------------------------------------


@dataclass
class UsageTotals:
    """Accumulated LLM usage for one scope (usually one user request)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    fallbacks: int = 0


_usage: ContextVar[UsageTotals | None] = ContextVar("llm_usage", default=None)


@contextmanager
def track_usage() -> Iterator[UsageTotals]:
    """Collect token/cost totals for every LLM call made inside the block."""
    totals = UsageTotals()
    token = _usage.set(totals)
    try:
        yield totals
    finally:
        _usage.reset(token)


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, pricing: dict[str, ModelPrice]
) -> float:
    """Return the USD cost of one call; 0.0 if the model has no price entry."""
    price = pricing.get(model)
    if price is None:
        return 0.0
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


class UsageCallback(BaseCallbackHandler):
    """Logs tokens, cost, and errors for every call of one configured model."""

    def __init__(
        self, role: str, cfg: ModelConfig, pricing: dict[str, ModelPrice], *, is_fallback: bool
    ) -> None:
        self.role = role
        self.cfg = cfg
        self.pricing = pricing
        self.is_fallback = is_fallback

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Record usage from LangChain's provider-neutral ``usage_metadata``."""
        input_tokens = output_tokens = 0
        for generations in response.generations:
            for gen in generations:
                if isinstance(gen, ChatGeneration):
                    usage = getattr(gen.message, "usage_metadata", None) or {}
                    input_tokens += int(usage.get("input_tokens", 0))
                    output_tokens += int(usage.get("output_tokens", 0))
        cost = estimate_cost(self.cfg.model, input_tokens, output_tokens, self.pricing)
        totals = _usage.get()
        if totals is not None:
            totals.calls += 1
            totals.input_tokens += input_tokens
            totals.output_tokens += output_tokens
            totals.cost_usd += cost
        log.info(
            "llm_call",
            role=self.role,
            provider=self.cfg.provider,
            model=self.cfg.model,
            fallback=self.is_fallback,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Log a failed call: ERROR + action hint for config errors, WARNING otherwise."""
        fields = {
            "role": self.role,
            "provider": self.cfg.provider,
            "model": self.cfg.model,
            "fallback": self.is_fallback,
            "error_type": type(error).__name__,
        }
        if isinstance(error, CONFIG_ERRORS):
            log.error(
                "llm_config_error",
                **fields,
                action="check the API key in .env and the model id in config.yaml; "
                "this error is never masked by the fallback",
            )
        else:
            log.warning("llm_error", **fields, will_fall_back=isinstance(error, FALLBACK_ERRORS))


# --- model construction ----------------------------------------------------------------------


def _api_key(provider: str, settings: Settings) -> SecretStr | None:
    entry = _API_KEYS.get(provider)
    return getattr(settings, entry[1]) if entry else None


def _has_key(provider: str, settings: Settings) -> bool:
    # Providers we don't know about may use their own env vars; assume they're configured.
    return provider not in _API_KEYS or _api_key(provider, settings) is not None


def _build_model(
    cfg: ModelConfig, *, role: str, settings: Settings, is_fallback: bool = False
) -> BaseChatModel:
    """Instantiate one chat model from config. Tests replace this with fakes."""
    kwargs: dict[str, Any] = {
        **_PROVIDER_DEFAULTS.get(cfg.provider, {}),
        "timeout": cfg.timeout_s,
        "max_retries": cfg.max_retries,
        "callbacks": [UsageCallback(role, cfg, settings.llm.pricing, is_fallback=is_fallback)],
    }
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    kwargs.update(cfg.extra)  # provider-specific knobs from config.yaml
    if (entry := _API_KEYS.get(cfg.provider)) and (key := _api_key(cfg.provider, settings)):
        kwargs[entry[0]] = key
    model = init_chat_model(cfg.model, model_provider=cfg.provider, **kwargs)
    if not isinstance(model, BaseChatModel):  # only happens if `model` were omitted
        raise ConfigurationError(f"Could not build chat model {cfg.provider}:{cfg.model}")
    return model


def _with_capabilities(
    model: BaseChatModel,
    structured_output: type[BaseModel] | None,
    tools: Sequence[Any] | None,
) -> Runnable[LanguageModelInput, Any]:
    """Apply structured output or tools to a *concrete* model (before any fallback wrapping)."""
    if structured_output is not None and tools:
        raise ValueError("Use either structured_output or tools, not both.")
    if structured_output is not None:
        return model.with_structured_output(structured_output)
    if tools:
        return model.bind_tools(tools)
    return model


def _log_fallback(
    role: str, primary: ModelConfig, fallback: ModelConfig
) -> RunnableLambda[Any, Any]:
    def _passthrough(inp: Any) -> Any:
        totals = _usage.get()
        if totals is not None:
            totals.fallbacks += 1
        log.warning(
            "llm_fallback_used",
            role=role,
            from_model=f"{primary.provider}:{primary.model}",
            to_model=f"{fallback.provider}:{fallback.model}",
        )
        return inp

    return RunnableLambda(_passthrough, name="log_fallback")


def get_chat_model(
    role: Role,
    *,
    structured_output: type[BaseModel] | None = None,
    tools: Sequence[Any] | None = None,
    settings: Settings | None = None,
) -> Runnable[LanguageModelInput, Any]:
    """Return a ready-to-use chat model for ``role``.

    Args:
        role: which job the model does; mapped to provider/model in ``config.yaml``.
        structured_output: Pydantic class to parse the reply into (returns instances of it).
        tools: tools to bind for tool calling.
        settings: override settings (tests); defaults to the process-wide settings.

    Raises:
        ConfigurationError: the primary provider's API key is missing.

    At call time, errors in ``CONFIG_ERRORS`` (bad key, no permission, unknown model) and
    any error not in ``FALLBACK_ERRORS`` propagate without falling back (REQ-LLM-07).

    """
    settings = settings or get_settings()
    primary_cfg = settings.llm.roles[role]
    if not _has_key(primary_cfg.provider, settings):
        raise ConfigurationError(
            f"No API key configured for provider '{primary_cfg.provider}' (role '{role}')"
        )

    primary = _with_capabilities(
        _build_model(primary_cfg, role=role, settings=settings), structured_output, tools
    )
    run_config: dict[str, Any] = {
        "run_name": f"llm:{role}",
        "tags": [f"role:{role}"],
        "metadata": {"role": role},
    }

    fb_cfg = settings.llm.fallback
    use_fallback = (
        fb_cfg is not None
        and role not in settings.llm.fallback_exempt_roles
        and (fb_cfg.provider, fb_cfg.model) != (primary_cfg.provider, primary_cfg.model)
    )
    if not use_fallback or fb_cfg is None:
        return primary.with_config(**run_config)
    if not _has_key(fb_cfg.provider, settings):
        log.warning("llm_fallback_disabled", role=role, reason=f"no API key for {fb_cfg.provider}")
        return primary.with_config(**run_config)

    fallback = _log_fallback(role, primary_cfg, fb_cfg) | _with_capabilities(
        _build_model(fb_cfg, role=role, settings=settings, is_fallback=True),
        structured_output,
        tools,
    )
    return primary.with_fallbacks([fallback], exceptions_to_handle=FALLBACK_ERRORS).with_config(
        **run_config
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the configured embedding model (no cross-provider fallback; see ADR-05)."""
    settings = settings or get_settings()
    cfg = settings.llm.embeddings
    if not _has_key(cfg.provider, settings):
        raise ConfigurationError(f"No API key configured for embeddings provider '{cfg.provider}'")
    kwargs: dict[str, Any] = {}
    if (entry := _API_KEYS.get(cfg.provider)) and (key := _api_key(cfg.provider, settings)):
        kwargs[entry[0]] = key
    return init_embeddings(cfg.model, provider=cfg.provider, **kwargs)
