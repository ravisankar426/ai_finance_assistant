from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from structlog.testing import capture_logs

import src.core.llm as llm_module
from src.core.config import PROJECT_ROOT, ModelConfig, ModelPrice, Settings
from src.core.errors import ConfigurationError
from src.core.llm import (
    CONFIG_ERRORS,
    FALLBACK_ERRORS,
    UsageCallback,
    estimate_cost,
    get_chat_model,
    get_embeddings,
    track_usage,
)
from tests.fakes import ScriptedChatModel


class Answer(BaseModel):
    text: str


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace model construction with fakes; record which configs were built."""
    state: dict[str, Any] = {"built": [], "primary_fail": False, "primary_error": ModelAPIError}

    def build(cfg: ModelConfig, *, role: str, settings: Settings, is_fallback: bool = False) -> Any:
        state["built"].append((role, cfg.provider, cfg.model, is_fallback))
        reply = '{"text": "from fallback"}' if is_fallback else '{"text": "from primary"}'
        model = ScriptedChatModel(
            reply=reply,
            fail=state["primary_fail"] and not is_fallback,
            fail_with=state["primary_error"],
        )
        state["fallback" if is_fallback else "primary"] = model
        return model

    monkeypatch.setattr(llm_module, "_build_model", build)
    return state


def test_role_resolves_to_configured_model(settings: Settings, fake_models: dict[str, Any]) -> None:
    """REQ-LLM-02: the gateway builds exactly the provider/model config.yaml names for the role."""
    model = get_chat_model("router", settings=settings)
    assert model.invoke([HumanMessage("hi")]).content == '{"text": "from primary"}'
    router_cfg = settings.llm.roles["router"]
    assert ("router", router_cfg.provider, router_cfg.model, False) in fake_models["built"]


def test_fallback_used_when_primary_fails(settings: Settings, fake_models: dict[str, Any]) -> None:
    """REQ-LLM-03: a failing primary call is retried on the Gemini fallback, and that is logged."""
    fake_models["primary_fail"] = True
    model = get_chat_model("agent", settings=settings)
    with track_usage() as usage, capture_logs() as logs:
        reply = model.invoke([HumanMessage("hi")])
    assert reply.content == '{"text": "from fallback"}'
    assert fake_models["primary"].calls == 1
    assert fake_models["fallback"].calls == 1
    assert usage.fallbacks == 1
    events = {e["event"]: e for e in logs}
    assert events["llm_fallback_used"]["to_model"].startswith("google_genai:")


def test_fallback_not_used_when_primary_succeeds(
    settings: Settings, fake_models: dict[str, Any]
) -> None:
    """REQ-LLM-03: the fallback is only touched on failure."""
    get_chat_model("agent", settings=settings).invoke("hi")
    assert fake_models["fallback"].calls == 0


async def test_fallback_works_async(settings: Settings, fake_models: dict[str, Any]) -> None:
    """REQ-LLM-03: fallback also works on the async path used by the API."""
    fake_models["primary_fail"] = True
    reply = await get_chat_model("agent", settings=settings).ainvoke("hi")
    assert reply.content == '{"text": "from fallback"}'


def test_structured_output_applied_to_primary_and_fallback(
    settings: Settings, fake_models: dict[str, Any]
) -> None:
    """REQ-LLM-03: capabilities are attached to each model before fallback wrapping."""
    fake_models["primary_fail"] = True
    result = get_chat_model("router", structured_output=Answer, settings=settings).invoke("hi")
    assert result == Answer(text="from fallback")


def test_tools_are_bound(settings: Settings, fake_models: dict[str, Any]) -> None:
    """REQ-LLM-01: agents get tool-calling through the gateway too."""

    def get_quote(ticker: str) -> str:
        """Fake tool."""
        return ticker

    model = get_chat_model("agent", tools=[get_quote], settings=settings)
    assert model.invoke("hi").content == '{"text": "from primary"}'


def test_structured_output_and_tools_are_exclusive(
    settings: Settings, fake_models: dict[str, Any]
) -> None:
    """REQ-LLM-01: misuse fails loudly at build time."""
    with pytest.raises(ValueError, match="either"):
        get_chat_model("agent", structured_output=Answer, tools=[print], settings=settings)


def test_judge_is_exempt_from_fallback(settings: Settings, fake_models: dict[str, Any]) -> None:
    """REQ-LLM-06: the judge never silently switches provider."""
    fake_models["primary_fail"] = True
    with pytest.raises(ModelAPIError, match="provider down"):
        get_chat_model("judge", settings=settings).invoke("score this")
    assert all(not fb for (_, _, _, fb) in fake_models["built"])


@pytest.mark.parametrize(
    "error", [ModelRateLimitError, ModelAPIError, ModelTimeoutError, ContextOverflowError]
)
def test_transient_errors_fall_back(
    settings: Settings, fake_models: dict[str, Any], error: type[Exception]
) -> None:
    """REQ-LLM-03: rate limits, 5xx, timeouts, and context overflow fall back to Gemini."""
    fake_models["primary_fail"] = True
    fake_models["primary_error"] = error
    assert get_chat_model("agent", settings=settings).invoke("hi").text == (
        '{"text": "from fallback"}'
    )


@pytest.mark.parametrize(
    "error", [ModelAuthenticationError, ModelPermissionDeniedError, ModelNotFoundError]
)
def test_config_errors_fail_loud(
    settings: Settings, fake_models: dict[str, Any], error: type[Exception]
) -> None:
    """REQ-LLM-07: bad key / no permission / unknown model never fall back; logged at ERROR."""
    fake_models["primary_fail"] = True
    fake_models["primary_error"] = error
    model = get_chat_model("agent", settings=settings)
    with capture_logs() as logs, pytest.raises(error):
        model.invoke("hi")
    assert fake_models["fallback"].calls == 0
    assert not any(e["event"] == "llm_fallback_used" for e in logs)


async def test_config_errors_fail_loud_async_and_streaming(
    settings: Settings, fake_models: dict[str, Any]
) -> None:
    """REQ-LLM-07: the policy holds on the async and streaming paths the API uses."""
    fake_models["primary_fail"] = True
    fake_models["primary_error"] = ModelAuthenticationError
    model = get_chat_model("agent", settings=settings)
    with pytest.raises(ModelAuthenticationError):
        await model.ainvoke("hi")
    with pytest.raises(ModelAuthenticationError):
        async for _ in model.astream("hi"):
            pass
    assert fake_models["fallback"].calls == 0


def test_unanticipated_errors_fail_loud(settings: Settings, fake_models: dict[str, Any]) -> None:
    """REQ-LLM-07: errors outside the fallback allowlist surface instead of being masked."""
    fake_models["primary_fail"] = True
    fake_models["primary_error"] = RuntimeError
    with pytest.raises(RuntimeError):
        get_chat_model("agent", settings=settings).invoke("hi")
    assert fake_models["fallback"].calls == 0


def test_config_error_is_logged_at_error_level_with_action() -> None:
    """REQ-LLM-07: config errors are logged loudly with what to fix."""
    cfg = ModelConfig(provider="openai", model="m")
    model = ScriptedChatModel(
        fail=True,
        fail_with=ModelAuthenticationError,
        callbacks=[UsageCallback("agent", cfg, {}, is_fallback=False)],
    )
    with capture_logs() as logs, pytest.raises(ModelAuthenticationError):
        model.invoke("hi")
    record = next(e for e in logs if e["event"] == "llm_config_error")
    assert record["log_level"] == "error"
    assert "API key" in record["action"]


def _provider_error_classes() -> list[type[BaseException]]:
    """Every provider-neutral-mapped error class both integrations can raise."""
    import inspect

    from langchain_core.exceptions import ModelError
    from langchain_google_genai import chat_models as google_mod
    from langchain_openai.chat_models import base as openai_mod

    return [
        obj
        for mod in (openai_mod, google_mod)
        for obj in vars(mod).values()
        if inspect.isclass(obj) and issubclass(obj, ModelError) and obj.__module__ == mod.__name__
    ]


def test_every_provider_error_is_classified() -> None:
    """REQ-LLM-07: each mapped provider error is either fail-loud or fall-back, never both.

    Guards the allowlist against library upgrades adding error types we haven't classified.
    """
    classes = _provider_error_classes()
    assert len(classes) >= 10  # both integrations were actually inspected
    for cls in classes:
        is_config = issubclass(cls, CONFIG_ERRORS)
        is_fallback = issubclass(cls, FALLBACK_ERRORS)
        assert is_config != is_fallback, (
            f"{cls.__module__}.{cls.__name__} is ambiguous/unclassified"
        )


def test_missing_fallback_key_degrades_to_primary_only(
    settings: Settings, fake_models: dict[str, Any]
) -> None:
    """REQ-LLM-03: no Gemini key -> warn and run without fallback instead of crashing."""
    settings.google_api_key = None
    with capture_logs() as logs:
        get_chat_model("agent", settings=settings).invoke("hi")
    assert any(e["event"] == "llm_fallback_disabled" for e in logs)
    assert len(fake_models["built"]) == 1


def test_missing_primary_key_fails_fast(settings: Settings, fake_models: dict[str, Any]) -> None:
    """REQ-LLM-02: a missing primary key is a configuration error, raised at build time."""
    settings.openai_api_key = None
    with pytest.raises(ConfigurationError, match="openai"):
        get_chat_model("agent", settings=settings)


def test_no_fallback_configured(settings: Settings, fake_models: dict[str, Any]) -> None:
    """REQ-LLM-03: fallback is optional."""
    settings.llm.fallback = None
    get_chat_model("agent", settings=settings).invoke("hi")
    assert len(fake_models["built"]) == 1


def test_usage_callback_records_tokens_and_cost() -> None:
    """REQ-LLM-04: tokens and cost are recorded for every call, from provider-neutral metadata."""
    cfg = ModelConfig(provider="openai", model="m")
    pricing = {"m": ModelPrice(input=1.0, output=2.0)}
    model = ScriptedChatModel(
        usage={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
        callbacks=[UsageCallback("agent", cfg, pricing, is_fallback=False)],
    )
    with track_usage() as usage, capture_logs() as logs:
        model.invoke("hi")
        model.invoke("again")
    assert usage.calls == 2
    assert usage.input_tokens == 2000
    assert usage.output_tokens == 1000
    assert usage.cost_usd == pytest.approx(2 * (1000 * 1.0 + 500 * 2.0) / 1_000_000)
    call_log = next(e for e in logs if e["event"] == "llm_call")
    assert call_log["model"] == "m"
    assert call_log["input_tokens"] == 1000


def test_usage_callback_logs_errors() -> None:
    """REQ-LLM-04: failed calls are logged with the error type."""
    cfg = ModelConfig(provider="openai", model="m")
    model = ScriptedChatModel(
        fail=True, callbacks=[UsageCallback("agent", cfg, {}, is_fallback=False)]
    )
    with capture_logs() as logs, pytest.raises(ModelAPIError):
        model.invoke("hi")
    err = next(e for e in logs if e["event"] == "llm_error")
    assert err["error_type"] == "ModelAPIError"
    assert err["will_fall_back"] is True


def test_estimate_cost() -> None:
    """REQ-LLM-04: cost = tokens x price per million; unknown models cost 0."""
    pricing = {"m": ModelPrice(input=0.40, output=1.60)}
    assert estimate_cost("m", 1_000_000, 0, pricing) == pytest.approx(0.40)
    assert estimate_cost("m", 0, 1_000_000, pricing) == pytest.approx(1.60)
    assert estimate_cost("unknown", 10, 10, pricing) == 0.0


def test_real_builder_wires_config_into_provider_classes(settings: Settings) -> None:
    """REQ-LLM-02: the real builder passes timeout, retries, temperature, key (no network call)."""
    openai_model = llm_module._build_model(
        settings.llm.roles["agent"], role="agent", settings=settings
    )
    assert openai_model.max_retries == settings.llm.roles["agent"].max_retries  # type: ignore[attr-defined]
    assert openai_model.request_timeout == settings.llm.roles["agent"].timeout_s  # type: ignore[attr-defined]
    assert openai_model.temperature == settings.llm.roles["agent"].temperature  # type: ignore[attr-defined]
    assert openai_model.stream_usage is True  # type: ignore[attr-defined]
    judge_cfg = settings.llm.roles["judge"].model_copy(update={"max_tokens": 256})
    gemini_model = llm_module._build_model(judge_cfg, role="judge", settings=settings)
    assert type(gemini_model).__name__ == "ChatGoogleGenerativeAI"
    assert gemini_model.max_retries == settings.llm.roles["judge"].max_retries  # type: ignore[attr-defined]


def test_extra_config_is_passed_to_provider(settings: Settings) -> None:
    """REQ-LLM-02: provider-specific knobs come from config, not code."""
    cfg = settings.llm.roles["judge"].model_copy(update={"extra": {"thinking_budget": 128}})
    model = llm_module._build_model(cfg, role="judge", settings=settings)
    assert model.thinking_budget == 128  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "content",
    [
        "OK",  # OpenAI shape
        [{"type": "text", "text": "OK", "extras": {"signature": "abc"}}],  # Gemini 3 shape
    ],
)
def test_text_accessor_is_provider_neutral(content: Any) -> None:
    """REQ-LLM-02: callers read `.text`, which is identical for both providers' reply shapes."""
    from langchain_core.messages import AIMessage

    assert AIMessage(content=content).text == "OK"


def test_real_gateway_builds_fallback_chain(settings: Settings) -> None:
    """REQ-LLM-03: with real classes, the agent role is OpenAI with a Gemini fallback."""
    chain = get_chat_model("agent", settings=settings)
    assert "Fallbacks" in type(chain.bound).__name__  # type: ignore[attr-defined]


def test_get_embeddings(settings: Settings) -> None:
    """REQ-LLM-01: embeddings come from the gateway; missing key fails fast."""
    emb = get_embeddings(settings)
    assert type(emb).__name__ == "OpenAIEmbeddings"
    settings.openai_api_key = None
    with pytest.raises(ConfigurationError):
        get_embeddings(settings)


VENDOR_MODULES = (
    "openai",
    "anthropic",
    "google.genai",
    "google.generativeai",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
)


def test_no_vendor_imports_outside_gateway() -> None:
    """REQ-LLM-01: only src/core/llm.py may know which LLM vendors exist."""
    offenders = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        if path == PROJECT_ROOT / "src" / "core" / "llm.py":
            continue
        tree = ast.parse(Path(path).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [
                f"{path.relative_to(PROJECT_ROOT)}: {n}"
                for n in names
                if any(n == v or n.startswith(v + ".") for v in VENDOR_MODULES)
            ]
    assert offenders == []
