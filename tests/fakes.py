"""Deterministic fake chat models for tests (no network, no keys)."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import ModelAPIError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


class ScriptedChatModel(BaseChatModel):
    """Returns ``reply`` (or raises ``fail_with`` if ``fail``); counts calls.

    The default failure is ``ModelAPIError`` — a provider 5xx, which the gateway falls back on.
    """

    reply: str = "ok"
    fail: bool = False
    fail_with: type[Exception] = ModelAPIError
    usage: dict[str, int] | None = None
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        if self.fail:
            raise self.fail_with("provider down")
        message = AIMessage(content=self.reply, usage_metadata=self.usage)  # type: ignore[arg-type]
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        def parse(_: Any) -> BaseModel:
            self.calls += 1
            if self.fail:
                raise self.fail_with("provider down")
            return schema.model_validate_json(self.reply)  # type: ignore[no-any-return]

        return RunnableLambda(parse)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable[Any, Any]:
        return self.bind(tools_bound=[getattr(t, "__name__", str(t)) for t in tools])
