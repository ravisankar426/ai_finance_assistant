"""Deterministic fakes for tests (no network, no keys)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import ModelAPIError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field


class ScriptedChatModel(BaseChatModel):
    """Returns ``reply`` (or raises ``fail_with`` if ``fail``); counts calls, records inputs.

    The default failure is ``ModelAPIError`` — a provider 5xx, which the gateway falls back on.
    """

    reply: str = "ok"
    fail: bool = False
    fail_with: type[Exception] = ModelAPIError
    usage: dict[str, int] | None = None
    calls: int = 0
    seen: list[Any] = Field(default_factory=list)  # inputs of every call (what reached the "LLM")

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
        self.seen.append(messages)
        if self.fail:
            raise self.fail_with("provider down")
        message = AIMessage(content=self.reply, usage_metadata=self.usage)  # type: ignore[arg-type]
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        def parse(inp: Any) -> BaseModel:
            self.calls += 1
            self.seen.append(inp)
            if self.fail:
                raise self.fail_with("provider down")
            return schema.model_validate_json(self.reply)  # type: ignore[no-any-return]

        return RunnableLambda(parse)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable[Any, Any]:
        return self.bind(tools_bound=[getattr(t, "__name__", str(t)) for t in tools])


def seen_text(model: ScriptedChatModel) -> str:
    """Everything the fake model was sent, flattened to one string."""

    def flat(item: Any) -> str:
        if isinstance(item, BaseMessage):
            return item.text
        if isinstance(item, Sequence) and not isinstance(item, str):
            return "\n".join(flat(i) for i in item)
        return str(item)

    return "\n".join(flat(call) for call in model.seen)


def route_json(intents: list[str], question: str = "What is an index fund?") -> str:
    """JSON a router model would return."""
    return json.dumps({"intents": intents, "standalone_question": question})


class FakeRetriever:
    """Retriever returning fixed chunks (optionally none)."""

    def __init__(self, chunks: list[Any] | None = None) -> None:
        self.chunks = chunks or []
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, *, k: int, **filters: Any) -> list[Any]:
        self.calls.append({"query": query, "k": k, **filters})
        return self.chunks[:k]


class HashingEmbeddings(Embeddings):
    """Deterministic offline embeddings: hashed bag-of-words vectors.

    Cosine similarity reflects shared (tokenized) words, so ranking and gating logic can be
    tested without an API. ``fail_with`` simulates an embeddings outage on queries.
    """

    def __init__(self, dim: int = 4096, fail_with: BaseException | None = None) -> None:
        self.dim = dim
        self.fail_with = fail_with
        self.document_calls = 0

    def _vector(self, text: str) -> list[float]:
        from src.rag.text import tokenize

        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # noqa: S324 — not security
            vec[h % self.dim] += 1.0
        return vec if any(vec) else [1e-6] * self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail_with is not None:
            raise self.fail_with
        return self._vector(text)


class StatusError(Exception):
    """An SDK-style error carrying an HTTP status code (unmapped by LangChain)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
