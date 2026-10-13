"""LangGraph state (design section 5).

Reducers decide how a node's return value is merged into the state:
- ``messages`` uses ``add_messages``: appends, or replaces a message with the same id
  (the input guard uses that to swap in the PII-redacted text).
- ``agent_results`` uses :func:`add_or_reset`: parallel agents each return
  ``{"agent_results": [result]}`` and the results are concatenated (fan-in). Because the
  checkpointer keeps state across turns, the input guard returns ``None`` to clear last
  turn's results.
- Everything else is "last write wins".
"""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.core.models import AgentResult, Citation, Holding, Intent, UserProfile


def add_or_reset(
    left: list[AgentResult] | None, right: list[AgentResult] | None
) -> list[AgentResult]:
    """Concatenate parallel agent results; ``None`` clears the list for a new turn."""
    if right is None:
        return []
    return [*(left or []), *right]


class GraphState(TypedDict, total=False):
    """Shared state of one conversation thread."""

    messages: Annotated[list[AnyMessage], add_messages]
    profile: UserProfile
    portfolio: list[Holding]  # remembered holdings (REQ-PF-01), from chat or the Portfolio tab
    request_id: str
    question: str  # standalone version of the latest user message (router output)
    intents: list[Intent]
    agent_results: Annotated[list[AgentResult], add_or_reset]
    final_answer: str
    citations: list[Citation]
    blocked: bool  # input guard removed everything (prompt injection only)


class AgentInput(TypedDict):
    """The slice of state each agent receives via ``Send`` (design section 7a)."""

    intent: Intent
    question: str
    messages: list[AnyMessage]
    profile: UserProfile
    request_id: str
    portfolio: NotRequired[list[Holding] | None]
    others: NotRequired[list[Intent]]  # other agents answering this turn (multi-intent)
