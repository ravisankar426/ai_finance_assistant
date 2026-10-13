"""Conversation memory: checkpointers and history compaction (REQ-WF-04, REQ-WF-07, ADR-11)."""

from __future__ import annotations

import sqlite3
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.core.config import WorkflowConfig
from src.core.models import AgentResult, Citation, Holding, UserProfile
from src.utils.logging import get_logger

log = get_logger(__name__)

# Our own types stored in checkpoints. LangGraph refuses to deserialize unregistered classes
# (a safety measure against loading arbitrary objects from a tampered store), so we allowlist
# exactly these.
CHECKPOINT_TYPES: tuple[type[Any], ...] = (AgentResult, Citation, Holding, UserProfile)

SUMMARY_PREFIX = "Summary of the earlier conversation:"

SUMMARIZE_PROMPT = """Summarize this conversation between a user and a personal-finance education
assistant in at most 120 words, for the assistant's own memory. Keep: the user's goals, holdings,
risk tolerance, knowledge level, numbers they gave, and open questions. Omit greetings."""


def checkpoint_serializer() -> JsonPlusSerializer:
    """Build a serializer that only rebuilds LangChain types plus our explicit allowlist."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in CHECKPOINT_TYPES]
    )


def build_checkpointer(cfg: WorkflowConfig) -> BaseCheckpointSaver[Any]:
    """In-memory (default) or SQLite-backed checkpointer, both with the strict serializer."""
    if cfg.checkpointer == "sqlite":
        cfg.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(cfg.sqlite_path, check_same_thread=False)  # Streamlit/API threads
        log.info("checkpointer", kind="sqlite", path=str(cfg.sqlite_path))
        return SqliteSaver(conn, serde=checkpoint_serializer())
    return InMemorySaver(serde=checkpoint_serializer())


class HistoryCompactor:
    """Replace old messages with a summary once the thread gets long (REQ-WF-07)."""

    def __init__(self, model: Runnable[Any, Any] | None, cfg: WorkflowConfig) -> None:
        self.model = model
        self.cfg = cfg

    def needed(self, messages: list[AnyMessage]) -> bool:
        """Return True when the thread is over the configured length."""
        return self.model is not None and len(messages) > self.cfg.summarize_after_messages

    def __call__(self, messages: list[AnyMessage]) -> list[Any]:
        """Return add_messages updates: remove old messages, add one summary message."""
        old = messages[: -self.cfg.keep_recent_messages]
        if not old or self.model is None:
            return []
        transcript = "\n".join(f"{m.type}: {m.text}" for m in old)
        try:
            reply = self.model.invoke([SystemMessage(SUMMARIZE_PROMPT), HumanMessage(transcript)])
        except Exception as exc:  # summarization is an optimization; never fail the turn
            log.warning("history_summary_failed", error_type=type(exc).__name__)
            return []
        log.info("history_compacted", removed=len(old), kept=self.cfg.keep_recent_messages)
        # add_messages appends, so rebuild the list to put the summary FIRST:
        # remove everything, then re-add [summary, *recent].
        return [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(f"{SUMMARY_PREFIX} {reply.text.strip()}"),
            *messages[-self.cfg.keep_recent_messages :],
        ]
