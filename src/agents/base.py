"""Agent contract and the safety wrapper every agent node runs inside (design section 7).

An agent turns an :class:`AgentInput` into an :class:`AgentResult`. ``run_safely`` adds the
graph-level guarantees so individual agents don't have to:
- an agent crash becomes ``AgentResult(error=...)`` and the other agents' answers still reach
  the user (REQ-WF-05);
- configuration errors are re-raised, never swallowed (REQ-LLM-07).
"""

from __future__ import annotations

import time
from typing import Protocol

from src.core.llm import CONFIG_ERRORS
from src.core.models import AGENT_LABELS, AgentResult
from src.utils.logging import get_logger
from src.workflow.state import AgentInput

log = get_logger(__name__)


class Agent(Protocol):
    """Anything with a name that can answer an :class:`AgentInput`."""

    name: str

    def run(self, inp: AgentInput) -> AgentResult:
        """Produce this agent's contribution to the answer."""
        ...


def run_safely(agent: Agent, inp: AgentInput) -> AgentResult:
    """Run ``agent``; convert unexpected failures into a degraded result."""
    start = time.perf_counter()
    try:
        result = agent.run(inp)
    except CONFIG_ERRORS:
        raise
    except Exception as exc:
        log.exception("agent_failed", agent=agent.name, error_type=type(exc).__name__)
        label = AGENT_LABELS.get(agent.name, agent.name)
        result = AgentResult(
            agent=agent.name,
            error=f"The {label} assistant is temporarily unavailable. Please try again shortly.",
        )
    log.info(
        "agent_done",
        agent=agent.name,
        ok=result.error is None,
        citations=len(result.citations),
        latency_ms=round((time.perf_counter() - start) * 1000),
    )
    return result
