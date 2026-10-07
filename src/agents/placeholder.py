"""Stand-in for agents that are not built yet, so every intent has a working route."""

from __future__ import annotations

from src.core.models import AGENT_LABELS, AgentResult
from src.workflow.state import AgentInput


class PlaceholderAgent:
    """Explains politely that a specialist is coming soon."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, inp: AgentInput) -> AgentResult:
        """Return a 'coming soon' note for this specialist."""
        label = AGENT_LABELS.get(self.name, self.name)
        return AgentResult(
            agent=self.name,
            answer=f"The {label} assistant is coming soon — it's still being built.",
        )
