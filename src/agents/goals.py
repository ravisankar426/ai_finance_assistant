"""Goal Planning agent (REQ-GP-01..05, REQ-WF-08..10).

check inputs (goal amount + horizon required; risk profile from the message, the remembered
profile, or a follow-up question) -> compute (goal engine) -> retrieve concepts -> explain.
A stated risk tolerance is remembered in the user profile for later turns (REQ-WF-08).
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.agents.grounded import LEVEL_STYLE, renumber_citations
from src.core.config import PlanningConfig
from src.core.goal_engine import describe_plan, plan_goal
from src.core.llm import CONFIG_ERRORS
from src.core.models import AgentResult
from src.rag.retriever import Retriever
from src.utils.logging import get_logger
from src.workflow.state import AgentInput

log = get_logger(__name__)

ASK_GOAL = (
    "Happy to help you plan! What's the goal, how much do you want (in today's dollars), and in "
    "how many years? For example: “$60,000 for a house down payment in 5 years.” If you already "
    "have savings for it, or know how much you can set aside each month, tell me that too."
)
ASK_RISK = (
    "One more thing: how would you describe your risk tolerance — conservative, moderate, or "
    "aggressive? (A hint: if your investments dropped 20%, would you sell, wait it out, or invest "
    "more?) You can also take the short questionnaire in the Goals tab."
)


class GoalRequest(BaseModel):
    """Goal parameters stated in the conversation."""

    goal: str | None = Field(default=None, description="Short name, e.g. 'retirement', 'house'.")
    target_amount: float | None = Field(default=None, description="Dollar amount wanted.")
    years: float | None = Field(
        default=None, description="Years until the money is needed (convert ages or dates)."
    )
    current_savings: float | None = Field(default=None, description="Already saved for this goal.")
    monthly_contribution: float | None = Field(
        default=None, description="What the user says they can save per month."
    )
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] | None = Field(
        default=None, description="Only if the user states or clearly describes it."
    )


EXTRACT_PROMPT = """Extract the financial goal from the conversation (latest message first).
Use null for anything not stated; do not guess. Convert "by age 65, I'm 40" to years=25 and
"$1,500 a month" to monthly_contribution=1500. Map "I can't handle losses" -> conservative,
"I'm fine with big swings for growth" -> aggressive, "somewhere in the middle" -> moderate."""

SYSTEM = """You are the Goal Planning agent of an EDUCATIONAL personal-finance assistant.
Rules:
- Explain ONLY the facts in the PLAN block; never compute new numbers.
- Explain inflation-adjusted targets, the three return scenarios (illustrative assumptions, not
  forecasts), the required monthly saving, and whether the plan is on track.
- If the goal is at risk, walk through the levers in the block (save more, allow more time, or
  adjust the target) neutrally.
- Present the model allocation as an educational illustration for the risk profile and horizon,
  never as a recommendation; don't name specific funds or securities.
- Cite the numbered concept sources like [1] when explaining concepts.
- Keep it under 250 words; no disclaimer.
{level_style}

PLAN (computed by the system):
{facts}
{assumption_note}

Concept sources:
{sources}"""


class GoalsAgent:
    """Turns a stated goal into a computed plan."""

    name = "goals"

    def __init__(
        self,
        model: Runnable[Any, Any],
        extractor: Runnable[Any, Any],
        retriever: Retriever,
        config: PlanningConfig,
        *,
        history_messages: int = 6,
    ) -> None:
        self.model = model
        self.extractor = extractor  # structured output -> GoalRequest
        self.retriever = retriever
        self.config = config
        self.history_messages = history_messages

    def _extract(self, inp: AgentInput) -> GoalRequest:
        recent = inp["messages"][-(self.history_messages + 1) :]
        transcript = "\n".join(f"{m.type}: {m.text}" for m in reversed(recent))
        try:
            req: GoalRequest = self.extractor.invoke(
                [SystemMessage(EXTRACT_PROMPT), HumanMessage(transcript)]
            )
            return req
        except CONFIG_ERRORS:
            raise
        except Exception as exc:
            log.warning("goal_extraction_failed", error_type=type(exc).__name__)
            return GoalRequest()

    def run(self, inp: AgentInput) -> AgentResult:
        """Plan the goal, or ask for what's missing."""
        req = self._extract(inp)
        profile = inp["profile"]
        risk = req.risk_tolerance or profile.risk_tolerance
        updates: dict[str, Any] = {}
        if req.risk_tolerance and req.risk_tolerance != profile.risk_tolerance:
            profile = profile.model_copy(update={"risk_tolerance": req.risk_tolerance})
            updates["profile"] = profile  # remembered for later turns (REQ-WF-08)

        if not req.target_amount or not req.years:  # REQ-WF-10: ask, don't assume
            return AgentResult(agent=self.name, follow_up_question=ASK_GOAL, state_updates=updates)
        if risk is None:
            return AgentResult(agent=self.name, follow_up_question=ASK_RISK, state_updates=updates)

        plan = plan_goal(
            goal=req.goal or "your goal",
            target_today=req.target_amount,
            years=req.years,
            current_savings=req.current_savings or 0.0,
            planned_monthly=req.monthly_contribution,
            risk_profile=risk,
            cfg=self.config,
        )
        if req.goal and req.goal not in profile.goals:
            profile = profile.model_copy(
                update={"goals": [*profile.goals, req.goal], "horizon_years": round(req.years)}
            )
            updates["profile"] = profile
        assumption = (
            "ASSUMPTION: starting savings not stated, so $0 was used — mention this and invite "
            "the user to share it."
            if req.current_savings is None
            else ""
        )
        concepts = self.retriever.retrieve(
            f"{plan.goal} compound interest inflation time horizon asset allocation risk tolerance",
            k=4,
            categories=("retirement", "portfolio", "basics"),
        )
        sources = "\n\n".join(
            f"[{i}] {c.title} — {c.section}\n{c.text}" for i, c in enumerate(concepts, 1)
        )
        system = SYSTEM.format(
            level_style=LEVEL_STYLE[profile.knowledge_level],
            facts=describe_plan(plan),
            assumption_note=assumption,
            sources=sources or "(none)",
        )
        reply = self.model.invoke([SystemMessage(system), HumanMessage(inp["question"])])
        body, citations = renumber_citations(reply.text.strip(), [c.citation() for c in concepts])
        footer = (
            "_Returns and inflation are illustrative assumptions (see config), not forecasts; "
            "actual results will differ._"
        )
        return AgentResult(
            agent=self.name,
            answer=f"{body}\n\n{footer}",
            citations=citations,
            data={"plan": plan.model_dump(mode="json")},
            state_updates=updates,
        )
