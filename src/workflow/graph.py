"""The LangGraph workflow (design section 6).

    START -> input_guard -> router -(Send per intent)-> agent_<name>... -> synthesizer
             |                                                              ^
             +--(only a prompt-injection attempt)--> guard_refusal ---------+
    synthesizer -> output_guard -(thread too long)-> compact_history -> END

- ``input_guard``: PII redaction (GR-05) and prompt-injection removal (GR-02).
- ``router`` + :func:`dispatch`: intents -> parallel ``Send`` fan-out (WF-01, WF-02).
- ``synthesizer``: merges results; adds a short overview when several agents answered,
  rejected if it introduces numbers the agents didn't produce (WF-02).
- ``output_guard``: directive/guarantee rewriting (GR-04, GR-07) and the disclaimer (GR-03).
- ``compact_history``: summarizes old turns in long threads (WF-07).
- :func:`build_graph` takes every dependency as an argument (tests pass fakes); optional parts
  (rewriter, overview model, compactor) can be omitted. :func:`build_default_graph` wires the
  real ones at startup, so a missing API key fails immediately.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from src.agents.base import Agent, run_safely
from src.agents.entities import TickerExtractor, TickerRequest
from src.agents.goals import GoalRequest, GoalsAgent
from src.agents.market import MarketAgent
from src.agents.news import NewsAgent
from src.agents.portfolio import HoldingsRequest, PortfolioAgent
from src.agents.qa import QAAgent
from src.agents.tax import TaxAgent
from src.core.config import Settings, WorkflowConfig, get_settings
from src.core.guards import (
    INJECTION_REPLY,
    OUT_OF_SCOPE_REPLY,
    ComplianceRewriter,
    redact_user_text,
    remove_injection,
    with_disclaimer,
)
from src.core.llm import get_chat_model, get_embeddings
from src.core.models import (
    AGENT_LABELS,
    AGENT_NAMES,
    STATE_UPDATE_KEYS,
    AgentResult,
    Citation,
    UserProfile,
)
from src.data.market_service import MarketDataService, build_market_service
from src.rag.index import load_or_build
from src.rag.knowledge_base import load_articles
from src.rag.retriever import HybridRetriever
from src.utils.logging import get_logger, new_request_id
from src.workflow.memory import (
    CHECKPOINT_TYPES,
    HistoryCompactor,
    build_checkpointer,
    checkpoint_serializer,
)
from src.workflow.router import RouteDecision, Router
from src.workflow.state import AgentInput, GraphState

log = get_logger(__name__)

OUT_OF_SCOPE_NODE = "out_of_scope"
GUARD_REFUSAL_NODE = "guard_refusal"
NON_FINANCE_AGENTS = {"out_of_scope", "guard"}  # answers that don't get the disclaimer

__all__ = ["CHECKPOINT_TYPES", "build_default_graph", "build_graph", "checkpoint_serializer"]


# Node protocols: LangGraph matches nodes by a parameter named ``state``, which a plain
# ``Callable[[X], Y]`` annotation (no parameter names) cannot express.
class StateNode(Protocol):
    """A node that receives the full graph state."""

    def __call__(self, state: GraphState) -> dict[str, Any]: ...


class AgentNode(Protocol):
    """A node that receives one agent's ``Send`` payload."""

    def __call__(self, state: AgentInput) -> dict[str, Any]: ...


def agent_node_name(agent: str) -> str:
    """Graph node name for an agent."""
    return f"agent_{agent}"


# --- input side -------------------------------------------------------------------------------


def input_guard(state: GraphState) -> dict[str, Any]:
    """Redact PII, remove prompt-injection sentences, reset per-turn fields."""
    latest = state["messages"][-1]
    clean = redact_user_text(latest.text)
    clean, removed = remove_injection(clean)  # REQ-GR-02
    update: dict[str, Any] = {
        "agent_results": None,  # clears last turn's results (see add_or_reset)
        "request_id": state.get("request_id") or new_request_id(),
        "blocked": bool(removed) and not clean.strip(),
    }
    if removed:
        log.warning("prompt_injection_removed", sentences=len(removed), remainder=bool(clean))
    if clean != latest.text:
        update["messages"] = [HumanMessage(clean or "[removed]", id=latest.id)]  # replace in place
    if "profile" not in state:
        update["profile"] = UserProfile()
    return update


def after_input_guard(state: GraphState) -> str:
    """Skip routing when the message was nothing but an injection attempt."""
    return GUARD_REFUSAL_NODE if state.get("blocked") else "router"


def guard_refusal(state: GraphState) -> dict[str, Any]:
    """Decline instruction-override requests (REQ-GR-02)."""
    return {"agent_results": [AgentResult(agent="guard", answer=INJECTION_REPLY)]}


def dispatch(state: GraphState) -> list[Send]:
    """Fan out: one ``Send`` per intent, each carrying only what that agent needs."""
    sends = []
    for intent in state["intents"]:
        payload: AgentInput = {
            "intent": intent,
            "question": state["question"],
            "messages": state["messages"],
            "profile": state["profile"],
            "request_id": state["request_id"],
            "portfolio": state.get("portfolio"),
        }
        node = OUT_OF_SCOPE_NODE if intent == "out_of_scope" else agent_node_name(intent)
        sends.append(Send(node, payload))
    return sends


def out_of_scope_node(state: AgentInput) -> dict[str, Any]:
    """Polite refusal for non-finance requests (REQ-GR-01)."""
    return {"agent_results": [AgentResult(agent="out_of_scope", answer=OUT_OF_SCOPE_REPLY)]}


def make_agent_node(agent: Agent) -> AgentNode:
    """Wrap an agent so its result lands in ``agent_results`` (fan-in via the reducer)."""

    def node(state: AgentInput) -> dict[str, Any]:
        result = run_safely(agent, state)
        # Agents may persist whitelisted facts (holdings, profile) for later turns.
        updates = {k: v for k, v in result.state_updates.items() if k in STATE_UPDATE_KEYS}
        return {"agent_results": [result], **updates}

    node.__name__ = agent_node_name(agent.name)
    return node


# --- output side ------------------------------------------------------------------------------

OVERVIEW_PROMPT = """Several specialist assistants answered one user question (below). Write ONE or
TWO plain sentences that tie their answers together for the user. Do not add facts, numbers,
advice, or recommendations that aren't in the answers. No headings, no lists."""

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def make_synthesizer(overview_model: Runnable[Any, Any] | None) -> StateNode:
    """Merge agent results into one answer (REQ-WF-02, REQ-WF-05, REQ-WF-10)."""

    def synthesizer(state: GraphState) -> dict[str, Any]:
        order = {name: i for i, name in enumerate((*AGENT_NAMES, "out_of_scope", "guard"))}
        results = sorted(state.get("agent_results", []), key=lambda r: order.get(r.agent, 99))
        ok = [r for r in results if r.error is None and (r.answer or r.follow_up_question)]
        failed = [r for r in results if r.error is not None]

        answered = [r for r in ok if r.answer]
        if len(answered) == 1:
            parts = [answered[0].answer]
        else:
            parts = [f"**{AGENT_LABELS.get(r.agent, r.agent)}**\n\n{r.answer}" for r in answered]
            overview = _overview(overview_model, state.get("question", ""), answered)
            if overview:
                parts.insert(0, f"**In short:** {overview}")
        parts += [f"> ⚠️ {r.error}" for r in failed]  # partial answer + notice (REQ-WF-05)
        parts += [f"**Quick question:** {r.follow_up_question}" for r in ok if r.follow_up_question]
        answer = "\n\n".join(parts) or "Sorry, I couldn't produce an answer. Please try again."

        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for r in ok:
            for c in r.citations:
                if (c.title, c.url) not in seen:
                    seen.add((c.title, c.url))
                    citations.append(c)
        return {"final_answer": answer, "citations": citations}

    return synthesizer


def _overview(
    model: Runnable[Any, Any] | None, question: str, answered: list[AgentResult]
) -> str | None:
    """Return a 1-2 sentence bridge, or None if it invents numbers the agents didn't give."""
    if model is None:
        return None
    combined = "\n\n".join(f"[{AGENT_LABELS.get(r.agent, r.agent)}]\n{r.answer}" for r in answered)
    try:
        reply = model.invoke(
            [SystemMessage(OVERVIEW_PROMPT), HumanMessage(f"Question: {question}\n\n{combined}")]
        )
        text = str(reply.text).strip()
    except Exception as exc:  # optional nicety; the sections stand on their own
        log.warning("overview_failed", error_type=type(exc).__name__)
        return None
    invented = _numbers(text) - _numbers(combined)
    if invented or not text:
        log.warning("overview_rejected", invented_numbers=sorted(invented))
        return None
    return text


def make_output_guard(rewriter: ComplianceRewriter | None) -> StateNode:
    """Enforce compliance (GR-04, GR-07), add the disclaimer (GR-03), record the turn."""

    def output_guard(state: GraphState) -> dict[str, Any]:
        answer = state["final_answer"]
        if rewriter is not None:
            answer, flagged = rewriter.enforce(answer)
            if flagged:
                log.warning("compliance_flags", count=len(flagged))
        results = state.get("agent_results", [])
        non_finance = bool(results) and all(r.agent in NON_FINANCE_AGENTS for r in results)
        final = answer if non_finance else with_disclaimer(answer)
        # History keeps the answer without the disclaimer, so it doesn't clutter later prompts.
        return {"final_answer": final, "messages": [AIMessage(answer)]}

    return output_guard


def make_compactor_node(compactor: HistoryCompactor) -> StateNode:
    """Summarize old turns once the thread is long (REQ-WF-07)."""

    def compact_history(state: GraphState) -> dict[str, Any]:
        return {"messages": compactor(state["messages"])}

    return compact_history


# --- assembly ---------------------------------------------------------------------------------


def build_graph(
    *,
    router: StateNode,
    agents: Mapping[str, Agent],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    rewriter: ComplianceRewriter | None = None,
    overview_model: Runnable[Any, Any] | None = None,
    compactor: HistoryCompactor | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble and compile the workflow from injected parts."""
    missing = sorted(set(AGENT_NAMES) - set(agents))
    if missing:
        raise ValueError(f"No agent registered for: {missing}")
    graph = StateGraph(GraphState)
    graph.add_node("input_guard", input_guard)
    graph.add_node(GUARD_REFUSAL_NODE, guard_refusal)
    graph.add_node("router", router)
    graph.add_node(OUT_OF_SCOPE_NODE, out_of_scope_node, input_schema=AgentInput)
    for name in AGENT_NAMES:
        graph.add_node(
            agent_node_name(name), make_agent_node(agents[name]), input_schema=AgentInput
        )
    graph.add_node("synthesizer", make_synthesizer(overview_model))
    graph.add_node("output_guard", make_output_guard(rewriter))

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", after_input_guard, [GUARD_REFUSAL_NODE, "router"])
    graph.add_edge(GUARD_REFUSAL_NODE, "synthesizer")
    graph.add_conditional_edges(
        "router", dispatch, [OUT_OF_SCOPE_NODE, *(agent_node_name(n) for n in AGENT_NAMES)]
    )
    for node in (OUT_OF_SCOPE_NODE, *(agent_node_name(n) for n in AGENT_NAMES)):
        graph.add_edge(node, "synthesizer")
    graph.add_edge("synthesizer", "output_guard")
    if compactor is not None:
        graph.add_node("compact_history", make_compactor_node(compactor))
        graph.add_conditional_edges(
            "output_guard",
            lambda s: "compact_history" if compactor.needed(s["messages"]) else END,
            ["compact_history", END],
        )
        graph.add_edge("compact_history", END)
    else:
        graph.add_edge("output_guard", END)
    return graph.compile(checkpointer=checkpointer or build_checkpointer(WorkflowConfig()))


def build_retriever(settings: Settings) -> HybridRetriever:
    """Load (or build) the knowledge-base index and wrap it in the hybrid retriever."""
    rag = settings.rag
    embeddings = get_embeddings(settings)
    store = load_or_build(
        load_articles(rag.knowledge_base_dir), embeddings, settings.llm.embeddings.model, rag
    )
    return HybridRetriever(
        store,
        embeddings,
        fetch_k=rag.fetch_k,
        rrf_k=rag.rrf_k,
        min_cosine=rag.min_cosine,
        min_keyword_coverage=rag.min_keyword_coverage,
    )


def build_default_graph(
    settings: Settings | None = None,
    *,
    market: MarketDataService | None = None,
    retriever: HybridRetriever | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Wire real models, the knowledge base, market data, guards, and the checkpointer.

    ``market``/``retriever`` can be passed in so the API shares one cache and one index.
    """
    settings = settings or get_settings()
    wf = settings.workflow
    history = wf.history_messages
    rag = settings.rag
    retriever = retriever or build_retriever(settings)
    agent_model = get_chat_model("agent", settings=settings)
    market = market or build_market_service(settings)
    extractor = TickerExtractor(
        get_chat_model("router", structured_output=TickerRequest, settings=settings)
    )
    agents: dict[str, Agent] = {
        "qa": QAAgent(agent_model, retriever, top_k=rag.top_k, history_messages=history),
        "tax": TaxAgent(agent_model, retriever, top_k=rag.top_k, history_messages=history),
        "market": MarketAgent(agent_model, extractor, market, retriever),
        "news": NewsAgent(
            agent_model,
            extractor,
            market,
            retriever,
            lookback_days=settings.market.news_lookback_days,
        ),
        "portfolio": PortfolioAgent(
            agent_model,
            get_chat_model("router", structured_output=HoldingsRequest, settings=settings),
            market,
            retriever,
            settings.portfolio,
        ),
        "goals": GoalsAgent(
            agent_model,
            get_chat_model("router", structured_output=GoalRequest, settings=settings),
            retriever,
            settings.planning,
            history_messages=history,
        ),
    }
    router = Router(
        get_chat_model("router", structured_output=RouteDecision, settings=settings),
        history_messages=history,
    )
    synth_model = get_chat_model("synthesizer", settings=settings)
    log.info("graph_built", agents=sorted(agents), checkpointer=wf.checkpointer)
    return build_graph(
        router=router,
        agents=agents,
        checkpointer=build_checkpointer(wf),
        rewriter=ComplianceRewriter(get_chat_model("guard", settings=settings)),
        overview_model=synth_model if wf.combined_overview else None,
        compactor=HistoryCompactor(synth_model, wf),
    )
