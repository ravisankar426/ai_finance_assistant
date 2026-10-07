"""The LangGraph workflow (design section 6).

    START -> input_guard -> router --(Send per intent)--> agent_<name> ... -> synthesizer
          -> output_guard -> END

- ``router`` returns intents; :func:`dispatch` turns them into ``Send`` objects, so several
  agents run **in parallel** in one step and their results are merged by the
  ``agent_results`` reducer (REQ-WF-02).
- Conversation memory comes from the checkpointer, keyed by ``thread_id`` (REQ-WF-04).
- :func:`build_graph` takes its dependencies as arguments (tests pass fakes);
  :func:`build_default_graph` wires the real models and builds them **at startup**, so a
  missing API key stops the app immediately instead of failing on the first question.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from src.agents.base import Agent, run_safely
from src.agents.placeholder import PlaceholderAgent
from src.agents.qa import QAAgent
from src.core.config import Settings, get_settings
from src.core.guards import OUT_OF_SCOPE_REPLY, redact_user_text, with_disclaimer
from src.core.llm import get_chat_model
from src.core.models import AGENT_LABELS, AGENT_NAMES, AgentResult, Citation, UserProfile
from src.rag.knowledge_base import load_articles
from src.rag.retriever import KeywordRetriever
from src.utils.logging import get_logger, new_request_id
from src.workflow.router import RouteDecision, Router
from src.workflow.state import AgentInput, GraphState

log = get_logger(__name__)

OUT_OF_SCOPE_NODE = "out_of_scope"


# Node protocols: LangGraph matches nodes by a parameter named ``state``, which a plain
# ``Callable[[X], Y]`` annotation (no parameter names) cannot express.
class StateNode(Protocol):
    """A node that receives the full graph state."""

    def __call__(self, state: GraphState) -> dict[str, Any]: ...  # noqa: D102


class AgentNode(Protocol):
    """A node that receives one agent's ``Send`` payload."""

    def __call__(self, state: AgentInput) -> dict[str, Any]: ...  # noqa: D102


def agent_node_name(agent: str) -> str:
    """Graph node name for an agent."""
    return f"agent_{agent}"


# --- nodes ----------------------------------------------------------------------------------


def input_guard(state: GraphState) -> dict[str, Any]:
    """Redact PII in the new message, reset per-turn fields, default the profile."""
    latest = state["messages"][-1]
    clean = redact_user_text(latest.text)
    update: dict[str, Any] = {
        "agent_results": None,  # clears last turn's results (see add_or_reset)
        "request_id": state.get("request_id") or new_request_id(),
    }
    if clean != latest.text:
        log.info("pii_redacted", message_id=latest.id)
        update["messages"] = [HumanMessage(clean, id=latest.id)]  # same id -> replaced in place
    if "profile" not in state:
        update["profile"] = UserProfile()
    return update


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
        return {"agent_results": [run_safely(agent, state)]}

    node.__name__ = agent_node_name(agent.name)
    return node


def synthesizer(state: GraphState) -> dict[str, Any]:
    """Merge agent results into one answer (deterministic today; LLM merge on Day 7)."""
    order = {name: i for i, name in enumerate((*AGENT_NAMES, "out_of_scope"))}
    results = sorted(state.get("agent_results", []), key=lambda r: order.get(r.agent, 99))
    ok = [r for r in results if r.error is None and r.answer]
    failed = [r for r in results if r.error is not None]

    if len(ok) == 1:
        parts = [ok[0].answer]
    else:
        parts = [f"**{AGENT_LABELS.get(r.agent, r.agent)}**\n\n{r.answer}" for r in ok]
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


def output_guard(state: GraphState) -> dict[str, Any]:
    """Add the disclaimer to finance answers (REQ-GR-03) and record the turn in history."""
    answer = state["final_answer"]
    only_refusal = all(r.agent == "out_of_scope" for r in state.get("agent_results", []))
    final = answer if only_refusal else with_disclaimer(answer)
    # History keeps the answer without the disclaimer, so it doesn't clutter later prompts.
    return {"final_answer": final, "messages": [AIMessage(answer)]}


# --- assembly -------------------------------------------------------------------------------


def build_graph(
    *,
    router: StateNode,
    agents: Mapping[str, Agent],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble and compile the workflow from injected parts."""
    missing = sorted(set(AGENT_NAMES) - set(agents))
    if missing:
        raise ValueError(f"No agent registered for: {missing}")
    graph = StateGraph(GraphState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("router", router)
    graph.add_node(OUT_OF_SCOPE_NODE, out_of_scope_node, input_schema=AgentInput)
    for name in AGENT_NAMES:
        graph.add_node(
            agent_node_name(name), make_agent_node(agents[name]), input_schema=AgentInput
        )
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("output_guard", output_guard)

    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "router")
    graph.add_conditional_edges(
        "router", dispatch, [OUT_OF_SCOPE_NODE, *(agent_node_name(n) for n in AGENT_NAMES)]
    )
    for node in (OUT_OF_SCOPE_NODE, *(agent_node_name(n) for n in AGENT_NAMES)):
        graph.add_edge(node, "synthesizer")
    graph.add_edge("synthesizer", "output_guard")
    graph.add_edge("output_guard", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver(serde=checkpoint_serializer()))


# Our own types stored in checkpoints. LangGraph refuses to deserialize unregistered classes
# (a safety measure against loading arbitrary objects from a tampered store), so we allowlist
# exactly these.
CHECKPOINT_TYPES: tuple[type[Any], ...] = (AgentResult, Citation, UserProfile)


def checkpoint_serializer() -> JsonPlusSerializer:
    """Build a serializer that only rebuilds LangChain types plus our explicit allowlist."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in CHECKPOINT_TYPES]
    )


def build_default_graph(settings: Settings | None = None) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Wire real models, the knowledge base, and an in-memory checkpointer."""
    settings = settings or get_settings()
    history = settings.workflow.history_messages
    retriever = KeywordRetriever(
        load_articles(settings.rag.knowledge_base_dir), min_score=settings.rag.min_score
    )
    agents: dict[str, Agent] = {name: PlaceholderAgent(name) for name in AGENT_NAMES}
    agents["qa"] = QAAgent(
        get_chat_model("agent", settings=settings),
        retriever,
        top_k=settings.rag.top_k,
        history_messages=history,
    )
    router = Router(
        get_chat_model("router", structured_output=RouteDecision, settings=settings),
        history_messages=history,
    )
    log.info("graph_built", agents=sorted(agents), articles=len(retriever.articles))
    return build_graph(router=router, agents=agents)
