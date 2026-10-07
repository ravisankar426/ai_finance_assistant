"""End-to-end workflow tests with fake models (no network)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import ModelAuthenticationError
from langchain_core.messages import AIMessageChunk, HumanMessage

import src.workflow.graph as graph_module
from src.agents.placeholder import PlaceholderAgent
from src.agents.qa import QAAgent
from src.core.config import Settings
from src.core.guards import DISCLAIMER
from src.core.models import AGENT_NAMES, AgentResult, UserProfile
from src.workflow.graph import build_graph
from src.workflow.router import RouteDecision, Router
from src.workflow.state import AgentInput, add_or_reset
from tests.fakes import HashingEmbeddings, ScriptedChatModel, route_json, seen_text
from tests.kb import offline_retriever, rag_config


def make_graph(
    settings: Settings,
    *,
    intents: list[str],
    qa_reply: str = "Index funds track an index [1].",
    extra_agents: dict[str, Any] | None = None,
) -> tuple[Any, ScriptedChatModel, ScriptedChatModel]:
    router_model = ScriptedChatModel(reply=route_json(intents))
    qa_model = ScriptedChatModel(reply=qa_reply)
    retriever = offline_retriever()
    agents: dict[str, Any] = {n: PlaceholderAgent(n) for n in AGENT_NAMES}
    agents["qa"] = QAAgent(qa_model, retriever)
    agents.update(extra_agents or {})
    router = Router(router_model.with_structured_output(RouteDecision))
    return build_graph(router=router, agents=agents), router_model, qa_model


def _ask(graph: Any, text: str, thread: str = "t1", **extra: Any) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}}
    return graph.invoke({"messages": [HumanMessage(text)], **extra}, config)  # type: ignore[no-any-return]


def test_single_intent_answer_with_citation_and_disclaimer(settings: Settings) -> None:
    """REQ-WF-01, REQ-QA-01, REQ-GR-03: routed to Q&A, cited, disclaimer appended."""
    graph, _, _ = make_graph(settings, intents=["qa"])
    out = _ask(graph, "What is an index fund?")
    assert out["intents"] == ["qa"]
    assert [r.agent for r in out["agent_results"]] == ["qa"]
    assert out["final_answer"].startswith("Index funds track an index [1].")
    assert out["final_answer"].endswith(DISCLAIMER)
    assert [c.title for c in out["citations"]] == ["Index Funds"]


def test_multi_intent_fans_out_in_parallel(settings: Settings) -> None:
    """REQ-WF-02: several intents -> several agents in one step -> one synthesized answer."""
    graph, _, _ = make_graph(settings, intents=["qa", "market", "news"])
    out = _ask(graph, "What is an index fund, and NVDA's price and news?")
    assert sorted(r.agent for r in out["agent_results"]) == ["market", "news", "qa"]
    answer = out["final_answer"]
    assert (
        answer.index("**Finance Q&A**")
        < answer.index("**Market Analysis**")
        < answer.index("**News Synthesizer**")
    )


def test_out_of_scope_refusal_has_no_disclaimer(settings: Settings) -> None:
    """REQ-GR-01: off-topic requests get a polite redirect, not finance boilerplate."""
    graph, _, qa_model = make_graph(settings, intents=["out_of_scope"])
    out = _ask(graph, "Give me a lasagna recipe")
    assert "personal-finance education assistant" in out["final_answer"]
    assert DISCLAIMER not in out["final_answer"]
    assert qa_model.calls == 0


def test_pii_redacted_before_any_llm_and_in_history(settings: Settings) -> None:
    """REQ-GR-05: the router, the agent, and the stored history never see the raw SSN."""
    graph, router_model, qa_model = make_graph(settings, intents=["qa"])
    out = _ask(graph, "What is an index fund? my ssn is 123-45-6789")
    for model in (router_model, qa_model):
        assert "123-45-6789" not in seen_text(model)
    # The router sees the (redacted) raw message; the agent gets the router's standalone question.
    assert "[REDACTED_SSN]" in seen_text(router_model)
    assert "123-45-6789" not in out["messages"][0].text


def test_memory_across_turns_and_isolated_threads(settings: Settings) -> None:
    """REQ-WF-04: a thread remembers earlier turns; another thread doesn't see them."""
    graph, _, qa_model = make_graph(settings, intents=["qa"])
    _ask(graph, "What is an index fund?", thread="a")
    second = _ask(graph, "What about its fees?", thread="a")
    assert len(second["messages"]) == 4  # user, assistant, user, assistant
    assert [r.agent for r in second["agent_results"]] == ["qa"]  # last turn's results cleared
    assert "What is an index fund?" in seen_text(qa_model)  # history reached the agent
    other = _ask(graph, "What is an ETF?", thread="b")
    assert len(other["messages"]) == 2


def test_profile_defaults_and_is_kept(settings: Settings) -> None:
    """REQ-WF-08: a default profile is created and a supplied one persists in the thread."""
    graph, _, qa_model = make_graph(settings, intents=["qa"])
    assert _ask(graph, "What is an index fund?", thread="p")["profile"] == UserProfile()
    _ask(graph, "What is an ETF?", thread="p", profile=UserProfile(knowledge_level="advanced"))
    out = _ask(graph, "And bonds?", thread="p")
    assert out["profile"].knowledge_level == "advanced"
    assert "experienced" in seen_text(qa_model)


def test_agent_failure_gives_partial_answer(settings: Settings) -> None:
    """REQ-WF-05: one agent crashing doesn't lose the other agents' answers."""

    class Broken:
        name = "market"

        def run(self, inp: AgentInput) -> AgentResult:
            raise RuntimeError("provider exploded")

    graph, _, _ = make_graph(settings, intents=["qa", "market"], extra_agents={"market": Broken()})
    out = _ask(graph, "What is an index fund and NVDA's price?")
    assert "Index funds track an index" in out["final_answer"]
    assert "Market Analysis assistant is temporarily unavailable" in out["final_answer"]
    assert out["citations"]  # the successful agent's sources survive


def test_config_error_fails_the_turn_loudly(settings: Settings) -> None:
    """REQ-LLM-07: a bad key inside an agent stops the request instead of degrading quietly."""
    qa_model_fail = ScriptedChatModel(fail=True, fail_with=ModelAuthenticationError)
    graph, _, _ = make_graph(
        settings,
        intents=["qa"],
        extra_agents={
            "qa": QAAgent(
                qa_model_fail,
                offline_retriever(),
            )
        },
    )
    with pytest.raises(ModelAuthenticationError):
        _ask(graph, "What is an index fund?")


def test_agent_tokens_stream(settings: Settings) -> None:
    """REQ-WF-06: agent output is available token-by-token via stream_mode='messages'."""
    graph, _, _ = make_graph(settings, intents=["qa"])
    config = {"configurable": {"thread_id": "s"}}
    nodes = set()
    for message, meta in graph.stream(
        {"messages": [HumanMessage("What is an index fund?")]}, config, stream_mode="messages"
    ):
        if isinstance(message, AIMessageChunk) or message.type == "ai":
            nodes.add(meta["langgraph_node"])
    assert "agent_qa" in nodes


def test_checkpoints_restore_our_types_without_warnings(settings: Settings) -> None:
    """REQ-WF-04: our models are allowlisted for checkpoint deserialization."""
    graph, _, _ = make_graph(settings, intents=["qa"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ask(graph, "What is an index fund?", thread="w")
        _ask(graph, "and ETFs?", thread="w")


def test_all_agents_must_be_registered(settings: Settings) -> None:
    """REQ-WF-01: the graph refuses to build with an intent that has no agent."""
    with pytest.raises(ValueError, match="No agent registered"):
        build_graph(router=lambda s: {}, agents={"qa": PlaceholderAgent("qa")})


def test_add_or_reset_reducer() -> None:
    """REQ-WF-02: parallel results concatenate; None resets for a new turn."""
    a, b = AgentResult(agent="qa"), AgentResult(agent="market")
    assert add_or_reset([a], [b]) == [a, b]
    assert add_or_reset(None, [a]) == [a]
    assert add_or_reset([a, b], None) == []


def test_build_default_graph_wires_real_parts(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-LLM-01: the default graph gets its models from the gateway, built at startup."""
    built: list[str] = []

    def fake_get_chat_model(role: str, **kwargs: Any) -> Any:
        built.append(role)
        model = ScriptedChatModel(reply=route_json(["qa"]) if role == "router" else "Answer [1].")
        return (
            model.with_structured_output(kwargs["structured_output"])
            if kwargs.get("structured_output")
            else model
        )

    monkeypatch.setattr(graph_module, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(graph_module, "get_embeddings", lambda _settings: HashingEmbeddings())
    settings.rag = rag_config(tmp_path)
    graph = graph_module.build_default_graph(settings)
    assert (tmp_path / "index" / "index.faiss").exists()  # built and saved at startup
    assert sorted(built) == ["agent", "router", "router"]  # router + ticker extractor
    assert _ask(graph, "What is an index fund?")["final_answer"].startswith("Answer [1].")


def test_follow_up_question_reaches_the_user(settings: Settings) -> None:
    """REQ-WF-10: an agent's follow-up question is shown even with no other answer text."""

    class Asks:
        name = "market"

        def run(self, inp: AgentInput) -> AgentResult:
            return AgentResult(agent="market", follow_up_question="Which ticker?")

    graph, _, _ = make_graph(settings, intents=["market"], extra_agents={"market": Asks()})
    out = _ask(graph, "What's the stock price?")
    assert "**Quick question:** Which ticker?" in out["final_answer"]
