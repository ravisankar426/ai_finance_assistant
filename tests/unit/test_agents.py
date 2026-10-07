from __future__ import annotations

import pytest
from langchain_core.exceptions import ModelAuthenticationError
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import run_safely
from src.agents.placeholder import PlaceholderAgent
from src.agents.qa import NOT_COVERED, QAAgent
from src.core.models import AgentResult, UserProfile
from src.rag.retriever import RetrievedChunk
from src.workflow.state import AgentInput
from tests.fakes import FakeRetriever, ScriptedChatModel, seen_text


def _chunk(n: int, title: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"a{n}#0",
        article_id=f"a{n}",
        title=title,
        category="investing",
        url=f"https://example.com/{n}",
        text=f"{title} body text.",
        score=0.9,
    )


CHUNKS = [_chunk(1, "Index Funds"), _chunk(2, "ETFs"), _chunk(3, "Bonds")]


def _input(question: str = "What is an index fund?", level: str = "beginner") -> AgentInput:
    return {
        "intent": "qa",
        "question": question,
        "messages": [HumanMessage("earlier q"), AIMessage("earlier a"), HumanMessage(question)],
        "profile": UserProfile(knowledge_level=level),  # type: ignore[arg-type]
        "request_id": "r1",
    }


def test_qa_returns_only_cited_sources() -> None:
    """REQ-QA-01: the answer keeps only the sources it cites inline."""
    model = ScriptedChatModel(reply="Index funds track an index [1], unlike bonds [3].")
    result = QAAgent(model, FakeRetriever(CHUNKS)).run(_input())
    assert result.agent == "qa"
    assert [c.title for c in result.citations] == ["Index Funds", "Bonds"]
    assert result.data["retrieval_scores"] == [0.9, 0.9, 0.9]


def test_qa_without_markers_cites_all_sources() -> None:
    """REQ-QA-01: if the model forgets markers, all retrieved sources are attributed."""
    model = ScriptedChatModel(reply="Index funds track an index.")
    result = QAAgent(model, FakeRetriever(CHUNKS[:2])).run(_input())
    assert len(result.citations) == 2


def test_qa_ignores_out_of_range_markers() -> None:
    """REQ-QA-01: a hallucinated [9] doesn't become a citation."""
    model = ScriptedChatModel(reply="Text [1] and [9].")
    result = QAAgent(model, FakeRetriever(CHUNKS)).run(_input())
    assert [c.title for c in result.citations] == ["Index Funds"]


def test_qa_not_covered_skips_the_llm() -> None:
    """REQ-QA-02: nothing relevant -> honest 'not covered', no citations, no LLM call."""
    model = ScriptedChatModel()
    result = QAAgent(model, FakeRetriever([])).run(_input("How do I bake bread?"))
    assert result.answer == NOT_COVERED
    assert result.citations == []
    assert model.calls == 0


def test_qa_prompt_has_sources_level_history_and_excludes_tax() -> None:
    """REQ-QA-01, REQ-GR-06, REQ-WF-04: sources, knowledge level, and history reach the LLM."""
    model = ScriptedChatModel(reply="x [1]")
    retriever = FakeRetriever(CHUNKS)
    QAAgent(model, retriever, top_k=2).run(_input(level="advanced"))
    prompt = seen_text(model)
    assert "[1] Index Funds" in prompt and "[2] ETFs" in prompt and "Bonds" not in prompt
    assert "experienced" in prompt  # advanced style
    assert "earlier q" in prompt  # history
    assert retriever.calls[0]["exclude_categories"] == ("tax",)


def test_run_safely_turns_crashes_into_degraded_results() -> None:
    """REQ-WF-05: an agent crash becomes an error result instead of failing the turn."""

    class Broken:
        name = "market"

        def run(self, inp: AgentInput) -> AgentResult:
            raise RuntimeError("boom")

    result = run_safely(Broken(), _input())
    assert result.agent == "market"
    assert result.error is not None and "Market Analysis" in result.error


def test_run_safely_reraises_config_errors() -> None:
    """REQ-LLM-07: configuration errors are never swallowed by the agent wrapper."""

    class BadKey:
        name = "qa"

        def run(self, inp: AgentInput) -> AgentResult:
            raise ModelAuthenticationError("bad key")

    with pytest.raises(ModelAuthenticationError):
        run_safely(BadKey(), _input())


def test_placeholder_agent() -> None:
    """REQ-WF-01: every intent has a working route while specialists are being built."""
    result = PlaceholderAgent("news").run(_input())
    assert "News Synthesizer" in result.answer and result.error is None
