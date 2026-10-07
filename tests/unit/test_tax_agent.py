from __future__ import annotations

from dataclasses import replace

from langchain_core.messages import HumanMessage

from src.agents.qa import QAAgent
from src.agents.tax import TAX_NOT_COVERED, TAX_REFERRAL, TaxAgent
from src.core.models import UserProfile
from src.rag.retriever import RetrievedChunk
from src.workflow.state import AgentInput
from tests.fakes import FakeRetriever, ScriptedChatModel, seen_text
from tests.kb import offline_retriever


def _chunk(
    n: int, title: str, *, tax_year: int | None = None, difficulty: str = "beginner"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"art{n}#0",
        article_id=f"art{n}",
        title=title,
        category="tax",
        url=f"https://irs.example/{n}",
        section="Limits",
        text=f"{title} text.",
        score=0.8,
        tax_year=tax_year,
        difficulty=difficulty,
    )


def _input(question: str = "What is the 401(k) limit?", level: str = "beginner") -> AgentInput:
    return {
        "intent": "tax",
        "question": question,
        "messages": [HumanMessage(question)],
        "profile": UserProfile(knowledge_level=level),  # type: ignore[arg-type]
        "request_id": "r",
    }


def test_tax_agent_only_searches_tax_articles() -> None:
    """REQ-TX-01: the Tax agent retrieves from the tax category only."""
    retriever = FakeRetriever([_chunk(1, "401(k) Plans", tax_year=2026)])
    TaxAgent(ScriptedChatModel(reply="x [1]"), retriever).run(_input())
    assert retriever.calls[0]["categories"] == ("tax",)


def test_sources_are_labeled_with_tax_year_and_prompt_requires_it() -> None:
    """REQ-TX-02: each source shows its tax year; the rules demand stating it."""
    model = ScriptedChatModel(reply="For tax year 2026 the limit is $24,500 [1].")
    chunks = [_chunk(1, "401(k) Plans", tax_year=2026), _chunk(2, "Wash Sales")]
    TaxAgent(model, FakeRetriever(chunks)).run(_input())
    prompt = seen_text(model)
    assert "401(k) Plans — Limits (tax year 2026;" in prompt
    assert "Wash Sales — Limits (no specific tax year;" in prompt
    assert "name the tax year" in prompt


def test_referral_always_appended_once() -> None:
    """REQ-TX-03: the professional referral is deterministic, not up to the model."""
    agent = TaxAgent(
        ScriptedChatModel(reply="Limit is $24,500 [1]."), FakeRetriever([_chunk(1, "401(k)")])
    )
    assert agent.run(_input()).answer.endswith(TAX_REFERRAL)
    already = TaxAgent(
        ScriptedChatModel(reply="See a tax professional about your case [1]."),
        FakeRetriever([_chunk(1, "401(k)")]),
    )
    assert TAX_REFERRAL not in already.run(_input()).answer


def test_tax_not_covered_mentions_professional() -> None:
    """REQ-QA-02, REQ-TX-03: nothing relevant -> honest message incl. referral, no LLM call."""
    model = ScriptedChatModel()
    result = TaxAgent(model, FakeRetriever([])).run(
        _input("How are lottery winnings taxed in Ohio?")
    )
    assert result.answer == TAX_NOT_COVERED and model.calls == 0
    assert "tax professional" in result.answer


def test_tax_agent_on_real_kb_finds_wash_sale_article() -> None:
    """REQ-TX-01: end-to-end retrieval over the real tax articles."""
    model = ScriptedChatModel(reply="A wash sale disallows the loss [1].")
    result = TaxAgent(model, offline_retriever()).run(_input("wash sale 30 days rule"))
    assert result.citations[0].title == "The Wash-Sale Rule"
    assert all(c.category == "tax" for c in result.citations)


def test_learn_next_excludes_cited_and_matches_level() -> None:
    """REQ-QA-03: suggestions are uncited related articles suited to the reader's level."""
    ranked = [
        _chunk(1, "Cited"),
        _chunk(2, "Advanced Topic", difficulty="advanced"),
        _chunk(3, "Next One", difficulty="intermediate"),
        replace(
            _chunk(3, "Next One", difficulty="intermediate"), chunk_id="art3#1"
        ),  # same article
        _chunk(4, "Another"),
        _chunk(5, "Third"),
        _chunk(6, "Fourth"),
    ]
    model = ScriptedChatModel(reply="Answer [1].")
    result = QAAgent(model, FakeRetriever(ranked), top_k=1, learn_next=3).run(_input())
    titles = [a["title"] for a in result.data["learn_next"]]
    assert titles == ["Next One", "Another", "Third"]  # no 'Cited', no 'Advanced' for a beginner
    advanced = QAAgent(ScriptedChatModel(reply="A [1]."), FakeRetriever(ranked), top_k=1).run(
        _input(level="advanced")
    )
    assert advanced.data["learn_next"][0]["title"] == "Advanced Topic"


def test_learn_next_on_real_kb() -> None:
    """REQ-QA-03: real retrieval produces related suggestions with links."""
    result = QAAgent(ScriptedChatModel(reply="Index funds [1]."), offline_retriever()).run(
        _input("What is an index fund and how do fees matter?")
    )
    suggestions = result.data["learn_next"]
    assert 1 <= len(suggestions) <= 3
    assert all(s["url"].startswith("https://") for s in suggestions)
    assert result.citations[0].title not in {s["title"] for s in suggestions}
