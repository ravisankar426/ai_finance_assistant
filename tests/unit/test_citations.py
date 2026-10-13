"""Inline [n] markers always match the numbered source list the user sees (REQ-RAG-05)."""

from __future__ import annotations

from src.agents.grounded import remap_markers, renumber_citations
from src.core.models import AgentResult, Citation
from src.workflow.graph import make_synthesizer


def cit(article: str, chunk: int = 1) -> Citation:
    return Citation(
        title=article.title(),
        url=f"https://example.org/{article}",
        category="basics",
        chunk_id=f"{article}#{chunk}",
    )


EMERGENCY = [cit("emergency", i) for i in range(1, 5)]  # four chunks of one article


def test_chunks_of_one_article_collapse_to_one_source() -> None:
    """The judge's failing case: [3][2][4][1] over one article -> all [1], one source."""
    answer = "Avoid debt [3]. Save 3-6 months [2]. Start small [2]. Automate [4]. Rebuild [1]."
    text, cited = renumber_citations(answer, EMERGENCY)
    assert cited == [EMERGENCY[2]]
    assert (
        text == "Avoid debt [1]. Save 3-6 months [1]. Start small [1]. Automate [1]. Rebuild [1]."
    )


def test_numbering_follows_reading_order() -> None:
    """First source mentioned becomes [1]."""
    sources = [cit("a"), cit("b"), cit("c")]
    text, cited = renumber_citations("B first [2]. Then C [3]. Then B again [2].", sources)
    assert [c.url for c in cited] == [sources[1].url, sources[2].url]
    assert text == "B first [1]. Then C [2]. Then B again [1]."


def test_invalid_markers_are_dropped_and_repeats_collapsed() -> None:
    """Hallucinated [9] disappears cleanly; '[1] [1]' after collapsing becomes '[1]'."""
    sources = [cit("a", 1), cit("a", 2)]
    text, cited = renumber_citations("Fact [1] [2]. Made up [9].", sources)
    assert text == "Fact [1]. Made up." and len(cited) == 1


def test_no_markers_uses_fallback_unchanged() -> None:
    """No valid markers -> text untouched, fallback sources cited once per URL."""
    text, cited = renumber_citations("Plain answer.", EMERGENCY, fallback=EMERGENCY)
    assert text == "Plain answer." and cited == [EMERGENCY[0]]
    assert renumber_citations("Plain.", EMERGENCY) == ("Plain.", [])


def test_remap_markers_leaves_other_brackets() -> None:
    """Only numeric markers change; markdown links are untouched."""
    assert remap_markers("[see](u) and [2]", {2: 5}) == "[see](u) and [5]"


def test_synthesizer_builds_one_global_source_list() -> None:
    """REQ-WF-02: per-agent numbers -> one global list in display order (Market before Tax)."""
    shared, tax_only, market_only = cit("roth"), cit("ira-limits"), cit("trend")
    results = [
        AgentResult(
            agent="tax", answer="Roth basics [1]. Limits [2].", citations=[shared, tax_only]
        ),
        AgentResult(
            agent="market", answer="Trend [1]. Also Roth [2].", citations=[market_only, shared]
        ),
    ]
    out = make_synthesizer(None)({"agent_results": results, "question": "q"})  # type: ignore[arg-type]
    assert [c.url for c in out["citations"]] == [market_only.url, shared.url, tax_only.url]
    assert "Trend [1]. Also Roth [2]." in out["final_answer"]
    assert "Roth basics [2]. Limits [3]." in out["final_answer"]  # shared source keeps [2]
