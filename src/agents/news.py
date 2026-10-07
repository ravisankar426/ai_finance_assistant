"""News Synthesizer agent: recent headlines, summarized and put in context (REQ-NW-01..04).

Free-tier news sources provide headlines, not article text, so the agent is told it only
has headlines and must not invent details. Every headline it mentions is linked (REQ-NW-02).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import sha1
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.agents.entities import TickerExtractor
from src.agents.grounded import LEVEL_STYLE, cited_chunks
from src.core.errors import AppError
from src.core.models import AgentResult, Citation
from src.core.news import select_headlines
from src.data.market_service import MarketDataService
from src.data.models import NewsItem
from src.rag.retriever import Retriever
from src.workflow.state import AgentInput

MARKET_TOPIC = "stock market"

SYSTEM = """You are the News Synthesizer agent of an EDUCATIONAL personal-finance assistant.
You have ONLY HEADLINES (title, publisher, time) — not the articles.
Rules:
- Summarize the main themes in 3-5 short bullets. Cite headlines by number, like [2].
- Do not invent facts, numbers, or quotes that aren't in the headlines.
- Then add 1-2 sentences on what news like this can mean for a long-term beginner investor,
  in general terms (e.g. why single headlines move prices short-term). Cite concept sources
  by number if you use them.
- NEVER predict prices or recommend buying, selling, or holding anything.
- Keep it under 200 words. Do not add a disclaimer; the system adds one.
{level_style}

HEADLINES:
{headlines}

Concept sources:
{sources}"""


class NewsAgent:
    """Headlines for up to two tickers, or the overall market."""

    name = "news"

    def __init__(
        self,
        model: Runnable[Any, Any],
        extractor: TickerExtractor,
        market: MarketDataService,
        retriever: Retriever,
        *,
        lookback_days: int = 7,
        max_headlines: int = 6,
        now: Any = None,
    ) -> None:
        self.model = model
        self.extractor = extractor
        self.market = market
        self.retriever = retriever
        self.lookback_days = lookback_days
        self.max_headlines = max_headlines
        self._now = now  # injectable clock for tests and recorded data

    def run(self, inp: AgentInput) -> AgentResult:
        """Fetch, filter, deduplicate, and summarize headlines."""
        req = self.extractor.extract(inp["question"])
        subjects = req.tickers[:2] or [MARKET_TOPIC]
        selected: list[NewsItem] = []
        sources: set[str] = set()
        for subject in subjects:
            try:
                feed = self.market.get_news(subject)
            except AppError:
                continue
            sources.add(feed.source)
            now = self._now() if self._now else self._reference_time(feed.source, feed.items)
            selected += select_headlines(
                feed.items,
                ticker=None if subject == MARKET_TOPIC else subject,
                now=now,
                lookback_days=self.lookback_days,
                limit=self.max_headlines,
            )
        selected = list({i.url: i for i in selected}.values())[: self.max_headlines]
        label = ", ".join(subjects) if req.tickers else "the overall market"
        if not selected:  # REQ-NW-04
            return AgentResult(
                agent=self.name,
                answer=f"I couldn't find recent news about {label} from the last "
                f"{self.lookback_days} days.",
            )

        concepts = self.retriever.retrieve(
            f"{inp['question']} market volatility news", k=2, categories=("markets",)
        )
        headlines = "\n".join(
            f"[{i}] {n.title} — {n.publisher}, {n.published_at:%b %d %H:%M} UTC"
            for i, n in enumerate(selected, 1)
        )
        offset = len(selected)
        concept_text = "\n\n".join(
            f"[{offset + i}] {c.title} — {c.section}\n{c.text}" for i, c in enumerate(concepts, 1)
        )
        system = SYSTEM.format(
            level_style=LEVEL_STYLE[inp["profile"].knowledge_level],
            headlines=headlines,
            sources=concept_text or "(none)",
        )
        reply = self.model.invoke([SystemMessage(system), HumanMessage(inp["question"])])
        text = reply.text.strip()
        footer = (
            f"_Headlines via {', '.join(sorted(sources))}; summaries are based on headlines only._"
        )
        return AgentResult(
            agent=self.name,
            answer=f"{text}\n\n{footer}",
            citations=self._citations(text, selected) + cited_chunks(text, concepts, offset=offset),
            data={"headlines": [n.model_dump(mode="json") for n in selected], "subjects": subjects},
        )

    @staticmethod
    def _reference_time(source: str, items: list[NewsItem]) -> datetime:
        """'Now' for the lookback window; recorded data is anchored to its newest headline."""
        if source.startswith("recorded") and items:
            return max(i.published_at for i in items)
        return datetime.now(UTC)

    @staticmethod
    def _citations(text: str, items: list[NewsItem]) -> list[Citation]:
        """One citation per headline the summary references (all, if it referenced none)."""
        numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(items)}
        used = [items[n - 1] for n in sorted(numbers)] or items
        return [
            Citation(
                title=f"{n.title} ({n.publisher})",
                url=n.url,
                category="news",
                chunk_id="news:" + sha1(n.url.encode(), usedforsecurity=False).hexdigest()[:12],
            )
            for n in used
        ]
