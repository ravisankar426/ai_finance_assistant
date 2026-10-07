"""Market overview: major index ETFs + the 11 sector ETFs (REQ-MK-03).

Partial failures are tolerated: tickers that can't be fetched are listed in ``unavailable``
and the rest is still returned (REQ-WF-05 spirit at the data level).
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.core.errors import AppError
from src.core.market_analytics import INDEX_ETFS, SECTOR_ETFS
from src.data.market_service import MarketDataService


class OverviewRow(BaseModel):
    """One line of the overview."""

    ticker: str
    label: str
    price: float
    change_pct: float
    source: str
    stale: bool = False


class MarketOverview(BaseModel):
    """Index and sector snapshot."""

    indices: list[OverviewRow] = Field(default_factory=list)
    sectors: list[OverviewRow] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    as_of: datetime

    @property
    def any_stale(self) -> bool:
        """True if any row came from the stale cache."""
        return any(r.stale for r in (*self.indices, *self.sectors))


def market_overview(service: MarketDataService) -> MarketOverview:
    """Quote every index and sector ETF; sectors sorted best to worst day."""
    overview = MarketOverview(as_of=datetime.now(UTC))
    for group, target in ((INDEX_ETFS, overview.indices), (SECTOR_ETFS, overview.sectors)):
        for ticker, label in group.items():
            try:
                q = service.get_quote(ticker)
            except AppError:
                overview.unavailable.append(ticker)
                continue
            target.append(
                OverviewRow(
                    ticker=ticker,
                    label=label,
                    price=q.price,
                    change_pct=q.change_pct,
                    source=q.source,
                    stale=q.stale,
                )
            )
    overview.sectors.sort(key=lambda r: r.change_pct, reverse=True)
    return overview


def describe_overview(o: MarketOverview) -> str:
    """Fact block for the LLM."""
    lines = ["MARKET OVERVIEW (daily change vs. previous close)"]
    lines += [f"  {r.label} ({r.ticker}): ${r.price:,.2f}, {r.change_pct:+.2f}%" for r in o.indices]
    if o.sectors:
        lines.append("  Sectors, best to worst today:")
        lines += [f"    {r.label} ({r.ticker}): {r.change_pct:+.2f}%" for r in o.sectors]
    if o.unavailable:
        lines.append(f"  Unavailable right now: {', '.join(o.unavailable)}")
    if o.any_stale:
        lines.append("  NOTE: some values are STALE (cached) because live data was unavailable.")
    return "\n".join(lines)
