"""Parse holdings from CSV text (REQ-PF-01) — used by the Portfolio tab and sample files.

Accepted columns (case-insensitive): ``ticker`` (or ``symbol``), ``quantity`` (or ``shares``),
optional ``cost_basis`` (price paid per share). Bad rows are reported, not silently dropped.
"""

from __future__ import annotations

import csv
import io

from pydantic import ValidationError

from src.core.models import Holding

_ALIASES = {
    "symbol": "ticker",
    "shares": "quantity",
    "cost": "cost_basis",
    "price_paid": "cost_basis",
}


def parse_holdings_csv(text: str) -> tuple[list[Holding], list[str]]:
    """Return (valid holdings, error messages for rejected rows)."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    if not reader.fieldnames:
        return [], ["The file is empty."]
    columns = {
        name: _ALIASES.get(name.strip().lower(), name.strip().lower()) for name in reader.fieldnames
    }
    if not {"ticker", "quantity"} <= set(columns.values()):
        return [], ["The file needs 'ticker' and 'quantity' columns."]
    holdings: list[Holding] = []
    errors: list[str] = []
    for line, raw in enumerate(reader, start=2):
        row = {columns[k]: (v or "").strip() for k, v in raw.items() if k is not None}
        ticker = row.get("ticker", "").upper().lstrip("$")
        try:
            holdings.append(
                Holding(
                    ticker=ticker,
                    quantity=float(row["quantity"]),
                    cost_basis=float(row["cost_basis"]) if row.get("cost_basis") else None,
                )
            )
        except (ValueError, ValidationError, KeyError):
            errors.append(
                f"Row {line} ({ticker or 'blank'}): quantity must be a positive number "
                "and cost basis, if given, a positive price."
            )
    return holdings, errors
