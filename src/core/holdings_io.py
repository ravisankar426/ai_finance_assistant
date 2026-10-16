"""Parse holdings from CSV text (REQ-PF-01) — used by the Portfolio tab and sample files.

Accepted columns (case-insensitive): ``ticker`` (or ``symbol``), ``quantity`` (or ``shares``),
optional ``cost_basis`` (price paid per share), optional ``purchase_date`` (YYYY-MM-DD; when
``cost_basis`` is blank, that day's closing price becomes the cost basis). Bad rows are
reported, not silently dropped.
"""

from __future__ import annotations

import csv
import io
from datetime import date

from pydantic import ValidationError

from src.core.models import Holding

_ALIASES = {
    "symbol": "ticker",
    "shares": "quantity",
    "cost": "cost_basis",
    "price_paid": "cost_basis",
    "date": "purchase_date",
    "bought": "purchase_date",
    "purchased": "purchase_date",
    "bought_on": "purchase_date",
}


def _row_error(row: dict[str, str]) -> str:
    """Explain what's wrong with a row the model rejected."""
    try:
        if float(row.get("quantity", "")) <= 0:
            return "quantity must be a positive number"
    except ValueError:
        return "quantity must be a positive number"
    if row.get("cost_basis"):
        try:
            if float(row["cost_basis"]) <= 0:
                return "cost basis must be a positive price"
        except ValueError:
            return "cost basis must be a positive price"
    if row.get("purchase_date"):
        try:
            if date.fromisoformat(row["purchase_date"]) > date.today():
                return "purchase date can't be in the future"
        except ValueError:
            return "purchase date must look like 2026-01-15"
    return "the values aren't valid"


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
                    purchase_date=date.fromisoformat(row["purchase_date"])
                    if row.get("purchase_date")
                    else None,
                )
            )
        except (ValueError, ValidationError, KeyError):
            errors.append(f"Row {line} ({ticker or 'blank'}): {_row_error(row)}.")
    return holdings, errors
