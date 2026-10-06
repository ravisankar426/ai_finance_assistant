"""PII redaction applied before text reaches an LLM or a log line (REQ-GR-05).

Design goal: low false positives in a finance context. Dollar amounts, years, ticker
symbols, and plan names like "401(k)" must pass through untouched, so:
- card numbers must also pass the Luhn checksum;
- phone numbers must be formatted (bare 10-digit numbers are left alone, since they
  are more likely to be amounts than phones).
"""

from __future__ import annotations

import re
from collections.abc import Callable

_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)")
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_ACCOUNT = re.compile(
    r"\b(account|acct|routing)(\s*(?:number|no\.?|#))?\s*[:#]?\s*\d{6,17}\b",
    re.IGNORECASE,
)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_sub(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return "[REDACTED_CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group()


def _account_sub(match: re.Match[str]) -> str:
    label = match.group(1) + (match.group(2) or "")
    return f"{label} [REDACTED_ACCOUNT]"


# Order matters: specific patterns first, so an SSN is not half-eaten by the phone rule.
_RULES: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    (_EMAIL, "[REDACTED_EMAIL]"),
    (_SSN, "[REDACTED_SSN]"),
    (_ACCOUNT, _account_sub),
    (_CARD_CANDIDATE, _card_sub),
    (_PHONE, "[REDACTED_PHONE]"),
]


def redact(text: str) -> str:
    """Return ``text`` with SSNs, card/account numbers, emails, and phone numbers masked."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
