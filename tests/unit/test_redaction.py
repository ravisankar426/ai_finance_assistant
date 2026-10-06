from __future__ import annotations

import pytest

from src.utils.redaction import redact


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("my ssn is 123-45-6789", "[REDACTED_SSN]"),
        ("email me at jane.doe@example.com please", "[REDACTED_EMAIL]"),
        ("call (555) 123-4567", "[REDACTED_PHONE]"),
        ("call +1 555-123-4567", "[REDACTED_PHONE]"),
        ("card 4111 1111 1111 1111 expires soon", "[REDACTED_CARD]"),
        ("card 4111-1111-1111-1111", "[REDACTED_CARD]"),
        ("my account number 12345678901 at Chase", "[REDACTED_ACCOUNT]"),
        ("routing #021000021", "[REDACTED_ACCOUNT]"),
    ],
)
def test_pii_is_redacted(text: str, token: str) -> None:
    """REQ-GR-05: SSNs, emails, phones, card and account numbers are masked."""
    out = redact(text)
    assert token in out


def test_redaction_removes_the_original_value() -> None:
    """REQ-GR-05: the raw value is gone, the surrounding words stay."""
    out = redact("ssn 123-45-6789, card 4111111111111111")
    assert "123-45-6789" not in out
    assert "4111111111111111" not in out
    assert out.startswith("ssn ")


@pytest.mark.parametrize(
    "text",
    [
        "I have $1,250,000 in my 401(k) and $7,000 in a Roth IRA",
        "Is 2026 a good year? My horizon is 30 years at 7% return",
        "I own 150 shares of AAPL and 20 of MSFT",
        "My budget is 5551234567 dollars",  # bare 10 digits: treated as an amount, not a phone
        "order id 1234567890123456",  # 16 digits but fails Luhn: not a card
        "SSN-like but invalid 000-12-3456",
    ],
)
def test_finance_text_is_not_redacted(text: str) -> None:
    """REQ-GR-05: low false positives — amounts, years, tickers, plan names pass through."""
    assert redact(text) == text
