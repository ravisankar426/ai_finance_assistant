from __future__ import annotations

import json

import pytest

from src.core.errors import AppError, LLMUnavailableError, MarketDataUnavailableError
from src.utils.logging import configure_logging, get_logger, new_request_id, request_context


def _last_json_line(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    return json.loads(lines[-1])  # type: ignore[no-any-return]


def test_logs_are_json_with_request_id(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-NFR-05: structured JSON logs carry request_id, level, timestamp, and fields."""
    configure_logging(level="INFO", fmt="json")
    log = get_logger("test")
    with request_context("req-123"):
        log.info("node_done", node="router", latency_ms=12)
    record = _last_json_line(capsys)
    assert record["request_id"] == "req-123"
    assert record["event"] == "node_done"
    assert record["node"] == "router"
    assert record["latency_ms"] == 12
    assert record["level"] == "info"
    assert record["module"] == "test"
    assert "timestamp" in record


def test_request_id_is_unbound_after_context(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-NFR-05: request ids don't leak into later, unrelated log lines."""
    configure_logging(fmt="json")
    log = get_logger()
    with request_context() as rid:
        assert len(rid) == 12
    log.info("after")
    assert "request_id" not in _last_json_line(capsys)


def test_logs_are_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-GR-05: PII never reaches a log line."""
    configure_logging(fmt="json")
    get_logger().info("user_message", text="my ssn is 123-45-6789")
    record = _last_json_line(capsys)
    assert "123-45-6789" not in json.dumps(record)
    assert "[REDACTED_SSN]" in str(record["text"])


def test_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-NFR-05: log level from config filters lower-severity lines."""
    configure_logging(level="WARNING", fmt="json")
    get_logger().info("hidden")
    assert capsys.readouterr().out == ""


def test_console_format_renders(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-NFR-05: human-readable console format for local development."""
    configure_logging(fmt="console")
    get_logger().info("hello_console")
    assert "hello_console" in capsys.readouterr().out


def test_request_ids_are_unique() -> None:
    """REQ-NFR-05: each request gets its own id."""
    assert len({new_request_id() for _ in range(100)}) == 100


def test_errors_carry_safe_user_messages() -> None:
    """REQ-API-03: errors expose a user-safe message separate from technical detail."""
    err = LLMUnavailableError("openai 503 after 3 retries")
    assert "503" in str(err)
    assert "503" not in err.user_message
    assert isinstance(err, AppError)
    custom = MarketDataUnavailableError("x", user_message="Quotes are delayed.")
    assert custom.user_message == "Quotes are delayed."
    assert str(AppError()) == AppError.user_message
