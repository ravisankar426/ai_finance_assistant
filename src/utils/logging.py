"""Structured logging with request correlation and PII redaction (REQ-NFR-05, REQ-GR-05).

Usage::

    configure_logging(level="INFO", fmt="json")      # once, at process start
    log = get_logger(__name__)
    with request_context() as request_id:            # once per user request
        log.info("router_decision", intents=["market"], latency_ms=412)

Every line emitted inside ``request_context`` carries ``request_id`` automatically,
because it is stored in a contextvar (safe across asyncio tasks).
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, Literal

import structlog

from src.utils.redaction import redact


def _redact_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Redact PII from every string value before it is rendered."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = redact(value)
    return event_dict


def configure_logging(level: str = "INFO", fmt: Literal["console", "json"] = "console") -> None:
    """Configure structlog for the whole process. Safe to call more than once."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_processor,  # last before rendering, so it sees everything
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a lazy structlog logger, optionally tagged with the module name.

    Initial values are passed to ``get_logger`` instead of calling ``.bind()``: binding a
    module-level logger would freeze it to whatever configuration exists at import time.
    """
    # Key is "module", not "logger": `logger` collides with a structlog.wrap_logger parameter.
    return structlog.get_logger(module=name) if name else structlog.get_logger()


def new_request_id() -> str:
    """Return a short random request id."""
    return uuid.uuid4().hex[:12]


@contextmanager
def request_context(request_id: str | None = None, **extra: Any) -> Iterator[str]:
    """Bind ``request_id`` (and any extra fields) to all logs emitted inside the block."""
    rid = request_id or new_request_id()
    tokens = structlog.contextvars.bind_contextvars(request_id=rid, **extra)
    try:
        yield rid
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
