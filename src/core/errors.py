"""Application error hierarchy.

Every error carries a ``user_message`` that is safe to show to end users (constitution P4).
The technical detail goes in the normal exception message, which is logged but never shown.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected application errors."""

    user_message = "Something went wrong. Please try again."

    def __init__(self, detail: str = "", *, user_message: str | None = None) -> None:
        super().__init__(detail or self.user_message)
        if user_message is not None:
            self.user_message = user_message


class ConfigurationError(AppError):
    """Missing or invalid configuration (e.g. an API key is not set)."""

    user_message = "The assistant is not configured correctly. Please contact the administrator."


class LLMUnavailableError(AppError):
    """All configured language models failed."""

    user_message = "The AI service is temporarily unavailable. Please try again in a minute."


class MarketDataUnavailableError(AppError):
    """No market-data provider could serve the request and no cached data exists."""

    user_message = "Live market data is unavailable right now. Please try again shortly."


class InvalidTickerError(AppError):
    """A ticker symbol is malformed or unknown."""

    user_message = "That ticker symbol doesn't look valid."


class KnowledgeBaseError(AppError):
    """The knowledge base could not be loaded or searched."""

    user_message = "The knowledge base is unavailable right now."
