"""Structured logging configuration for GreenWardrobe.

Provides consistent logging formats across all services and ensures
sensitive values (such as API keys or auth tokens) are masked.
"""

import logging
import re
import sys
from app.core.config import settings

# Patterns for potential secrets to redact from logs
_SECRET_PATTERNS = [
    re.compile(r"(AIza[0-9A-Za-z-_]{35})"),  # Google API key pattern
    re.compile(r"(bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE),
    re.compile(r"(key=)([a-zA-Z0-9_\-]+)", re.IGNORECASE),
]


class SanitizedFormatter(logging.Formatter):
    """Logging formatter that redacts sensitive patterns before output."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        sanitized = original
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub(r"[REDACTED_SECRET]", sanitized)
        return sanitized


def setup_logging() -> logging.Logger:
    """Initialize and configure root application logger."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = SanitizedFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger("greenwardrobe")
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if reloaded
    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logging()
