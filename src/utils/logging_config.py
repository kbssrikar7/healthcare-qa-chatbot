"""
Centralised loguru configuration for the Healthcare QA pipeline.

Call configure_logging() once at application startup (api/main.py).
- In development: human-readable coloured output to stderr.
- In production (LOG_FORMAT=json): structured JSON lines to stdout,
  suitable for log aggregators (Loki, CloudWatch, Datadog).
- X-Request-ID from the HTTP layer is propagated via contextvars.
"""

import os
import sys
from contextvars import ContextVar

from loguru import logger

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _add_request_id(record):
    record["extra"]["request_id"] = request_id_var.get("-")


def configure_logging(level: str = None, fmt: str = None) -> None:
    """
    Configure loguru sinks.

    Parameters
    ----------
    level : Log level string (DEBUG/INFO/WARNING/ERROR). Defaults to LOG_LEVEL env var or INFO.
    fmt   : 'json' for structured output, 'text' for human-readable. Defaults to LOG_FORMAT env var.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = fmt or os.getenv("LOG_FORMAT", "text").lower()

    logger.remove()
    logger.configure(patcher=_add_request_id)

    if fmt == "json":
        logger.add(
            sys.stdout,
            level=level,
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "[{extra[request_id]}] "
                "<level>{message}</level>"
            ),
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
