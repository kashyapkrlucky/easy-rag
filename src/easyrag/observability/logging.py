"""Logging setup: one JSON object per line (default) or plain text for development.

Context fields (``request_id``, ``index_run_id``) are added from contextvars, and extra
fields passed with ``logger.info(..., extra={...})`` become JSON keys. Uvicorn's loggers
are routed through the same handler; its access log is replaced by the request middleware.
"""

import json
import logging
import logging.config
from datetime import UTC, datetime
from typing import Any, Literal

from easyrag.config import LogLevel
from easyrag.observability.context import index_run_id, request_id

LogFormat = Literal["json", "text"]

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s%(context)s"
# Attributes every LogRecord has; anything else was passed via ``extra``.
_STANDARD = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "context",
    "color_message",  # added by uvicorn
}


class ContextFilter(logging.Filter):
    """Attach correlation IDs from contextvars to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        record.index_run_id = index_run_id.get()
        context = [
            f"{name}={value}"
            for name in ("request_id", "index_run_id")
            if (value := getattr(record, name))
        ]
        record.context = f" [{' '.join(context)}]" if context else ""
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and value is not None:
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        # default=str: an unserialisable extra must never break logging.
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: LogLevel, log_format: LogFormat = "json") -> None:
    """Configure the root and uvicorn loggers. Safe to call more than once."""
    formatter: dict[str, Any] = (
        {"()": JsonFormatter} if log_format == "json" else {"format": _TEXT_FORMAT}
    )
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"context": {"()": ContextFilter}},
            "formatters": {"default": formatter},
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["context"],
                    "stream": "ext://sys.stderr",
                }
            },
            "root": {"level": level, "handlers": ["stderr"]},
            "loggers": {
                # Same output as everything else; the access log is replaced by
                # RequestContextMiddleware (one line per request, with request_id).
                "uvicorn": {"handlers": ["stderr"], "propagate": False},
                "uvicorn.error": {"handlers": ["stderr"], "propagate": False},
                "uvicorn.access": {"handlers": [], "propagate": False},
            },
        }
    )
