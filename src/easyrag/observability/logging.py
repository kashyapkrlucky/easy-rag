"""Logging setup.

Plain-text logs to stderr for now; structured JSON logging arrives in Phase 20.
"""

import logging.config

from easyrag.config import LogLevel

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: LogLevel) -> None:
    """Configure the root logger. Safe to call more than once."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {"format": _FORMAT}},
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                }
            },
            "root": {"level": level, "handlers": ["stderr"]},
        }
    )
