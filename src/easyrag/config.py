"""Application configuration.

All settings are read from environment variables prefixed with ``EASYRAG_`` (and,
for local development, from a ``.env`` file). This is the only module that reads
the environment; everything else receives a ``Settings`` instance.
"""

from typing import Annotated, Literal

from pydantic import BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _upper(value: object) -> object:
    return value.upper() if isinstance(value, str) else value


class Settings(BaseSettings):
    """Validated, immutable application settings."""

    model_config = SettingsConfigDict(
        env_prefix="EASYRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    log_level: Annotated[LogLevel, BeforeValidator(_upper)] = "INFO"
