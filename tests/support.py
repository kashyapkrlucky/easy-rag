"""Shared test helpers (importable; fixtures live in conftest.py)."""

from collections.abc import Callable

from easyrag.config import Settings

# Nothing listens on port 1: connections are refused immediately.
UNREACHABLE_DATABASE_URL = "postgresql+psycopg://easyrag:secret@127.0.0.1:1/easyrag"
UNREACHABLE_OLLAMA_URL = "http://127.0.0.1:1"

SettingsFactory = Callable[..., Settings]
