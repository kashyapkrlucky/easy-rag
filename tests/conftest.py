import os
from typing import Any

import pytest

from easyrag.config import Settings
from tests.support import UNREACHABLE_DATABASE_URL, UNREACHABLE_OLLAMA_URL, SettingsFactory


@pytest.fixture
def make_settings(monkeypatch: pytest.MonkeyPatch) -> SettingsFactory:
    """Build ``Settings`` isolated from the developer's shell environment and ``.env``.

    Dependencies default to unreachable addresses, so nothing touches a real service
    unless a test opts in explicitly.
    """
    for name in list(os.environ):
        if name.startswith("EASYRAG_") and not name.startswith("EASYRAG_TEST_"):
            monkeypatch.delenv(name)

    def factory(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "database_url": UNREACHABLE_DATABASE_URL,
            "ollama_base_url": UNREACHABLE_OLLAMA_URL,
            **overrides,
        }
        return Settings(_env_file=None, **values)

    return factory
