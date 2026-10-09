from typing import Any

import pytest

from easyrag.config import Settings
from tests.support import UNREACHABLE_DATABASE_URL, SettingsFactory

_SETTINGS_ENV_VARS = (
    "EASYRAG_LOG_LEVEL",
    "EASYRAG_DATABASE_URL",
    "EASYRAG_HEALTH_CHECK_TIMEOUT_SECONDS",
)


@pytest.fixture
def make_settings(monkeypatch: pytest.MonkeyPatch) -> SettingsFactory:
    """Build ``Settings`` isolated from the developer's shell environment and ``.env``."""
    for name in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def factory(**overrides: Any) -> Settings:
        values: dict[str, Any] = {"database_url": UNREACHABLE_DATABASE_URL, **overrides}
        return Settings(_env_file=None, **values)

    return factory
