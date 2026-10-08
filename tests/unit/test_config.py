import pytest
from pydantic import ValidationError

from easyrag.config import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EASYRAG_LOG_LEVEL", raising=False)

    assert Settings(_env_file=None).log_level == "INFO"


def test_reads_prefixed_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EASYRAG_LOG_LEVEL", "debug")

    assert Settings(_env_file=None).log_level == "DEBUG"


def test_rejects_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EASYRAG_LOG_LEVEL", "verbose")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_are_immutable() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.log_level = "ERROR"  # type: ignore[misc]
