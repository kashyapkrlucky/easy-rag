import pytest
from pydantic import ValidationError

from easyrag.config import Settings
from tests.support import UNREACHABLE_DATABASE_URL, SettingsFactory


def test_defaults(make_settings: SettingsFactory) -> None:
    settings = make_settings()

    assert settings.log_level == "INFO"
    assert settings.health_check_timeout_seconds == 2.0


def test_reads_prefixed_environment_variables(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYRAG_LOG_LEVEL", "debug")
    monkeypatch.setenv("EASYRAG_DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.database_url.get_secret_value() == "postgresql+psycopg://u:p@db:5432/x"


def test_database_url_is_required(make_settings: SettingsFactory) -> None:
    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)


def test_database_url_is_not_revealed(make_settings: SettingsFactory) -> None:
    settings = make_settings()

    assert "secret" not in repr(settings)
    assert "secret" not in str(settings.database_url)
    assert settings.database_url.get_secret_value() == UNREACHABLE_DATABASE_URL


@pytest.mark.parametrize(
    "overrides",
    [{"log_level": "verbose"}, {"health_check_timeout_seconds": 0}],
)
def test_rejects_invalid_values(
    make_settings: SettingsFactory, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_settings_are_immutable(make_settings: SettingsFactory) -> None:
    settings = make_settings()

    with pytest.raises(ValidationError):
        settings.log_level = "ERROR"  # type: ignore[misc]
