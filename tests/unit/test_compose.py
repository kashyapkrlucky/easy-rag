"""Invariants of the end-user Docker Compose files (ARCHITECTURE D38).

`docker compose up` must work without any configuration, and the defaults must stay safe.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support import SettingsFactory

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"
HOST_OLLAMA = ROOT / "docker-compose.host-ollama.yml"


def load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


SERVICES: dict[str, Any] = load(COMPOSE)["services"]


def test_no_variable_is_required() -> None:
    # ${VAR:?message} / ${VAR?message} would make `docker compose up` fail without a .env.
    for path in (COMPOSE, HOST_OLLAMA):
        assert not re.search(r"\$\{[^}]*\?", path.read_text(encoding="utf-8")), path.name


def test_ports_are_published_on_localhost_only() -> None:
    for name, service in SERVICES.items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)


def test_every_service_has_bounded_logs() -> None:
    for name, service in SERVICES.items():
        assert service["logging"]["options"]["max-size"], name


@pytest.mark.parametrize("name", ["app", "db", "ollama"])
def test_long_running_services_have_memory_limits(name: str) -> None:
    assert SERVICES[name]["deploy"]["resources"]["limits"]["memory"]


def test_database_password_is_passed_as_a_file() -> None:
    app, db = SERVICES["app"]["environment"], SERVICES["db"]["environment"]

    assert "POSTGRES_PASSWORD" not in db and db["POSTGRES_PASSWORD_FILE"]
    assert app["EASYRAG_DATABASE_PASSWORD_FILE"] == db["POSTGRES_PASSWORD_FILE"]
    # The URL carries no password: "<user>@db", not "<user>:<password>@db".
    assert app["EASYRAG_DATABASE_URL"].startswith(
        "postgresql+psycopg://${POSTGRES_USER:-easyrag}@db:"
    )
    for name in ("app", "db"):
        assert SERVICES[name]["depends_on"]["init"]["condition"] == (
            "service_completed_successfully"
        )


def test_app_container_is_hardened() -> None:
    app = SERVICES["app"]

    assert app["read_only"] is True
    assert app["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in app["security_opt"]


def test_app_does_not_depend_on_the_bundled_ollama() -> None:
    # Required for the host-Ollama override, which disables the ollama service.
    assert "ollama" not in SERVICES["app"]["depends_on"]
    assert "ollama-pull" not in SERVICES  # models are pulled by the app


def test_host_ollama_override_disables_the_bundled_ollama() -> None:
    override = load(HOST_OLLAMA)["services"]

    assert override["ollama"]["profiles"]
    assert "host.docker.internal" in override["app"]["environment"]["EASYRAG_OLLAMA_BASE_URL"]


@pytest.mark.parametrize(
    "files",
    [
        ["docker-compose.yml"],
        ["docker-compose.yml", "docker-compose.host-ollama.yml"],
        ["docker-compose.yml", "docker-compose.dev.yml"],
    ],
)
def test_compose_files_are_valid_without_configuration(files: list[str]) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker is not installed")
    arguments = [docker, "compose", "--project-directory", str(ROOT)]
    for name in files:
        arguments += ["-f", str(ROOT / name)]
    # No .env and no inherited settings: what a first-time user has.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EASYRAG_", "POSTGRES_"))}
    env.pop("COMPOSE_FILE", None)
    result = subprocess.run(
        [*arguments, "--env-file", os.devnull, "config", "-q"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# Read by Docker Compose or the tests, not by Settings.
_NOT_SETTINGS = {
    "EASYRAG_DOCS_HOST_DIR",
    "EASYRAG_PORT",
    "EASYRAG_DB_PORT",
    "EASYRAG_OLLAMA_PORT",
    "EASYRAG_OLLAMA_MEMORY_LIMIT",
    "EASYRAG_APP_MEMORY_LIMIT",
    "EASYRAG_DB_MEMORY_LIMIT",
    "EASYRAG_TEST_DATABASE_URL",
    "EASYRAG_TEST_OLLAMA_URL",
}
# Set by Compose itself; documented in prose.
_SET_BY_COMPOSE = {"EASYRAG_DOCS_DIR", "EASYRAG_MODEL_CACHE_DIR"}


def test_env_example_documents_every_setting_and_nothing_else() -> None:
    from easyrag.config import Settings

    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    mentioned = set(re.findall(r"\bEASYRAG_[A-Z0-9_]+\b", text))
    settings = {f"EASYRAG_{name.upper()}" for name in Settings.model_fields}

    assert not (mentioned - settings - _NOT_SETTINGS), "unknown variables in .env.example"
    assert not (settings - mentioned - _SET_BY_COMPOSE), "settings missing from .env.example"


def test_env_example_sets_nothing() -> None:
    # Copying it must not pin defaults: every assignment is commented out.
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    assert [line for line in lines if line.strip() and not line.startswith("#")] == []


def test_env_example_values_are_valid_when_uncommented(
    tmp_path: Path, make_settings: SettingsFactory
) -> None:
    from easyrag.config import Settings

    # make_settings has cleared EASYRAG_* from the environment: only the file counts.

    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assignments = [line[2:] for line in lines if re.match(r"# [A-Z][A-Z0-9_]*=", line)]
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join([*assignments, "EASYRAG_DATABASE_URL=postgresql+psycopg://u:p@h/d"]) + "\n"
    )

    Settings(_env_file=env_file)
