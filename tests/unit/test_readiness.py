import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyrag.config import Settings
from easyrag.container import AppContainer, build_container
from easyrag.health import CheckStatus, ReadinessCheck
from easyrag.main import create_app
from tests.support import SettingsFactory


def check_returning(status: CheckStatus) -> ReadinessCheck:
    async def check() -> CheckStatus:
        return status

    return check


def app_with_checks(settings: Settings, checks: Mapping[str, ReadinessCheck]) -> FastAPI:
    def factory(settings: Settings) -> AppContainer:
        return replace(build_container(settings), readiness_checks=checks)

    return create_app(settings, container_factory=factory)


def get_ready(app: FastAPI) -> tuple[int, dict[str, object]]:
    with TestClient(app) as client:
        response = client.get("/health/ready")
    return response.status_code, response.json()


def test_ready_when_every_check_passes(make_settings: SettingsFactory) -> None:
    checks = {"database": check_returning("ok"), "embedding_model": check_returning("ok")}

    assert get_ready(app_with_checks(make_settings(), checks)) == (
        200,
        {"status": "ready", "checks": {"database": "ok", "embedding_model": "ok"}},
    )


@pytest.mark.parametrize("failing", ["outdated", "missing", "misconfigured", "unavailable"])
def test_not_ready_when_any_check_fails(
    make_settings: SettingsFactory, failing: CheckStatus
) -> None:
    checks = {"database": check_returning("ok"), "other": check_returning(failing)}

    status, body = get_ready(app_with_checks(make_settings(), checks))

    assert status == 503
    assert body == {"status": "not_ready", "checks": {"database": "ok", "other": failing}}


def test_check_raising_is_reported_unavailable_without_leaking_details(
    make_settings: SettingsFactory, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken() -> CheckStatus:
        raise ConnectionError("postgresql://easyrag:secret@db failed")

    app = app_with_checks(make_settings(), {"database": broken})
    # create_app() reconfigures logging, replacing the root handlers that caplog installed.
    root = logging.getLogger()
    root.addHandler(caplog.handler)
    try:
        status, body = get_ready(app)
    finally:
        root.removeHandler(caplog.handler)

    assert (status, body["checks"]) == (503, {"database": "unavailable"})
    assert "ConnectionError" in caplog.text
    assert "secret" not in caplog.text


def test_hanging_check_times_out(make_settings: SettingsFactory) -> None:
    async def hanging() -> CheckStatus:
        await asyncio.sleep(60)
        return "ok"

    settings = make_settings(health_check_timeout_seconds=0.05)
    status, body = get_ready(app_with_checks(settings, {"database": hanging}))

    assert (status, body["checks"]) == (503, {"database": "unavailable"})


def test_checks_run_concurrently(make_settings: SettingsFactory) -> None:
    async def slow() -> CheckStatus:
        await asyncio.sleep(0.3)
        return "ok"

    app = app_with_checks(make_settings(), {"a": slow, "b": slow, "c": slow})
    with TestClient(app) as client:
        started = time.perf_counter()
        response = client.get("/health/ready")
        elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 0.6  # sequential execution would take at least 0.9 s


def test_default_container_registers_all_dependencies(make_settings: SettingsFactory) -> None:
    assert list(build_container(make_settings()).readiness_checks) == [
        "database",
        "schema",
        "embedding_model",
        "llm_model",
        "index",
    ]


def test_unreachable_dependencies_are_reported(make_settings: SettingsFactory) -> None:
    # Real checks against addresses where nothing listens.
    status, body = get_ready(create_app(make_settings()))

    assert status == 503
    checks = body["checks"]
    assert isinstance(checks, dict)
    assert set(checks) == set(build_container(make_settings()).readiness_checks)
    assert set(checks.values()) == {"unavailable"}
