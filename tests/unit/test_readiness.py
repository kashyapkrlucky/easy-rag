import asyncio
import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.api.routes import health
from easyrag.main import create_app
from easyrag.storage.migrate import SchemaStatus
from tests.support import SettingsFactory


async def healthy_ping(engine: AsyncEngine, timeout_seconds: float) -> None:
    return None


def schema_reporting(status: SchemaStatus) -> object:
    async def check(engine: AsyncEngine) -> SchemaStatus:
        return status

    return check


def test_ready_when_database_responds_and_schema_is_current(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "ping", healthy_ping)
    monkeypatch.setattr(health, "schema_status", schema_reporting(SchemaStatus.CURRENT))

    with TestClient(create_app(make_settings())) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"database": "ok", "schema": "ok"}}


def test_not_ready_when_schema_is_outdated(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "ping", healthy_ping)
    monkeypatch.setattr(health, "schema_status", schema_reporting(SchemaStatus.OUTDATED))

    with TestClient(create_app(make_settings())) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "ok", "schema": "outdated"},
    }


def test_not_ready_when_database_is_unreachable(
    make_settings: SettingsFactory, caplog: pytest.LogCaptureFixture
) -> None:
    with TestClient(create_app(make_settings())) as client, caplog.at_level(logging.WARNING):
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "unavailable", "schema": "unavailable"},
    }
    assert "secret" not in caplog.text  # credentials never reach the logs


def test_not_ready_when_database_check_times_out(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hanging_ping(engine: AsyncEngine, timeout_seconds: float) -> None:
        async with asyncio.timeout(timeout_seconds):
            await asyncio.sleep(60)

    monkeypatch.setattr(health, "ping", hanging_ping)

    with TestClient(create_app(make_settings(health_check_timeout_seconds=0.05))) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["database"] == "unavailable"


def test_not_ready_when_schema_check_times_out(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hanging_schema_status(engine: AsyncEngine) -> SchemaStatus:
        await asyncio.sleep(60)
        return SchemaStatus.CURRENT

    monkeypatch.setattr(health, "ping", healthy_ping)
    monkeypatch.setattr(health, "schema_status", hanging_schema_status)

    with TestClient(create_app(make_settings(health_check_timeout_seconds=0.05))) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["schema"] == "unavailable"
