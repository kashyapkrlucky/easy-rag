import asyncio
import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.api.routes import health
from easyrag.main import create_app
from tests.support import SettingsFactory


def test_ready_when_database_responds(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def healthy_ping(engine: AsyncEngine, timeout_seconds: float) -> None:
        return None

    monkeypatch.setattr(health, "ping", healthy_ping)

    with TestClient(create_app(make_settings())) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"database": "ok"}}


def test_not_ready_when_database_is_unreachable(
    make_settings: SettingsFactory, caplog: pytest.LogCaptureFixture
) -> None:
    with TestClient(create_app(make_settings())) as client, caplog.at_level(logging.WARNING):
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "checks": {"database": "unavailable"}}
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
