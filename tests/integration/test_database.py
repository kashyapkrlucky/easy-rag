from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.main import create_app
from easyrag.storage.database import ping
from tests.support import SettingsFactory


async def test_ping_succeeds(engine: AsyncEngine) -> None:
    await ping(engine, timeout_seconds=5)


async def test_server_is_postgres_18_or_newer(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        version_num = (await connection.execute(text("SHOW server_version_num"))).scalar_one()

    assert int(version_num) >= 180000


async def test_pgvector_extension_is_available(engine: AsyncEngine) -> None:
    # Availability only; CREATE EXTENSION is performed by migrations (Phase 4).
    async with engine.connect() as connection:
        row = (
            await connection.execute(
                text("SELECT default_version FROM pg_available_extensions WHERE name = 'vector'")
            )
        ).one_or_none()

    assert row is not None, "pgvector is not installed in this PostgreSQL image"


def test_readiness_reports_ready(make_settings: SettingsFactory, database_url: str) -> None:
    with TestClient(create_app(make_settings(database_url=database_url))) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"database": "ok"}}
