"""Integration tests run against a real PostgreSQL + pgvector instance.

Start one with:
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d db
and point the tests at it:
    export EASYRAG_TEST_DATABASE_URL=postgresql+psycopg://easyrag:<password>@127.0.0.1:5432/easyrag

Each test gets its own freshly created database (dropped afterwards), so tests never
touch the database named in the URL beyond creating and dropping their own.

Without EASYRAG_TEST_DATABASE_URL every test in this package is skipped.
"""

import os
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from easyrag.storage.migrate import upgrade_to_head

TEST_DATABASE_URL_ENV = "EASYRAG_TEST_DATABASE_URL"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            if not os.environ.get(TEST_DATABASE_URL_ENV):
                item.add_marker(pytest.mark.skip(reason=f"{TEST_DATABASE_URL_ENV} is not set"))


@pytest.fixture
def database_url() -> Iterator[str]:
    """URL of a brand-new, empty database that is dropped after the test."""
    admin_url = make_url(os.environ[TEST_DATABASE_URL_ENV])
    name = f"easyrag_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
        yield admin_url.set(database=name).render_as_string(hide_password=False)
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def migrated_database_url(database_url: str) -> str:
    upgrade_to_head(database_url)
    return database_url


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()
