"""Integration tests run against a real PostgreSQL + pgvector instance.

Start one with:
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d db
and point the tests at it:
    export EASYRAG_TEST_DATABASE_URL=postgresql+psycopg://easyrag:<password>@127.0.0.1:5432/easyrag

Without EASYRAG_TEST_DATABASE_URL every test in this package is skipped.
"""

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

TEST_DATABASE_URL_ENV = "EASYRAG_TEST_DATABASE_URL"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            if not os.environ.get(TEST_DATABASE_URL_ENV):
                item.add_marker(pytest.mark.skip(reason=f"{TEST_DATABASE_URL_ENV} is not set"))


@pytest.fixture
def database_url() -> str:
    return os.environ[TEST_DATABASE_URL_ENV]


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()
