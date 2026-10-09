"""Database engine construction and connectivity checks."""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from easyrag.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    """Create the application's async engine. No connection is opened until first use."""
    return create_async_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
        connect_args={"connect_timeout": max(1, round(settings.health_check_timeout_seconds))},
    )


async def ping(engine: AsyncEngine, timeout_seconds: float) -> None:
    """Round-trip ``SELECT 1``. Raises on failure or when ``timeout_seconds`` elapses."""
    async with asyncio.timeout(timeout_seconds), engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
