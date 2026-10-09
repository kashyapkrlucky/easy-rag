"""Schema migration runner and schema status checks.

Run before the server starts (see docker/entrypoint.sh):

    python -m easyrag.storage.migrate
"""

import logging
from enum import StrEnum
from functools import cache

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.config import Settings
from easyrag.observability.logging import configure_logging

logger = logging.getLogger("easyrag.storage.migrate")  # stable name under `python -m`

SCRIPT_LOCATION = "easyrag.storage:migrations"


class SchemaStatus(StrEnum):
    CURRENT = "current"
    OUTDATED = "outdated"


def alembic_config(database_url: str) -> Config:
    """Alembic configuration that works from an installed package (no alembic.ini needed)."""
    config = Config()
    config.set_main_option("script_location", SCRIPT_LOCATION)
    config.attributes["database_url"] = database_url
    return config


@cache
def expected_heads() -> frozenset[str]:
    """Revision(s) the code expects the database to be at."""
    script = ScriptDirectory.from_config(alembic_config(database_url=""))
    return frozenset(script.get_heads())


def upgrade_to_head(database_url: str) -> None:
    command.upgrade(alembic_config(database_url), "head")


def _current_heads(connection: Connection) -> frozenset[str]:
    return frozenset(MigrationContext.configure(connection).get_current_heads())


async def schema_status(engine: AsyncEngine) -> SchemaStatus:
    """Compare the database's revision with the code's. Raises if the DB is unreachable."""
    async with engine.connect() as connection:
        current = await connection.run_sync(_current_heads)
    return SchemaStatus.CURRENT if current == expected_heads() else SchemaStatus.OUTDATED


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    logger.info("Applying database migrations (target: %s)", ", ".join(sorted(expected_heads())))
    upgrade_to_head(settings.database_url.get_secret_value())
    logger.info("Database schema is up to date")


if __name__ == "__main__":
    main()
