"""Schema migration runner and schema status checks.

Run before the server starts (see docker/entrypoint.sh):

    python -m easyrag.storage.migrate
"""

import logging
from enum import StrEnum
from functools import cache
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from pgvector.sqlalchemy import Vector
from sqlalchemy import Connection
from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.config import Settings
from easyrag.observability.logging import configure_logging
from easyrag.storage.models import APP_MANAGED_INDEX_PREFIX

logger = logging.getLogger("easyrag.storage.migrate")  # stable name under `python -m`

SCRIPT_LOCATION = "easyrag.storage:migrations"

# Let schema reflection recognise pgvector's ``vector`` type, so autogenerate comparisons
# (migrations and the models-match-migrations test) see it instead of NullType.
ischema_names["vector"] = Vector


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Schema-comparison filter: skip indexes managed by the application.

    The dimension-specific HNSW indexes are created by PostgresIndexStore, not by
    migrations (ARCHITECTURE D11), so autogenerate must neither create nor drop them.
    """
    return not (type_ == "index" and bool(name) and str(name).startswith(APP_MANAGED_INDEX_PREFIX))


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
