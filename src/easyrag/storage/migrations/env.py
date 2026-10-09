"""Alembic environment.

Used both by the Alembic CLI (developers, via ``alembic.ini``) and programmatically by
``easyrag.storage.migrate``. Migrations run synchronously with psycopg 3.
"""

from alembic import context
from sqlalchemy import create_engine, pool, text

from easyrag.config import Settings
from easyrag.storage.models import Base

# Arbitrary constant: serialises concurrent migration runs (e.g. two containers starting).
MIGRATION_LOCK_KEY = 0x6561737972616701

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    # Programmatic callers pass the URL as an attribute to avoid configparser
    # interpolation of '%' in passwords; the CLI falls back to the application settings.
    url = config.attributes.get("database_url")
    if isinstance(url, str):
        return url
    return Settings().database_url.get_secret_value()


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": MIGRATION_LOCK_KEY})
        connection.commit()
        try:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                transaction_per_migration=True,
            )
            with context.begin_transaction():
                context.run_migrations()
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": MIGRATION_LOCK_KEY})
            connection.commit()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
