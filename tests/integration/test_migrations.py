import hashlib
import os
import subprocess
import sys

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import Connection, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from easyrag.main import create_app
from easyrag.storage.migrate import alembic_config, expected_heads, upgrade_to_head
from easyrag.storage.models import Base
from tests.support import SettingsFactory

APP_TABLES = {"documents", "chunks", "index_state"}
HASH = hashlib.sha256(b"content").hexdigest()


def _tables(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names())


def _vector_installed(connection: Connection) -> bool:
    query = text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    return connection.execute(query).first() is not None


def test_upgrade_creates_schema_and_extension(migrated_database_url: str) -> None:
    engine = create_engine(migrated_database_url)
    with engine.connect() as connection:
        assert _tables(connection) >= APP_TABLES
        assert _vector_installed(connection)
        heads = set(MigrationContext.configure(connection).get_current_heads())
    engine.dispose()

    assert heads == expected_heads() == {"0001"}


def test_downgrade_and_upgrade_round_trip(migrated_database_url: str) -> None:
    config = alembic_config(migrated_database_url)
    engine = create_engine(migrated_database_url)

    command.downgrade(config, "base")
    with engine.connect() as connection:
        assert not APP_TABLES & _tables(connection)
        assert not _vector_installed(connection)

    command.upgrade(config, "head")
    with engine.connect() as connection:
        assert _tables(connection) >= APP_TABLES
    engine.dispose()


def test_upgrade_is_idempotent(migrated_database_url: str) -> None:
    upgrade_to_head(migrated_database_url)  # second run is a no-op


def test_concurrent_migration_processes_are_serialised(database_url: str) -> None:
    # Simulates several containers starting at once; each runs the entrypoint's command.
    # Without the advisory lock in env.py, the losers fail with duplicate-object errors.
    env = {**os.environ, "EASYRAG_DATABASE_URL": database_url, "EASYRAG_LOG_LEVEL": "WARNING"}
    command_line = [sys.executable, "-m", "easyrag.storage.migrate"]
    processes = [
        subprocess.Popen(command_line, env=env, stderr=subprocess.PIPE, text=True) for _ in range(4)
    ]

    for process in processes:
        _, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr


def test_models_match_migrations(migrated_database_url: str) -> None:
    engine = create_engine(migrated_database_url)
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        diff = compare_metadata(context, Base.metadata)
    engine.dispose()

    assert diff == [], f"ORM models and migrations differ: {diff}"


def test_constraints_and_cascade(migrated_database_url: str) -> None:
    engine = create_engine(migrated_database_url)
    insert_document = text(
        "INSERT INTO documents (source_path, title, content_hash, pipeline_fingerprint, size_bytes)"
        " VALUES (:path, 'Title', :hash, :hash, 10) RETURNING id"
    )
    insert_chunk = text(
        "INSERT INTO chunks (id, document_id, ordinal, heading_path, section_anchor, content,"
        " content_hash, start_line, end_line, token_estimate)"
        " VALUES (:id, :doc, 0, ARRAY['Intro'], 'intro', 'text', :hash, 1, 3, 2)"
    )

    with engine.begin() as connection:
        document_id = connection.execute(insert_document, {"path": "a.md", "hash": HASH}).scalar()
        connection.execute(insert_chunk, {"id": "c" * 32, "doc": document_id, "hash": HASH})
        connection.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
        assert connection.execute(text("SELECT count(*) FROM chunks")).scalar() == 0

    with (
        pytest.raises(IntegrityError, match="ck_documents_content_hash_sha256"),
        engine.begin() as connection,
    ):
        connection.execute(insert_document, {"path": "b.md", "hash": "not-a-hash"})

    with (
        pytest.raises(IntegrityError, match="uq_documents_source_path"),
        engine.begin() as connection,
    ):
        connection.execute(insert_document, {"path": "c.md", "hash": HASH})
        connection.execute(insert_document, {"path": "c.md", "hash": HASH})
    engine.dispose()


def test_readiness_requires_current_schema(
    make_settings: SettingsFactory, database_url: str
) -> None:
    with TestClient(create_app(make_settings(database_url=database_url))) as client:
        before = client.get("/health/ready")
        upgrade_to_head(database_url)
        after = client.get("/health/ready")

    # Ollama is not part of integration tests, so only the database checks are asserted.
    assert before.status_code == 503
    assert before.json()["checks"] | {"embedding_model": None} == {
        "database": "ok",
        "schema": "outdated",
        "embedding_model": None,
    }
    assert after.json()["checks"]["database"] == "ok"
    assert after.json()["checks"]["schema"] == "ok"
