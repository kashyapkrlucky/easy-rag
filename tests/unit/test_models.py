from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

from easyrag.storage.migrate import expected_heads
from easyrag.storage.models import Base


def test_schema_defines_expected_tables() -> None:
    assert set(Base.metadata.tables) == {"documents", "chunks", "index_state"}


def test_chunks_are_deleted_with_their_document() -> None:
    chunks = Base.metadata.tables["chunks"]
    (foreign_key,) = (c for c in chunks.constraints if isinstance(c, ForeignKeyConstraint))

    assert foreign_key.referred_table.name == "documents"
    assert foreign_key.ondelete == "CASCADE"


def test_constraint_names_follow_naming_convention() -> None:
    names = {
        str(constraint.name)
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint | UniqueConstraint | ForeignKeyConstraint)
    }

    assert "uq_documents_source_path" in names
    assert "uq_chunks_document_id_ordinal" in names
    assert "fk_chunks_document_id_documents" in names
    assert "ck_chunks_content_hash_sha256" in names


def test_migration_history_has_a_single_head() -> None:
    # Multiple heads would mean divergent migration branches that need merging.
    assert expected_heads() == {"0003"}
