"""Keyword search: chunks.search_vector (tsvector) with a GIN index, backfilled.

The vector holds two forms of the same text: 'simple' lexemes (no stemming, so identifiers
such as EASYRAG_EMBEDDING_MODEL or v1.5 survive intact; title/headings weight A, content
weight B) and 'english' stemmed lexemes (weight D), so "deploy" also matches
"deployment". Identifiers are queried in the simple form, ordinary words in the stemmed one.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Shared with storage/index_store.py, which writes the same expression for new chunks.
SEARCH_VECTOR_SQL = """
    UPDATE chunks AS c
    SET search_vector =
        setweight(
            to_tsvector('simple', d.title || ' ' || array_to_string(c.heading_path, ' ')), 'A'
        )
        || setweight(to_tsvector('simple', c.content), 'B')
        || setweight(
            to_tsvector('english', d.title || ' ' || array_to_string(c.heading_path, ' ')
                        || ' ' || c.content),
            'D'
        )
    FROM documents AS d
    WHERE d.id = c.document_id
"""


def upgrade() -> None:
    op.add_column("chunks", sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True))
    op.execute(SEARCH_VECTOR_SQL)
    op.create_index(
        op.f("ix_chunks_search_vector"), "chunks", ["search_vector"], postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_chunks_search_vector"), table_name="chunks")
    op.drop_column("chunks", "search_vector")
