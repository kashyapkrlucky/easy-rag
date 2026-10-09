"""Embedding reuse by input hash.

Adds chunks.embedding_input_hash (SHA-256 of the embedded text: contextual header +
content) with an index on (embedding_input_hash, embedding_model), and drops the
content_hash index, which was planned for reuse but would reuse stale vectors when only
a heading changed. Existing rows keep NULL and are simply not reused.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embedding_input_hash", sa.String(length=64), nullable=True))
    op.create_index(
        op.f("ix_chunks_embedding_input_hash"),
        "chunks",
        ["embedding_input_hash", "embedding_model"],
    )
    op.drop_index(op.f("ix_chunks_content_hash"), table_name="chunks")


def downgrade() -> None:
    op.create_index(op.f("ix_chunks_content_hash"), "chunks", ["content_hash"])
    op.drop_index(op.f("ix_chunks_embedding_input_hash"), table_name="chunks")
    op.drop_column("chunks", "embedding_input_hash")
