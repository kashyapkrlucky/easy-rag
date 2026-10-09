"""Chunk embeddings: dimension-agnostic vector column and producing model.

The HNSW index is not created here: it is dimension-specific and managed by
PostgresIndexStore (see ARCHITECTURE D11). No writer existed before this revision, so
``chunks`` is empty and the NOT NULL columns need no default.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embedding", Vector(), nullable=False))
    op.add_column("chunks", sa.Column("embedding_model", sa.Text(), nullable=False))


def downgrade() -> None:
    # Dropping the column also drops any app-managed expression index on it.
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "embedding")
