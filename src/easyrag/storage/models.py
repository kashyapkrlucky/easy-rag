"""ORM models: the single source of truth for the database schema.

Migrations under ``storage/migrations`` must produce exactly this schema; an
integration test fails if autogenerate detects any difference.

The ``embedding`` column on ``chunks`` is added in Phase 9 together with the
embedding model choice; the full-text column in Phase 17.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names, so later migrations can reference them reliably.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

SHA256_HEX = "^[0-9a-f]{64}$"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Document(Base):
    """One Markdown file under the docs root, identified by its relative path."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(f"content_hash ~ '{SHA256_HEX}'", name="content_hash_sha256"),
        CheckConstraint(f"pipeline_fingerprint ~ '{SHA256_HEX}'", name="fingerprint_sha256"),
        CheckConstraint("size_bytes >= 0", name="size_non_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=func.gen_random_uuid())
    # Relative POSIX path from the docs root, e.g. "guides/deployment.md".
    source_path: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    # Hash of parser/chunker version + parameters + embedding model (ARCHITECTURE D8).
    pipeline_fingerprint: Mapped[str] = mapped_column(String(64))
    front_matter: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    size_bytes: Mapped[int] = mapped_column(Integer)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.ordinal",
    )


class Chunk(Base):
    """A retrievable slice of one document section."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
        CheckConstraint("start_line >= 1 AND start_line <= end_line", name="line_range_valid"),
        CheckConstraint("token_estimate >= 0", name="token_estimate_non_negative"),
        CheckConstraint(f"content_hash ~ '{SHA256_HEX}'", name="content_hash_sha256"),
    )

    # Deterministic: derived from path, heading path, position and content (ARCHITECTURE D9).
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(Text))
    section_anchor: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    token_estimate: Mapped[int] = mapped_column(Integer)

    document: Mapped[Document] = relationship(back_populates="chunks")


class IndexState(Base):
    """Small key/value store for index-wide facts (e.g. embedding model and dimension)."""

    __tablename__ = "index_state"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
