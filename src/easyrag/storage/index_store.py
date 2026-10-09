"""PostgreSQL + pgvector implementation of the indexer's ``IndexStore``.

Besides rows, this store owns the dimension-specific HNSW expression index
(``ix_chunks_embedding_hnsw_d<N>``). It cannot live in a migration because the
dimension is runtime configuration; the vector column itself is dimension-agnostic.
"""

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from sqlalchemy import Table, bindparam, delete, func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from easyrag.domain.errors import IndexIncompatibleError
from easyrag.embeddings.base import Vector
from easyrag.health import CheckStatus
from easyrag.ingestion.indexer import DocumentToStore, StoredDocumentState, embedding_input_hash
from easyrag.storage.models import APP_MANAGED_INDEX_PREFIX, Chunk, Document, IndexState

logger = logging.getLogger(__name__)

MODEL_KEY = "embedding_model"
DIMENSION_KEY = "embedding_dimension"

# Arbitrary constant: one indexing writer at a time across processes (API, CLI, watcher).
INDEX_LOCK_KEY = 0x6561737972616702


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """Read model of an indexed document (for the API)."""

    source_path: str
    title: str
    content_hash: str
    size_bytes: int
    chunk_count: int
    indexed_at: datetime


_documents = cast(Table, Document.__table__)
_chunks = cast(Table, Chunk.__table__)
_index_state = cast(Table, IndexState.__table__)


def vector_index_name(dimension: int) -> str:
    return f"{APP_MANAGED_INDEX_PREFIX}d{dimension}"


class PostgresIndexStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def prepare(self, embedding_model: str, dimension: int) -> None:
        if not isinstance(dimension, int) or dimension < 1:  # interpolated into DDL below
            raise ValueError(f"invalid embedding dimension: {dimension!r}")

        async with self._engine.begin() as connection:
            await connection.execute(
                pg_insert(_index_state)
                .values(
                    [
                        {"key": MODEL_KEY, "value": embedding_model},
                        {"key": DIMENSION_KEY, "value": str(dimension)},
                    ]
                )
                .on_conflict_do_nothing(index_elements=["key"])
            )
            state = await read_index_state(connection)
            stored_model, stored_dimension = state[MODEL_KEY], int(state[DIMENSION_KEY])
            if (stored_model, stored_dimension) != (embedding_model, dimension):
                raise IndexIncompatibleError(
                    stored_model=stored_model,
                    stored_dimension=stored_dimension,
                    configured_model=embedding_model,
                    configured_dimension=dimension,
                )
            # pgvector's documented pattern for a dimension-agnostic column: an expression
            # index over a cast to the fixed dimension. Queries must use the same cast.
            await connection.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {vector_index_name(dimension)} ON chunks "
                    f"USING hnsw ((embedding::vector({dimension})) vector_cosine_ops)"
                )
            )

    @asynccontextmanager
    async def lock(self) -> AsyncIterator[None]:
        # A session-level advisory lock on a dedicated connection: released explicitly,
        # or automatically by PostgreSQL if the process dies and the connection drops.
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": INDEX_LOCK_KEY})
            await connection.commit()
            try:
                yield
            finally:
                await connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": INDEX_LOCK_KEY}
                )
                await connection.commit()

    async def document_states(self) -> Mapping[str, StoredDocumentState]:
        async with self._engine.connect() as connection:
            rows = await connection.execute(
                select(
                    _documents.c.source_path,
                    _documents.c.content_hash,
                    _documents.c.pipeline_fingerprint,
                )
            )
            return {
                path: StoredDocumentState(digest, fingerprint) for path, digest, fingerprint in rows
            }

    async def find_embeddings(
        self, input_hashes: Sequence[str], embedding_model: str
    ) -> Mapping[str, Vector]:
        if not input_hashes:
            return {}
        query = text(
            "SELECT DISTINCT ON (embedding_input_hash) embedding_input_hash, embedding::text "
            "FROM chunks WHERE embedding_model = :model AND embedding_input_hash IN :hashes"
        ).bindparams(bindparam("hashes", expanding=True))
        async with self._engine.connect() as connection:
            rows = await connection.execute(
                query, {"model": embedding_model, "hashes": list(input_hashes)}
            )
            return {digest: _parse_vector(literal) for digest, literal in rows}

    async def delete_documents(self, source_paths: Sequence[str]) -> None:
        if not source_paths:
            return
        async with self._engine.begin() as connection:
            await connection.execute(
                delete(_documents).where(_documents.c.source_path.in_(list(source_paths)))
            )
        logger.info("Deleted %d removed documents from the index", len(source_paths))

    async def list_documents(self) -> list[IndexedDocument]:
        chunk_counts = (
            select(_chunks.c.document_id, func.count().label("chunks"))
            .group_by(_chunks.c.document_id)
            .subquery()
        )
        query = (
            select(
                _documents.c.source_path,
                _documents.c.title,
                _documents.c.content_hash,
                _documents.c.size_bytes,
                func.coalesce(chunk_counts.c.chunks, 0),
                _documents.c.indexed_at,
            )
            .outerjoin(chunk_counts, chunk_counts.c.document_id == _documents.c.id)
            .order_by(_documents.c.source_path)
        )
        async with self._engine.connect() as connection:
            return [IndexedDocument(*row) for row in await connection.execute(query)]

    async def counts(self) -> tuple[int, int]:
        """(documents, chunks) currently indexed."""
        async with self._engine.connect() as connection:
            documents = (
                await connection.execute(select(func.count()).select_from(_documents))
            ).scalar_one()
            chunks = (
                await connection.execute(select(func.count()).select_from(_chunks))
            ).scalar_one()
        return int(documents), int(chunks)

    async def save_document(self, document: DocumentToStore) -> None:
        parsed = document.document
        values = {
            "source_path": parsed.source_path,
            "title": parsed.title,
            "content_hash": parsed.content_hash,
            "pipeline_fingerprint": document.pipeline_fingerprint,
            "front_matter": dict(parsed.front_matter),
            "size_bytes": document.size_bytes,
        }
        upsert = (
            pg_insert(_documents)
            .values(values)
            .on_conflict_do_update(
                index_elements=["source_path"],
                set_={**values, "indexed_at": func.now()},
            )
            .returning(_documents.c.id)
        )
        async with self._engine.begin() as connection:
            document_id = (await connection.execute(upsert)).scalar_one()
            await connection.execute(delete(_chunks).where(_chunks.c.document_id == document_id))
            if document.chunks:
                await connection.execute(
                    insert(_chunks),
                    [
                        {
                            "id": chunk.id,
                            "document_id": document_id,
                            "ordinal": chunk.ordinal,
                            "heading_path": list(chunk.heading_path),
                            "section_anchor": chunk.section_anchor,
                            "content": chunk.content,
                            "content_hash": chunk.content_hash,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "token_estimate": chunk.token_estimate,
                            "embedding_input_hash": embedding_input_hash(chunk),
                            "embedding": vector,
                            "embedding_model": document.embedding_model,
                        }
                        for chunk, vector in zip(document.chunks, document.embeddings, strict=True)
                    ],
                )

    async def reset(self) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(delete(_documents))  # chunks cascade
            await connection.execute(
                delete(_index_state).where(_index_state.c.key.in_([MODEL_KEY, DIMENSION_KEY]))
            )
            for name in await _app_managed_indexes(connection):
                await connection.execute(text(f'DROP INDEX IF EXISTS "{name}"'))
        logger.info("Index reset: all documents, chunks and embedding settings deleted")

    async def compatibility(self, embedding_model: str, dimension: int) -> CheckStatus:
        """Readiness: "misconfigured" when the index was built with another model/dimension."""
        async with self._engine.connect() as connection:
            state = await read_index_state(connection)
        if not state:
            return "ok"  # nothing indexed yet
        if (state.get(MODEL_KEY), state.get(DIMENSION_KEY)) != (embedding_model, str(dimension)):
            logger.error(
                "Index was built with %s (%s dimensions); configured: %s (%d). "
                "Run `easyrag index --rebuild`.",
                state.get(MODEL_KEY),
                state.get(DIMENSION_KEY),
                embedding_model,
                dimension,
            )
            return "misconfigured"
        return "ok"


async def read_index_state(connection: AsyncConnection) -> Mapping[str, str]:
    """The recorded embedding model and dimension ({} when nothing has been indexed)."""
    rows = await connection.execute(
        select(_index_state.c.key, _index_state.c.value).where(
            _index_state.c.key.in_([MODEL_KEY, DIMENSION_KEY])
        )
    )
    return {key: value for key, value in rows}


async def _app_managed_indexes(connection: AsyncConnection) -> list[str]:
    rows = await connection.execute(
        text(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname = current_schema() AND tablename = 'chunks' "
            "AND starts_with(indexname, :prefix)"
        ),
        {"prefix": APP_MANAGED_INDEX_PREFIX},
    )
    return list(rows.scalars())


def _parse_vector(literal: str) -> Vector:
    """pgvector's text form "[0.1,0.2,…]" → list of floats (avoids a numpy dependency)."""
    return [float(component) for component in literal.strip("[]").split(",")]
