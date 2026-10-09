"""PostgreSQL + pgvector implementation of ``VectorSearch`` (the read side of the index)."""

from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.domain.chunks import RetrievedChunk
from easyrag.domain.errors import IndexIncompatibleError
from easyrag.storage.index_store import DIMENSION_KEY, MODEL_KEY, read_index_state


def search_sql(dimension: int, *, with_min_score: bool) -> str:
    """SQL for a k-nearest-neighbour search.

    The inner query is a *pure* k-nearest-neighbour search: it orders by exactly the
    expression of the HNSW index (``embedding::vector(N) <=> query``) and has no WHERE
    clause, so nothing tempts the planner away from the index (pgvector's index only serves
    a single distance ``ORDER BY``; a filter usable by a btree index — PostgreSQL 18 skip
    scans included — made it sort all rows instead). The outer query breaks ties by chunk ID
    (determinism) and applies the model filter (defence in depth: the search already
    refuses a mismatched index) and the score threshold *after* the top-k selection.
    """
    if not isinstance(dimension, int) or dimension < 1:  # interpolated into SQL
        raise ValueError(f"invalid embedding dimension: {dimension!r}")
    distance = f"c.embedding::vector({dimension}) <=> CAST(:query AS vector({dimension}))"
    threshold = "AND 1 - nearest.distance >= :min_score" if with_min_score else ""
    return f"""
        SELECT nearest.*, 1 - nearest.distance AS score
        FROM (
            SELECT c.id AS chunk_id, d.source_path, d.title AS document_title,
                   c.heading_path, c.section_anchor, c.content, c.start_line, c.end_line,
                   c.embedding_model, {distance} AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY {distance}
            LIMIT :top_k
        ) AS nearest
        WHERE nearest.embedding_model = :embedding_model
        {threshold}
        ORDER BY nearest.distance, nearest.chunk_id
    """


class PostgresVectorStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def search(
        self,
        vector: Sequence[float],
        *,
        embedding_model: str,
        dimension: int,
        top_k: int,
        min_score: float | None,
    ) -> list[RetrievedChunk]:
        if len(vector) != dimension:
            raise ValueError(f"query vector has dimension {len(vector)}, expected {dimension}")

        parameters: dict[str, object] = {
            "query": "[" + ",".join(repr(float(component)) for component in vector) + "]",
            "embedding_model": embedding_model,
            "top_k": top_k,
        }
        if min_score is not None:
            parameters["min_score"] = min_score

        async with self._engine.connect() as connection:
            state = await read_index_state(connection)
            if not state:
                return []  # nothing indexed yet
            stored = (state[MODEL_KEY], int(state[DIMENSION_KEY]))
            if stored != (embedding_model, dimension):
                raise IndexIncompatibleError(
                    stored_model=stored[0],
                    stored_dimension=stored[1],
                    configured_model=embedding_model,
                    configured_dimension=dimension,
                )
            rows = await connection.execute(
                text(search_sql(dimension, with_min_score=min_score is not None)), parameters
            )
            return [
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    source_path=row.source_path,
                    document_title=row.document_title,
                    heading_path=tuple(row.heading_path),
                    section_anchor=row.section_anchor,
                    content=row.content,
                    start_line=row.start_line,
                    end_line=row.end_line,
                    score=float(row.score),
                    vector_score=float(row.score),
                )
                for row in rows
            ]
