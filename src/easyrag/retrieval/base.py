"""Retrieval ports.

``Retriever`` is what the API and (later) the RAG service depend on. ``VectorSearch``
is the storage capability a vector retriever needs; it is owned here, by its consumer,
and implemented by ``storage.vector_store.PostgresVectorStore``.
"""

from collections.abc import Sequence
from typing import Protocol

from easyrag.domain.chunks import RetrievedChunk


class Retriever(Protocol):
    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        """The most relevant chunks for ``query``, best first.

        ``top_k`` / ``min_score`` override the retriever's configured defaults when given.
        Results are deterministic for a given index state: ties are ordered by chunk ID.
        """
        ...


class VectorSearch(Protocol):
    async def search(
        self,
        vector: Sequence[float],
        *,
        embedding_model: str,
        dimension: int,
        top_k: int,
        min_score: float | None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks by cosine similarity among vectors from ``embedding_model``.

        Raises ``IndexIncompatibleError`` if the index was built with another model or
        dimension; returns ``[]`` for an empty index.
        """
        ...
