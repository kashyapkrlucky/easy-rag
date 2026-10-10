"""Reranking: reorder retrieved candidates with a more precise (and slower) model.

Retrieval is tuned for recall; a reranker such as a cross-encoder reads query and chunk
together and judges relevance more precisely. ``RerankingRetriever`` wraps any retriever:
it fetches a wider candidate set, reranks it and keeps the best ``top_k``.

Reranking is an optional precision stage, so a reranker that is not ready (or fails)
degrades to the base retriever's order instead of failing the request.
"""

import logging
from collections.abc import Sequence
from typing import Protocol

from easyrag.domain.chunks import RetrievedChunk
from easyrag.retrieval.base import Retriever
from easyrag.retrieval.vector import MAX_TOP_K

logger = logging.getLogger(__name__)


class RerankerUnavailableError(Exception):
    """The reranker cannot score right now (e.g. its model is still loading)."""


class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """The ``top_n`` most relevant chunks, best first, with ``rerank_score`` set.

        Raises ``RerankerUnavailableError`` when it cannot score.
        """
        ...


class RerankingRetriever:
    def __init__(
        self,
        *,
        base: Retriever,
        reranker: Reranker,
        candidates: int = 20,
        default_top_k: int = 5,
    ) -> None:
        if not 1 <= candidates <= MAX_TOP_K:
            raise ValueError(f"candidates must be between 1 and {MAX_TOP_K}")
        if not 1 <= default_top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        self._base = base
        self._reranker = reranker
        self._candidates = candidates
        self._default_top_k = default_top_k

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        top_k = self._default_top_k if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        candidates = await self._base.retrieve(
            query, top_k=max(self._candidates, top_k), min_score=min_score
        )
        if not candidates:
            return []
        try:
            return await self._reranker.rerank(query, candidates, top_k)
        except RerankerUnavailableError as exc:
            logger.warning("Reranking skipped (%s); using retrieval order", exc)
            return candidates[:top_k]


class NoOpReranker:
    """Keeps the retrieval order; for tests and for explicitly disabled reranking."""

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return list(chunks[:top_n])
