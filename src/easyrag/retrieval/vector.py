"""Dense (vector) retrieval: embed the query, return the nearest chunks."""

import logging

from easyrag.domain.chunks import RetrievedChunk
from easyrag.embeddings.base import EmbeddingProvider
from easyrag.retrieval.base import VectorSearch

logger = logging.getLogger(__name__)

MAX_TOP_K = 50


class VectorRetriever:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        search: VectorSearch,
        default_top_k: int = 5,
        default_min_score: float | None = None,
    ) -> None:
        _validate(default_top_k, default_min_score)
        self._embeddings = embeddings
        self._search = search
        self._default_top_k = default_top_k
        self._default_min_score = default_min_score

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        top_k = self._default_top_k if top_k is None else top_k
        min_score = self._default_min_score if min_score is None else min_score
        _validate(top_k, min_score)
        if not query.strip():
            raise ValueError("query must not be blank")

        vector = await self._embeddings.embed_query(query)
        results = await self._search.search(
            vector,
            embedding_model=self._embeddings.model_id,
            dimension=self._embeddings.dimension,
            top_k=top_k,
            min_score=min_score,
        )
        logger.debug("Retrieved %d chunks (top_k=%d, min_score=%s)", len(results), top_k, min_score)
        return results


def _validate(top_k: int, min_score: float | None) -> None:
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if min_score is not None and not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1 and 1")
