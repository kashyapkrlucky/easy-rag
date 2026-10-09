"""Hybrid retrieval: vector + keyword results merged by Reciprocal Rank Fusion.

RRF uses only ranks, which matters because cosine similarity and keyword coverage live on
incomparable scales:  fused(c) = Σ_lists 1 / (RRF_K + rank_list(c)).  Ties are broken
by chunk ID, so the result is deterministic.
"""

import asyncio
from dataclasses import dataclass, replace

from easyrag.domain.chunks import RetrievedChunk
from easyrag.retrieval.base import Retriever
from easyrag.retrieval.vector import MAX_TOP_K

# The constant from the original RRF paper (Cormack et al., 2009); damps the advantage of
# the very top ranks so that agreement between the lists dominates.
RRF_K = 60
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 20


@dataclass(slots=True)
class _Fused:
    chunk: RetrievedChunk
    score: float = 0.0
    vector_score: float | None = None
    keyword_score: float | None = None


class HybridRetriever:
    def __init__(self, *, vector: Retriever, keyword: Retriever, default_top_k: int = 5) -> None:
        if not 1 <= default_top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        self._vector = vector
        self._keyword = keyword
        self._default_top_k = default_top_k

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        """``min_score`` (cosine) filters the vector candidates; keyword hits are not filtered."""
        top_k = self._default_top_k if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        candidates = min(max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES), MAX_TOP_K)

        vector_results, keyword_results = await asyncio.gather(
            self._vector.retrieve(query, top_k=candidates, min_score=min_score),
            self._keyword.retrieve(query, top_k=candidates),
        )
        return reciprocal_rank_fusion(vector_results, keyword_results)[:top_k]


def reciprocal_rank_fusion(
    vector_results: list[RetrievedChunk], keyword_results: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    fused: dict[str, _Fused] = {}
    for rank, chunk in enumerate(vector_results, start=1):
        entry = fused.setdefault(chunk.chunk_id, _Fused(chunk))
        entry.score += 1 / (RRF_K + rank)
        entry.vector_score = chunk.vector_score if chunk.vector_score is not None else chunk.score
    for rank, chunk in enumerate(keyword_results, start=1):
        entry = fused.setdefault(chunk.chunk_id, _Fused(chunk))
        entry.score += 1 / (RRF_K + rank)
        entry.keyword_score = (
            chunk.keyword_score if chunk.keyword_score is not None else chunk.score
        )
    ordered = sorted(fused.values(), key=lambda e: (-e.score, e.chunk.chunk_id))
    return [
        replace(e.chunk, score=e.score, vector_score=e.vector_score, keyword_score=e.keyword_score)
        for e in ordered
    ]
