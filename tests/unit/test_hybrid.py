from typing import Any

import pytest

from easyrag.domain.chunks import RetrievedChunk
from easyrag.retrieval.hybrid import RRF_K, HybridRetriever, reciprocal_rank_fusion


def chunk(chunk_id: str, score: float = 0.5, **scores: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id, f"{chunk_id}.md", chunk_id, (), "", "text", 1, 1, score, **scores
    )


class Stub:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append({"top_k": top_k, "min_score": min_score})
        return self.results[: top_k or len(self.results)]


def test_rrf_rewards_agreement_between_lists() -> None:
    vector = [
        chunk("a", vector_score=0.9),
        chunk("b", vector_score=0.8),
        chunk("c", vector_score=0.7),
    ]
    keyword = [chunk("c", keyword_score=0.9), chunk("d", keyword_score=0.5)]

    fused = reciprocal_rank_fusion(vector, keyword)

    assert [c.chunk_id for c in fused] == ["c", "a", "b", "d"]  # c is in both lists
    c = fused[0]
    assert c.score == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))
    assert (c.vector_score, c.keyword_score) == (0.7, 0.9)
    assert (fused[1].vector_score, fused[1].keyword_score) == (0.9, None)
    assert (fused[3].vector_score, fused[3].keyword_score) == (None, 0.5)


def test_rrf_breaks_ties_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion([chunk("z")], [chunk("y")])  # both rank 1 in one list

    assert [c.chunk_id for c in fused] == ["y", "z"]
    assert reciprocal_rank_fusion([chunk("z")], [chunk("y")]) == fused  # deterministic


def test_rrf_uses_plain_scores_when_component_scores_are_absent() -> None:
    (fused,) = reciprocal_rank_fusion([chunk("a", score=0.42)], [])

    assert fused.vector_score == 0.42


async def test_hybrid_widens_candidates_and_returns_top_k() -> None:
    vector = Stub([chunk(f"v{i}") for i in range(30)])
    keyword = Stub([chunk(f"k{i}") for i in range(30)])

    results = await HybridRetriever(vector=vector, keyword=keyword, default_top_k=5).retrieve("q")

    assert len(results) == 5
    assert vector.calls[0]["top_k"] == keyword.calls[0]["top_k"] == 20  # max(4·5, 20)


async def test_min_score_applies_to_the_vector_side_only() -> None:
    vector, keyword = Stub([]), Stub([])

    await HybridRetriever(vector=vector, keyword=keyword).retrieve("q", top_k=15, min_score=0.6)

    assert vector.calls == [{"top_k": 50, "min_score": 0.6}]  # min(4·15, 50)
    assert keyword.calls == [{"top_k": 50, "min_score": None}]


@pytest.mark.parametrize("top_k", [0, 51])
async def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError):
        await HybridRetriever(vector=Stub([]), keyword=Stub([])).retrieve("q", top_k=top_k)
