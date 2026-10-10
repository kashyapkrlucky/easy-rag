import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from easyrag.domain.chunks import RetrievedChunk
from easyrag.rerankers.fastembed import FastEmbedReranker
from easyrag.retrieval.rerank import NoOpReranker, RerankerUnavailableError, RerankingRetriever


def chunk(chunk_id: str, content: str = "text", score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id, f"{chunk_id}.md", "Guide", ("Guide", chunk_id.upper()), chunk_id, content, 1, 1,
        score, vector_score=score,
    )  # fmt: skip


class StubRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append({"top_k": top_k, "min_score": min_score})
        return self.results[: top_k or len(self.results)]


class ReversingReranker:
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return list(reversed(chunks))[:top_n]


class UnavailableReranker:
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        raise RerankerUnavailableError("model is loading")


CANDIDATES = [chunk(f"c{i:02d}") for i in range(30)]


# -- RerankingRetriever ----------------------------------------------------------------


async def test_fetches_a_wider_candidate_set_and_keeps_top_k() -> None:
    base = StubRetriever(CANDIDATES)

    results = await RerankingRetriever(
        base=base, reranker=ReversingReranker(), candidates=20
    ).retrieve("q", top_k=3, min_score=0.2)

    assert base.calls == [{"top_k": 20, "min_score": 0.2}]
    assert [c.chunk_id for c in results] == ["c19", "c18", "c17"]  # reranked within the 20


async def test_candidates_are_at_least_top_k() -> None:
    base = StubRetriever(CANDIDATES)

    await RerankingRetriever(base=base, reranker=NoOpReranker(), candidates=5).retrieve(
        "q", top_k=10
    )

    assert base.calls[0]["top_k"] == 10


async def test_unavailable_reranker_degrades_to_retrieval_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    results = await RerankingRetriever(
        base=StubRetriever(CANDIDATES), reranker=UnavailableReranker()
    ).retrieve("q", top_k=3)

    assert [c.chunk_id for c in results] == ["c00", "c01", "c02"]
    assert "Reranking skipped (model is loading)" in caplog.text


async def test_no_candidates_skip_reranking() -> None:
    assert (
        await RerankingRetriever(base=StubRetriever([]), reranker=UnavailableReranker()).retrieve(
            "q"
        )
        == []
    )


@pytest.mark.parametrize("options", [{"candidates": 0}, {"candidates": 51}, {"default_top_k": 0}])
def test_invalid_configuration_is_rejected(options: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        RerankingRetriever(base=StubRetriever([]), reranker=NoOpReranker(), **options)


# -- FastEmbedReranker (with a fake encoder; the real model is an opt-in e2e test) -------


class FakeEncoder:
    """Scores documents by how many query words they contain, as a logit."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
        documents = list(documents)
        self.calls.append((query, documents))
        words = query.lower().split()
        return [float(sum(w in d.lower() for w in words)) - 1.0 for d in documents]


def loaded(encoder: object) -> FastEmbedReranker:
    reranker = FastEmbedReranker("fake/model", Path("/unused"))
    reranker._encoder = encoder  # type: ignore[assignment]
    reranker._state = "ready"
    return reranker


async def test_scores_are_sigmoid_probabilities_sorted_best_first() -> None:
    encoder = FakeEncoder()
    chunks = [chunk("a", "nothing here"), chunk("b", "docker compose up"), chunk("c", "docker")]

    results = await loaded(encoder).rerank("docker compose", chunks, top_n=2)

    assert [c.chunk_id for c in results] == ["b", "c"]
    assert results[0].rerank_score == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert results[0].score == results[0].rerank_score
    assert results[0].vector_score == 0.5  # component scores are kept


async def test_documents_carry_their_contextual_header() -> None:
    encoder = FakeEncoder()

    await loaded(encoder).rerank("q", [chunk("a", "Body text.")], top_n=1)

    assert encoder.calls[0][1] == ["Guide > A\n\nBody text."]


async def test_ties_are_broken_by_chunk_id() -> None:
    results = await loaded(FakeEncoder()).rerank("zzz", [chunk("b"), chunk("a")], top_n=2)

    assert [c.chunk_id for c in results] == ["a", "b"]


async def test_not_loaded_raises_unavailable() -> None:
    reranker = FastEmbedReranker("fake/model", Path("/unused"))

    assert await reranker.readiness() == "missing"
    with pytest.raises(RerankerUnavailableError, match="not started"):
        await reranker.rerank("q", [chunk("a")], top_n=1)


async def test_inference_failure_raises_unavailable() -> None:
    class Broken:
        def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
            raise RuntimeError("onnx session crashed")

    with pytest.raises(RerankerUnavailableError, match="onnx session crashed"):
        await loaded(Broken()).rerank("q", [chunk("a")], top_n=1)


async def test_failed_load_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = FastEmbedReranker("fake/model", Path("/unused"))

    def fail() -> object:
        raise OSError("no network and no cached model")

    monkeypatch.setattr(reranker, "_create_encoder", fail)
    await reranker.load()

    assert reranker.state == "failed"
    assert await reranker.readiness() == "unavailable"


async def test_ready_after_successful_load(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = FastEmbedReranker("fake/model", Path("/unused"))
    monkeypatch.setattr(reranker, "_create_encoder", FakeEncoder)

    reranker.start_loading()
    assert reranker._task is not None
    await reranker._task

    assert (reranker.state, await reranker.readiness()) == ("ready", "ok")
    assert reranker.model_id == "fastembed/fake/model"
