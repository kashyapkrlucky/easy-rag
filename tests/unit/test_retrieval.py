from collections.abc import Sequence
from typing import Any

import pytest

from easyrag.domain.chunks import RetrievedChunk
from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.retrieval.vector import VectorRetriever


def retrieved(
    chunk_id: str, score: float = 0.5, heading_path: tuple[str, ...] = ()
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_path="guide.md",
        document_title="Guide",
        heading_path=heading_path,
        section_anchor="",
        content="text",
        start_line=1,
        end_line=1,
        score=score,
    )


class RecordingSearch:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.results = results or []

    async def search(self, vector: Sequence[float], **options: Any) -> list[RetrievedChunk]:
        self.calls.append({"vector": list(vector), **options})
        return self.results


async def test_query_is_embedded_as_a_query_and_searched_in_its_vector_space() -> None:
    embeddings = FakeEmbeddingProvider(dimension=16, model_id="fake/m")
    search = RecordingSearch([retrieved("a")])
    retriever = VectorRetriever(embeddings=embeddings, search=search, default_top_k=7)

    results = await retriever.retrieve("how to deploy")

    assert results == [retrieved("a")]
    assert embeddings.query_count == 1 and embeddings.document_batches == []
    (call,) = search.calls
    assert call["vector"] == await FakeEmbeddingProvider(16).embed_query("how to deploy")
    assert (call["embedding_model"], call["dimension"]) == ("fake/m", 16)
    assert (call["top_k"], call["min_score"]) == (7, None)


async def test_request_values_override_defaults() -> None:
    search = RecordingSearch()
    retriever = VectorRetriever(
        embeddings=FakeEmbeddingProvider(), search=search, default_top_k=5, default_min_score=0.2
    )

    await retriever.retrieve("q")
    await retriever.retrieve("q", top_k=3, min_score=0.6)

    assert [(c["top_k"], c["min_score"]) for c in search.calls] == [(5, 0.2), (3, 0.6)]


@pytest.mark.parametrize("options", [{"top_k": 0}, {"top_k": 51}, {"min_score": 1.5}])
async def test_invalid_options_are_rejected(options: dict[str, Any]) -> None:
    retriever = VectorRetriever(embeddings=FakeEmbeddingProvider(), search=RecordingSearch())

    with pytest.raises(ValueError):
        await retriever.retrieve("q", **options)


async def test_blank_query_is_rejected_without_embedding() -> None:
    embeddings = FakeEmbeddingProvider()

    with pytest.raises(ValueError, match="blank"):
        await VectorRetriever(embeddings=embeddings, search=RecordingSearch()).retrieve("   ")
    assert embeddings.query_count == 0


def test_invalid_defaults_are_rejected() -> None:
    with pytest.raises(ValueError):
        VectorRetriever(
            embeddings=FakeEmbeddingProvider(), search=RecordingSearch(), default_top_k=0
        )


def test_section_is_deepest_heading_or_document_title() -> None:
    assert retrieved("a", heading_path=("Deploy", "Docker")).section == "Docker"
    assert retrieved("a").section == "Guide"
