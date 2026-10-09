import math

import pytest

from easyrag.embeddings.base import (
    EmbeddingProvider,
    EmbeddingResponseError,
    normalize,
    validate_embeddings,
)
from easyrag.embeddings.fake import FakeEmbeddingProvider
from tests.contracts.embedding_provider import EmbeddingProviderContract


def cosine(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))  # inputs are unit vectors


class TestFakeEmbeddingProviderContract(EmbeddingProviderContract):
    @pytest.fixture
    def provider(self) -> EmbeddingProvider:
        return FakeEmbeddingProvider()


# -- FakeEmbeddingProvider specifics ---------------------------------------------------


async def test_shared_words_make_texts_similar() -> None:
    provider = FakeEmbeddingProvider(dimension=256)
    docker, api, query = await provider.embed_documents(
        [
            "Deploy the application with Docker Compose.",
            "The REST API exposes a chat endpoint.",
            "docker compose deploy",
        ]
    )

    assert cosine(query, docker) > cosine(query, api)


async def test_query_and_document_embeddings_are_consistent() -> None:
    provider = FakeEmbeddingProvider()
    (document,) = await provider.embed_documents(["Docker deployment"])

    assert cosine(await provider.embed_query("Docker deployment"), document) == pytest.approx(1.0)


async def test_text_without_words_still_gets_a_unit_vector() -> None:
    provider = FakeEmbeddingProvider()
    first, second = await provider.embed_documents(["!!!", "???"])

    assert math.isclose(math.fsum(c * c for c in first), 1.0)
    assert first != second


async def test_vectors_are_stable_across_instances() -> None:
    # BLAKE2b, not Python's randomised hash(): stable across processes and runs.
    first = await FakeEmbeddingProvider().embed_documents(["stable"])
    second = await FakeEmbeddingProvider().embed_documents(["stable"])

    assert first == second


async def test_calls_are_recorded() -> None:
    provider = FakeEmbeddingProvider()

    await provider.embed_documents(["a", "b", "c"])
    await provider.embed_documents([])
    await provider.embed_documents(["d"])
    await provider.embed_query("q")

    assert provider.document_batches == [3, 1]
    assert provider.embedded_document_count == 4
    assert provider.query_count == 1


def test_identity_is_configurable() -> None:
    provider = FakeEmbeddingProvider(dimension=8, model_id="fake/test")

    assert (provider.model_id, provider.dimension) == ("fake/test", 8)


def test_dimension_must_be_usable() -> None:
    with pytest.raises(ValueError):
        FakeEmbeddingProvider(dimension=1)


# -- shared validation helpers --------------------------------------------------------


def test_normalize_produces_unit_vectors() -> None:
    assert normalize([3.0, 4.0]) == [0.6, 0.8]


@pytest.mark.parametrize("vector", [[0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0]])
def test_normalize_rejects_degenerate_vectors(vector: list[float]) -> None:
    with pytest.raises(ValueError):
        normalize(vector)


def test_validate_embeddings_normalises_valid_output() -> None:
    assert validate_embeddings([[3.0, 4.0], [0.0, 2.0]], expected_count=2, dimension=2) == [
        [0.6, 0.8],
        [0.0, 1.0],
    ]


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[1.0, 0.0]], "expected 2 embeddings, received 1"),
        ([[1.0, 0.0], [1.0, 0.0, 0.0]], "embedding 1 has dimension 3, expected 2"),
        ([[1.0, 0.0], [math.nan, 0.0]], "embedding 1 contains non-finite values"),
        ([[0.0, 0.0], [1.0, 0.0]], "embedding 0 is a zero vector"),
    ],
)
def test_validate_embeddings_rejects_contract_violations(
    vectors: list[list[float]], message: str
) -> None:
    with pytest.raises(EmbeddingResponseError, match=message):
        validate_embeddings(vectors, expected_count=2, dimension=2)
