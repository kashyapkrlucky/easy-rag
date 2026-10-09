"""Contract tests for ``EmbeddingProvider`` implementations.

Usage — subclass and provide a ``provider`` fixture:

    class TestMyProviderContract(EmbeddingProviderContract):
        @pytest.fixture
        def provider(self) -> EmbeddingProvider:
            return MyProvider(...)
"""

import math
from collections.abc import Sequence

from easyrag.embeddings.base import EmbeddingProvider

# Real models may differ in the last bits between calls (e.g. GPU kernels).
TOLERANCE = 1e-4

TEXTS = [
    "Deploy the application with Docker Compose.",
    "The API exposes a chat endpoint.",
    "Configuration is read from environment variables.",
]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(component * component for component in vector))


def _close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= TOLERANCE for a, b in zip(left, right, strict=True)
    )


class EmbeddingProviderContract:
    def test_identity(self, provider: EmbeddingProvider) -> None:
        assert isinstance(provider.model_id, str) and provider.model_id
        assert isinstance(provider.dimension, int) and provider.dimension > 0

    async def test_one_vector_per_text_with_correct_shape(
        self, provider: EmbeddingProvider
    ) -> None:
        vectors = await provider.embed_documents(TEXTS)

        assert len(vectors) == len(TEXTS)
        for vector in vectors:
            assert len(vector) == provider.dimension
            assert all(isinstance(c, float) and math.isfinite(c) for c in vector)
            assert abs(_norm(vector) - 1.0) <= TOLERANCE

    async def test_vectors_are_returned_in_input_order(self, provider: EmbeddingProvider) -> None:
        batch = await provider.embed_documents(TEXTS)
        reversed_batch = await provider.embed_documents(TEXTS[::-1])

        for vector, expected in zip(batch, reversed_batch[::-1], strict=True):
            assert _close(vector, expected)

    async def test_batching_does_not_change_results(self, provider: EmbeddingProvider) -> None:
        batch = await provider.embed_documents(TEXTS)
        singles = [(await provider.embed_documents([text]))[0] for text in TEXTS]

        for from_batch, single in zip(batch, singles, strict=True):
            assert _close(from_batch, single)

    async def test_embedding_is_deterministic(self, provider: EmbeddingProvider) -> None:
        first = await provider.embed_documents(TEXTS[:1])
        second = await provider.embed_documents(TEXTS[:1])

        assert _close(first[0], second[0])

    async def test_empty_input_returns_empty_list(self, provider: EmbeddingProvider) -> None:
        assert await provider.embed_documents([]) == []

    async def test_query_vector_has_correct_shape(self, provider: EmbeddingProvider) -> None:
        vector = await provider.embed_query("How do I deploy the application?")

        assert len(vector) == provider.dimension
        assert abs(_norm(vector) - 1.0) <= TOLERANCE

    async def test_unicode_text_is_supported(self, provider: EmbeddingProvider) -> None:
        vectors = await provider.embed_documents(["Café crème — 検索拡張生成 🚀"])

        assert len(vectors) == 1 and len(vectors[0]) == provider.dimension
