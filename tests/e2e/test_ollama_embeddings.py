import math

import pytest

from easyrag.embeddings.base import EmbeddingProvider
from easyrag.embeddings.ollama import OllamaEmbeddingProvider
from tests.contracts.embedding_provider import EmbeddingProviderContract


def cosine(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


class TestRealOllamaContract(EmbeddingProviderContract):
    @pytest.fixture
    def provider(self, ollama_embeddings: OllamaEmbeddingProvider) -> EmbeddingProvider:
        return ollama_embeddings


async def test_model_is_ready(ollama_embeddings: OllamaEmbeddingProvider) -> None:
    assert await ollama_embeddings.readiness() == "ok"


async def test_semantically_related_text_ranks_higher(
    ollama_embeddings: OllamaEmbeddingProvider,
) -> None:
    # No shared keywords with the right answer: this needs a real semantic model.
    deployment, billing, cooking = await ollama_embeddings.embed_documents(
        [
            "Start the containers with docker compose up and check the health endpoint.",
            "Invoices are generated on the first day of every month.",
            "Simmer the tomato sauce for twenty minutes before adding basil.",
        ]
    )
    query = await ollama_embeddings.embed_query("How do I run the service?")

    scores = [cosine(query, d) for d in (deployment, billing, cooking)]
    assert scores[0] == max(scores), scores
