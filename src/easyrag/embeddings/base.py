"""Embedding provider port.

Every implementation must honour this contract (enforced by the shared test suite in
``tests/contracts/embedding_provider.py``):

- ``embed_documents`` returns exactly one vector per input text, in input order;
- every vector has exactly ``dimension`` finite floats and unit L2 norm;
- an empty input returns ``[]`` without contacting the backend;
- the same text always yields the same vector (within floating-point tolerance).

Documents and queries are embedded through separate methods because many embedding
models expect different prefixes or instructions for each.
"""

import math
from collections.abc import Sequence
from typing import Protocol

Vector = list[float]


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingProviderUnavailableError(EmbeddingError):
    """The backend cannot be reached, timed out, or the model is not available. Retryable."""


class EmbeddingResponseError(EmbeddingError):
    """The backend answered, but with output that violates the contract."""


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str:
        """Identifier stored with every vector, e.g. ``"ollama/nomic-embed-text"``."""
        ...

    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    async def embed_query(self, text: str) -> Vector: ...


def normalize(vector: Sequence[float]) -> Vector:
    """Scale ``vector`` to unit L2 norm. Raises ``ValueError`` for zero or non-finite input."""
    norm = math.sqrt(math.fsum(component * component for component in vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalise a zero or non-finite vector")
    return [component / norm for component in vector]


def validate_embeddings(
    vectors: Sequence[Sequence[float]], *, expected_count: int, dimension: int
) -> list[Vector]:
    """Check backend output against the contract and return unit-normalised copies.

    Raises ``EmbeddingResponseError`` on a wrong count, wrong dimension, non-finite
    components or a zero vector.
    """
    if len(vectors) != expected_count:
        raise EmbeddingResponseError(
            f"expected {expected_count} embeddings, received {len(vectors)}"
        )
    validated: list[Vector] = []
    for position, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise EmbeddingResponseError(
                f"embedding {position} has dimension {len(vector)}, expected {dimension}"
            )
        if not all(math.isfinite(component) for component in vector):
            raise EmbeddingResponseError(f"embedding {position} contains non-finite values")
        try:
            validated.append(normalize(vector))
        except ValueError as exc:
            raise EmbeddingResponseError(f"embedding {position} is a zero vector") from exc
    return validated
