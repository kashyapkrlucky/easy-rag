"""Ollama embedding adapter (``POST /api/embed``).

Ollama-specific details stay in this module: the HTTP API, error payloads, model
prefixes and model metadata. The rest of the application sees only
``EmbeddingProvider`` and the embedding errors.
"""

import hashlib
import logging
from collections.abc import Sequence

import httpx2

from easyrag.embeddings.base import (
    EmbeddingProviderUnavailableError,
    EmbeddingResponseError,
    Vector,
    validate_embeddings,
)
from easyrag.health import CheckStatus
from easyrag.ollama import (
    OllamaModelNotFoundError,
    OllamaRequestError,
    OllamaUnavailableError,
    model_info_value,
    post_json,
    show_model,
)

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    def __init__(
        self,
        client: httpx2.AsyncClient,
        *,
        model: str,
        dimension: int,
        batch_size: int = 32,
        query_prefix: str = "",
        document_prefix: str = "",
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._client = client
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    @property
    def model_id(self) -> str:
        """``ollama/<model>``, plus a short hash of the prefixes when any are set.

        Prefixes change every vector, so vectors produced with different prefixes belong
        to different vector spaces and must not share an index.
        """
        if not (self._query_prefix or self._document_prefix):
            return f"ollama/{self._model}"
        prefixes = f"{self._query_prefix}\x1f{self._document_prefix}".encode()
        return f"ollama/{self._model}+p:{hashlib.sha256(prefixes).hexdigest()[:8]}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed([self._document_prefix + text for text in batch]))
        return vectors

    async def embed_query(self, text: str) -> Vector:
        (vector,) = await self._embed([self._query_prefix + text])
        return vector

    async def readiness(self) -> CheckStatus:
        """Is the model available, and does its output size match the configured dimension?"""
        metadata = await show_model(self._client, self._model)
        if metadata is None:
            return "missing"
        reported = model_info_value(metadata, ".embedding_length")
        if reported is not None and reported != self._dimension:
            logger.error(
                "Embedding model %s produces %d dimensions but EASYRAG_EMBEDDING_DIMENSION=%d",
                self._model,
                reported,
                self._dimension,
            )
            return "misconfigured"
        return "ok"

    async def _embed(self, inputs: list[str]) -> list[Vector]:
        # truncate=False: fail loudly instead of silently embedding only part of a chunk.
        payload = {"model": self._model, "input": inputs, "truncate": False}
        try:
            data = await post_json(
                self._client,
                "/api/embed",
                payload,
                max_retries=self._max_retries,
                retry_backoff_seconds=self._retry_backoff_seconds,
            )
        except OllamaModelNotFoundError as exc:
            raise EmbeddingProviderUnavailableError(
                f"embedding {exc}; pull it with: ollama pull {self._model}"
            ) from exc
        except OllamaUnavailableError as exc:
            raise EmbeddingProviderUnavailableError(str(exc)) from exc
        except OllamaRequestError as exc:
            raise EmbeddingResponseError(str(exc)) from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingResponseError("Ollama response has no 'embeddings' list")
        return validate_embeddings(
            embeddings, expected_count=len(inputs), dimension=self._dimension
        )
