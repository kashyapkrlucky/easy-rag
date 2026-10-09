"""Ollama embedding adapter (``POST /api/embed``).

Ollama-specific details stay in this module: the HTTP API, error payloads, model
prefixes and model metadata. The rest of the application sees only
``EmbeddingProvider`` and the embedding errors.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import httpx2

from easyrag.embeddings.base import (
    EmbeddingError,
    EmbeddingProviderUnavailableError,
    EmbeddingResponseError,
    Vector,
    validate_embeddings,
)
from easyrag.health import CheckStatus

logger = logging.getLogger(__name__)

# Statuses worth retrying: rate limiting, overload (Ollama's queue is full) and server errors.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


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
        return f"ollama/{self._model}"

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
        response = await self._client.post("/api/show", json={"model": self._model})
        if response.status_code == 404:
            return "missing"
        response.raise_for_status()
        model_info = response.json().get("model_info") or {}
        reported = [
            value
            for key, value in model_info.items()
            if key.endswith(".embedding_length") and isinstance(value, int)
        ]
        if reported and reported[0] != self._dimension:
            logger.error(
                "Embedding model %s produces %d dimensions but EASYRAG_EMBEDDING_DIMENSION=%d",
                self._model,
                reported[0],
                self._dimension,
            )
            return "misconfigured"
        return "ok"

    async def _embed(self, inputs: list[str]) -> list[Vector]:
        # truncate=False: fail loudly instead of silently embedding only part of a chunk.
        payload = {"model": self._model, "input": inputs, "truncate": False}
        data = await self._post("/api/embed", payload)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingResponseError("Ollama response has no 'embeddings' list")
        return validate_embeddings(
            embeddings, expected_count=len(inputs), dimension=self._dimension
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        error: EmbeddingError = EmbeddingProviderUnavailableError("Ollama request was not sent")
        for attempt in range(self._max_retries + 1):
            if attempt:
                await asyncio.sleep(self._retry_backoff_seconds * 2 ** (attempt - 1))
            try:
                response = await self._client.post(path, json=payload)
            except httpx2.TimeoutException:
                error = EmbeddingProviderUnavailableError("Ollama request timed out")
                continue
            except httpx2.TransportError as exc:
                error = EmbeddingProviderUnavailableError(
                    f"cannot reach Ollama at {self._client.base_url} ({type(exc).__name__})"
                )
                continue

            if response.status_code == 200:
                return self._json(response)
            message = _error_message(response)
            if response.status_code == 404:
                raise EmbeddingProviderUnavailableError(
                    f"embedding model {self._model!r} is not available in Ollama ({message}); "
                    f"pull it with: ollama pull {self._model}"
                )
            if response.status_code in _RETRYABLE_STATUS:
                error = EmbeddingProviderUnavailableError(
                    f"Ollama returned HTTP {response.status_code}: {message}"
                )
                continue
            raise EmbeddingResponseError(
                f"Ollama rejected the request (HTTP {response.status_code}): {message}"
            )

        logger.warning("Ollama embedding failed after %d attempts: %s", attempt + 1, error)
        raise error

    @staticmethod
    def _json(response: httpx2.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingResponseError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise EmbeddingResponseError("Ollama returned an unexpected JSON document")
        return data


def _error_message(response: httpx2.Response) -> str:
    """Ollama reports errors as {"error": "..."}; fall back to the raw body."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return str(data["error"])
    return response.text[:200]
