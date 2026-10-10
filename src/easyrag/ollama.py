"""Shared Ollama HTTP plumbing for the embedding and LLM adapters.

Raises neutral errors; each adapter translates them into its own port's errors, so
the rest of the application never sees Ollama-specific exceptions.

Retry policy: connection errors, timeouts, 429 and 5xx are retried with exponential
backoff; 404 (model not pulled) and other 4xx responses fail immediately.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2

logger = logging.getLogger(__name__)

# Rate limiting, overload (Ollama's queue is full) and server errors are worth retrying.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class OllamaError(Exception):
    """Base class for Ollama transport and protocol errors."""


class OllamaUnavailableError(OllamaError):
    """Ollama could not be reached, timed out, or kept failing with retryable errors."""


class OllamaModelNotFoundError(OllamaError):
    """The requested model has not been pulled."""

    def __init__(self, model: str, message: str) -> None:
        super().__init__(f"model {model!r} is not available in Ollama ({message})")
        self.model = model


class OllamaRequestError(OllamaError):
    """Ollama rejected the request (non-retryable) or returned an unusable response."""


async def post_json(
    client: httpx2.AsyncClient,
    path: str,
    payload: dict[str, Any],
    *,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.5,
) -> dict[str, Any]:
    """POST ``payload`` and return the JSON object response, applying the retry policy."""
    model = str(payload.get("model", ""))
    error: OllamaError = OllamaUnavailableError("Ollama request was not sent")
    attempt = 0
    for attempt in range(max_retries + 1):
        if attempt:
            await asyncio.sleep(retry_backoff_seconds * 2 ** (attempt - 1))
        try:
            response = await client.post(path, json=payload)
        except httpx2.TimeoutException:
            error = OllamaUnavailableError("Ollama request timed out")
            continue
        except httpx2.TransportError as exc:
            error = OllamaUnavailableError(
                f"cannot reach Ollama at {client.base_url} ({type(exc).__name__})"
            )
            continue

        if response.status_code == 200:
            return _json_object(response)
        message = error_message(response)
        if response.status_code == 404:
            raise OllamaModelNotFoundError(model, message)
        if response.status_code in _RETRYABLE_STATUS:
            error = OllamaUnavailableError(
                f"Ollama returned HTTP {response.status_code}: {message}"
            )
            continue
        raise OllamaRequestError(
            f"Ollama rejected the request (HTTP {response.status_code}): {message}"
        )

    logger.warning("Ollama %s failed after %d attempts: %s", path, attempt + 1, error)
    raise error


async def show_model(client: httpx2.AsyncClient, model: str) -> dict[str, Any] | None:
    """Model metadata from ``/api/show``; ``None`` if the model is not pulled.

    No retries: used by readiness checks, which have their own timeout.
    """
    response = await client.post("/api/show", json={"model": model})
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return _json_object(response)


@dataclass(frozen=True, slots=True)
class PullProgress:
    status: str
    # Bytes over all layers seen so far; total is 0 until Ollama reports sizes.
    completed: int
    total: int


async def pull_model(
    client: httpx2.AsyncClient,
    model: str,
    on_progress: Callable[[PullProgress], None] | None = None,
) -> None:
    """Download ``model`` into Ollama (``/api/pull``, streamed), reporting progress.

    Raises ``OllamaUnavailableError`` when Ollama can't be reached, and
    ``OllamaRequestError`` when the pull fails (unknown model, no internet access, ...).
    """
    layers: dict[str, tuple[int, int]] = {}  # digest -> (completed, total)
    try:
        async with client.stream("POST", "/api/pull", json={"model": model}) as response:
            if response.status_code != 200:
                await response.aread()
                raise OllamaRequestError(
                    f"pulling {model!r} failed (HTTP {response.status_code}): "
                    f"{error_message(response)}"
                )
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError as exc:
                    raise OllamaRequestError("Ollama returned invalid pull progress") from exc
                if not isinstance(event, dict):
                    continue
                if isinstance(event.get("error"), str):
                    raise OllamaRequestError(f"pulling {model!r} failed: {event['error']}")
                digest, total = event.get("digest"), event.get("total")
                if isinstance(digest, str) and isinstance(total, int):
                    layers[digest] = (int(event.get("completed") or 0), total)
                status = str(event.get("status", ""))
                if on_progress is not None:
                    on_progress(
                        PullProgress(
                            status=status,
                            completed=sum(done for done, _ in layers.values()),
                            total=sum(size for _, size in layers.values()),
                        )
                    )
                if status == "success":
                    return
    except httpx2.TimeoutException as exc:
        raise OllamaUnavailableError(f"pulling {model!r} timed out") from exc
    except httpx2.TransportError as exc:
        raise OllamaUnavailableError(
            f"cannot reach Ollama at {client.base_url} ({type(exc).__name__})"
        ) from exc
    raise OllamaRequestError(f"pulling {model!r} ended without success")


def model_info_value(metadata: dict[str, Any], suffix: str) -> int | None:
    """An integer from ``model_info`` whose key ends with ``suffix`` (e.g. ".context_length").

    Keys are prefixed with the architecture ("llama.context_length", "qwen3.…").
    """
    info = metadata.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(suffix) and isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def error_message(response: httpx2.Response) -> str:
    """Ollama reports errors as {"error": "..."}; fall back to the raw body."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return str(data["error"])
    return response.text[:200]


def _json_object(response: httpx2.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaRequestError("Ollama returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise OllamaRequestError("Ollama returned an unexpected JSON document")
    return data
