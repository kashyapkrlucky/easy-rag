"""Cross-encoder reranker running an ONNX model on CPU via fastembed (ARCHITECTURE D33).

The model is downloaded once into ``cache_dir`` (a persistent volume in Docker) and then
used offline. Loading happens in the background at startup; until it finishes, reranking
raises ``RerankerUnavailableError`` and the caller keeps the retrieval order. Inference is
CPU-bound, so it runs in a worker thread instead of blocking the event loop.
"""

import asyncio
import contextlib
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from easyrag.domain.chunks import RetrievedChunk
from easyrag.health import CheckStatus
from easyrag.retrieval.rerank import RerankerUnavailableError

if TYPE_CHECKING:  # imported lazily at runtime: loading onnxruntime is slow
    from fastembed.rerank.cross_encoder import TextCrossEncoder

logger = logging.getLogger(__name__)

LoadState = Literal["not_started", "loading", "ready", "failed"]


class FastEmbedReranker:
    def __init__(
        self, model_name: str, cache_dir: Path, *, batch_size: int = 32, threads: int | None = None
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._threads = threads
        self._encoder: TextCrossEncoder | None = None
        self._state: LoadState = "not_started"
        self._error: str | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def model_id(self) -> str:
        return f"fastembed/{self._model_name}"

    @property
    def state(self) -> LoadState:
        return self._state

    def start_loading(self) -> None:
        """Download (first time only) and load the model in the background."""
        if self._task is None:
            self._task = asyncio.create_task(self.load(), name="easyrag-reranker-load")

    async def load(self) -> None:
        if self._state in ("loading", "ready"):
            return
        self._state = "loading"
        try:
            self._encoder = await asyncio.to_thread(self._create_encoder)
        except Exception as exc:
            self._state, self._error = "failed", f"{type(exc).__name__}: {exc}"
            logger.error("Could not load reranker %s: %s", self._model_name, self._error)
            return
        self._state = "ready"
        logger.info("Reranker %s loaded", self._model_name)

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def readiness(self) -> CheckStatus:
        if self._state == "ready":
            return "ok"
        return "unavailable" if self._state == "failed" else "missing"

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if self._encoder is None:
            raise RerankerUnavailableError(f"reranker model is {self._state.replace('_', ' ')}")
        documents = [_document_text(chunk) for chunk in chunks]
        try:
            logits = await asyncio.to_thread(self._score, query, documents)
        except Exception as exc:  # inference failure must not fail the request
            raise RerankerUnavailableError(f"{type(exc).__name__}: {exc}") from exc

        scored = [
            replace(chunk, score=probability, rerank_score=probability)
            for chunk, probability in zip(chunks, (_sigmoid(x) for x in logits), strict=True)
        ]
        scored.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return scored[:top_n]

    def _create_encoder(self) -> "TextCrossEncoder":
        # Local-first: no usage telemetry from the model libraries, in or out of Docker.
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import onnxruntime
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        onnxruntime.disable_telemetry_events()

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return TextCrossEncoder(
            model_name=self._model_name, cache_dir=str(self._cache_dir), threads=self._threads
        )

    def _score(self, query: str, documents: list[str]) -> list[float]:
        encoder = self._encoder
        assert encoder is not None
        return [float(x) for x in encoder.rerank(query, documents, batch_size=self._batch_size)]


def _document_text(chunk: RetrievedChunk) -> str:
    """The same contextual header the chunk was embedded with, then its content."""
    path = list(chunk.heading_path)
    if not path or path[0] != chunk.document_title:
        path.insert(0, chunk.document_title)
    return " > ".join(path) + "\n\n" + chunk.content


def _sigmoid(logit: float) -> float:
    return 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
