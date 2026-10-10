"""Make sure the configured Ollama models are present, pulling missing ones.

Runs in the background at startup (``EASYRAG_OLLAMA_PULL_MODELS``), so ``docker compose up``
needs no separate download step and works the same with the bundled or a host Ollama.
Ollama may still be starting, or the machine may be offline: attempts are retried with
backoff until every model is present. Readiness reports a model being pulled as "pulling".
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Sequence

import httpx2

from easyrag.ollama import OllamaError, PullProgress, pull_model, show_model

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]

# Log progress at most this often, so a large download doesn't flood the logs.
_PROGRESS_LOG_SECONDS = 10.0


class ModelPuller:
    def __init__(
        self,
        client: httpx2.AsyncClient,
        models: Sequence[str],
        *,
        initial_retry_seconds: float = 5.0,
        max_retry_seconds: float = 300.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._client = client
        self._models = tuple(dict.fromkeys(models))  # same model for both roles: pull once
        self._initial_retry_seconds = initial_retry_seconds
        self._max_retry_seconds = max_retry_seconds
        self._sleep = sleep
        self._pulling: str | None = None
        self._task: asyncio.Task[None] | None = None

    def is_pulling(self, model: str) -> bool:
        return self._pulling == model

    async def ensure(self, model: str) -> bool:
        """Pull ``model`` if it is missing. Returns True if it was pulled.

        Raises ``OllamaError`` (unreachable, pull failed) and ``httpx2`` errors from the
        presence check.
        """
        if await show_model(self._client, model) is not None:
            return False
        logger.info("Pulling model %s (first start, or the model was changed)", model)
        self._pulling = model
        started = time.monotonic()
        last_log = [started]

        def progress(update: PullProgress) -> None:
            now = time.monotonic()
            if update.total and now - last_log[0] >= _PROGRESS_LOG_SECONDS:
                last_log[0] = now
                logger.info(
                    "Pulling %s: %.0f%% of %.0f MB",
                    model,
                    100 * update.completed / update.total,
                    update.total / 1e6,
                )

        try:
            await pull_model(self._client, model, progress)
        finally:
            self._pulling = None
        logger.info("Pulled model %s in %.0fs", model, time.monotonic() - started)
        return True

    async def ensure_all(self) -> None:
        """One attempt for every model, in order (the CLI); raises on the first failure."""
        for model in self._models:
            await self.ensure(model)

    def start(self) -> None:
        """Pull missing models in the background, retrying until all are present."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._until_present(), name="easyrag-model-pull")

    async def wait(self) -> None:
        if self._task is not None:
            await asyncio.shield(self._task)

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _until_present(self) -> None:
        remaining = list(self._models)
        delay = self._initial_retry_seconds
        while remaining:
            try:
                await self.ensure(remaining[0])
            except (OllamaError, httpx2.HTTPError) as exc:
                logger.warning(
                    "Model %s is not available yet (%s); retrying in %.0fs",
                    remaining[0],
                    exc,
                    delay,
                )
                await self._sleep(delay)
                delay = min(delay * 2, self._max_retry_seconds)
                continue
            remaining.pop(0)
            delay = self._initial_retry_seconds
