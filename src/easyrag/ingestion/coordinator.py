"""Runs indexing in the background of the API process and remembers how it went.

- ``start_until_success()``: used at startup. Retries with exponential backoff while
  dependencies are not ready yet (Ollama still pulling models, database starting, docs
  directory not mounted), and gives up only on an incompatible index, which needs a human.
- ``trigger()``: one run on request; refuses while a run is in progress.
- ``status()``: a snapshot for the API.

Only one run executes at a time in this process; across processes the store's writer
lock serialises runs (ARCHITECTURE D10).
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from easyrag.domain.errors import IndexIncompatibleError
from easyrag.ingestion.indexer import IndexReport

logger = logging.getLogger(__name__)

RunState = Literal["idle", "running", "succeeded", "failed"]
Sleep = Callable[[float], Awaitable[None]]


class IndexRunner(Protocol):
    async def run(self, *, rebuild: bool = False) -> IndexReport: ...


class IndexRunInProgressError(Exception):
    """An indexing run is already in progress."""


@dataclass(frozen=True, slots=True)
class IndexStatus:
    state: RunState
    started_at: datetime | None
    finished_at: datetime | None
    last_report: IndexReport | None
    last_error: str | None
    # Startup only: consecutive failed attempts while waiting for dependencies.
    attempts: int


class IndexingCoordinator:
    def __init__(
        self,
        indexer: IndexRunner,
        *,
        initial_retry_seconds: float = 5.0,
        max_retry_seconds: float = 60.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._indexer = indexer
        self._initial_retry_seconds = initial_retry_seconds
        self._max_retry_seconds = max_retry_seconds
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._state: RunState = "idle"
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._last_report: IndexReport | None = None
        self._last_error: str | None = None
        self._attempts = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> IndexStatus:
        return IndexStatus(
            state=self._state,
            started_at=self._started_at,
            finished_at=self._finished_at,
            last_report=self._last_report,
            last_error=self._last_error,
            attempts=self._attempts,
        )

    def start_until_success(self) -> None:
        """Index in the background, retrying until one run succeeds (used at startup)."""
        self._start(self._until_success())

    def trigger(self) -> None:
        """Start one background run. Raises ``IndexRunInProgressError`` if one is running."""
        self._start(self._once())

    async def wait(self) -> None:
        """Wait for the current background run, if any (used by tests and shutdown)."""
        if self._task is not None:
            await asyncio.shield(self._task)

    async def aclose(self) -> None:
        if self.running and self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _start(self, work: Coroutine[Any, Any, None]) -> None:
        if self.running:
            work.close()  # never awaited: avoid a "coroutine was never awaited" warning
            raise IndexRunInProgressError("an indexing run is already in progress")
        self._task = asyncio.create_task(work, name="easyrag-indexing")

    async def _once(self) -> None:
        with contextlib.suppress(Exception):  # recorded in the status by _run
            await self._run()

    async def _until_success(self) -> None:
        delay = self._initial_retry_seconds
        self._attempts = 0
        while True:
            try:
                await self._run()
                return
            except IndexIncompatibleError:
                logger.error("Startup indexing stopped: the index must be rebuilt")
                return
            except Exception:
                self._attempts += 1
                logger.warning(
                    "Startup indexing attempt %d failed; retrying in %.0fs",
                    self._attempts,
                    delay,
                )
                await self._sleep(delay)
                delay = min(delay * 2, self._max_retry_seconds)

    async def _run(self) -> None:
        self._state = "running"
        self._started_at = datetime.now(UTC)
        self._finished_at = None
        try:
            report = await self._indexer.run()
        except asyncio.CancelledError:
            self._state, self._last_error = "failed", "cancelled"
            raise
        except Exception as exc:
            self._state, self._last_error = "failed", f"{type(exc).__name__}: {exc}"
            self._finished_at = datetime.now(UTC)
            logger.warning("Indexing failed: %s", self._last_error)
            raise
        self._state, self._last_error, self._last_report = "succeeded", None, report
        self._finished_at = datetime.now(UTC)
