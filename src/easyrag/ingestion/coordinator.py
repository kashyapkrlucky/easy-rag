"""Runs indexing in the background of the API process and remembers how it went.

- ``start_until_success()``: used at startup. Retries with exponential backoff while
  dependencies are not ready yet (Ollama still pulling models, database starting, docs
  directory not mounted), and gives up only on an incompatible index, which needs a human.
- ``trigger()``: one run on request; refuses while a run is in progress (HTTP 409).
- ``request_run()``: coalescing run request (file watcher, periodic resync). If a run is
  in progress, exactly one follow-up run is scheduled after it, however many requests
  arrive meanwhile, so changes made during a run are never lost.
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
from easyrag.observability.context import bound, index_run_id, new_id

logger = logging.getLogger(__name__)

RunState = Literal["idle", "running", "succeeded", "failed"]
Sleep = Callable[[float], Awaitable[None]]
# Observers (e.g. metrics); they must not affect indexing, so their errors are logged only.
OnReport = Callable[[IndexReport], Awaitable[None]]
OnFailure = Callable[[Exception], None]


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
    # A follow-up run has been requested while a run was in progress.
    pending: bool


class IndexingCoordinator:
    def __init__(
        self,
        indexer: IndexRunner,
        *,
        initial_retry_seconds: float = 5.0,
        max_retry_seconds: float = 60.0,
        sleep: Sleep = asyncio.sleep,
        on_report: OnReport | None = None,
        on_failure: OnFailure | None = None,
    ) -> None:
        self._indexer = indexer
        self._on_report = on_report
        self._on_failure = on_failure
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
        self._pending = False

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
            pending=self._pending,
        )

    def start_until_success(self) -> None:
        """Index in the background, retrying until one run succeeds (used at startup)."""
        self._start(self._until_success())

    def trigger(self) -> None:
        """Start one background run. Raises ``IndexRunInProgressError`` if one is running."""
        self._start(self._once())

    def request_run(self) -> None:
        """Run now, or once more after the current run (coalescing; never raises)."""
        if self.running:
            self._pending = True
            return
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
        await self._drain_pending()

    async def _drain_pending(self) -> None:
        while self._pending:
            self._pending = False
            with contextlib.suppress(Exception):
                await self._run()

    async def _until_success(self) -> None:
        delay = self._initial_retry_seconds
        self._attempts = 0
        while True:
            try:
                await self._run()
                break
            except IndexIncompatibleError:
                logger.error("Startup indexing stopped: the index must be rebuilt")
                break
            except Exception:
                self._attempts += 1
                logger.warning(
                    "Startup indexing attempt %d failed; retrying in %.0fs",
                    self._attempts,
                    delay,
                )
                await self._sleep(delay)
                delay = min(delay * 2, self._max_retry_seconds)
        await self._drain_pending()

    async def _run(self) -> None:
        self._state = "running"
        self._started_at = datetime.now(UTC)
        self._finished_at = None
        with bound(index_run_id, new_id()):  # correlates every log line of this run
            try:
                report = await self._indexer.run()
            except asyncio.CancelledError:
                self._state, self._last_error = "failed", "cancelled"
                raise
            except Exception as exc:
                self._state, self._last_error = "failed", f"{type(exc).__name__}: {exc}"
                self._finished_at = datetime.now(UTC)
                logger.warning("Indexing failed: %s", self._last_error)
                self._notify_failure(exc)
                raise
            self._state, self._last_error, self._last_report = "succeeded", None, report
            self._finished_at = datetime.now(UTC)
            await self._notify_report(report)

    async def _notify_report(self, report: IndexReport) -> None:
        if self._on_report is not None:
            try:
                await self._on_report(report)
            except Exception:
                logger.exception("Index report observer failed")

    def _notify_failure(self, exc: Exception) -> None:
        if self._on_failure is not None:
            try:
                self._on_failure(exc)
            except Exception:
                logger.exception("Index failure observer failed")
