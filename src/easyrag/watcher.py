"""Watch the docs directory and request incremental index runs (ARCHITECTURE D20).

An adapter (filesystem events via ``watchfiles``), so it lives outside the pipeline
packages, which must not depend on infrastructure libraries.

The watcher never indexes by itself: it calls ``on_change`` (the coordinator's coalescing
``request_run``), so bursts of file events collapse into at most one running and one
pending run. Three mechanisms feed it:

1. native filesystem events via ``watchfiles`` (debounced; polling can be forced for
   bind mounts or network filesystems that do not deliver events), plus one catch-up
   request whenever watching (re)starts: a change made before the backend has taken its
   initial snapshot would otherwise never be reported;
2. a periodic resync request, as a safety net for anything the events missed;
3. waiting for a docs directory that does not exist yet (mounts can appear late).
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from pathlib import Path, PurePath
from typing import Literal

from watchfiles import Change, awatch

from easyrag.ingestion.loader import MARKDOWN_SUFFIXES

logger = logging.getLogger(__name__)

WatcherState = Literal["disabled", "starting", "watching", "waiting_for_directory", "stopped"]


class MarkdownChangeFilter:
    """Pass only changes to Markdown files that the loader would read (not hidden ones)."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def __call__(self, change: Change, path: str) -> bool:
        candidate = PurePath(path)
        if candidate.suffix.lower() not in MARKDOWN_SUFFIXES:
            return False
        try:
            relative = candidate.relative_to(self._root)
        except ValueError:
            relative = candidate
        return not any(part.startswith(".") for part in relative.parts)


class DocsWatcher:
    def __init__(
        self,
        root: Path,
        on_change: Callable[[], None],
        *,
        debounce_ms: int = 1000,
        force_polling: bool = False,
        poll_delay_ms: int = 1000,
        resync_seconds: float = 300.0,
        missing_directory_retry_seconds: float = 10.0,
    ) -> None:
        self._root = root
        self._on_change = on_change
        self._debounce_ms = debounce_ms
        self._force_polling = force_polling
        self._poll_delay_ms = poll_delay_ms
        self._resync_seconds = resync_seconds
        self._missing_directory_retry_seconds = missing_directory_retry_seconds
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._state: WatcherState = "starting"

    @property
    def state(self) -> WatcherState:
        return self._state

    def start(self) -> None:
        self._stop.clear()
        self._tasks = [asyncio.create_task(self._watch(), name="easyrag-watcher")]
        if self._resync_seconds > 0:
            self._tasks.append(asyncio.create_task(self._resync(), name="easyrag-resync"))

    async def aclose(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        self._state = "stopped"

    async def _watch(self) -> None:
        while not self._stop.is_set():
            if not self._root.is_dir():
                if self._state != "waiting_for_directory":
                    logger.warning("Docs directory %s does not exist yet; waiting", self._root)
                self._state = "waiting_for_directory"
                await self._sleep_or_stop(self._missing_directory_retry_seconds)
                continue

            self._state = "watching"
            self._on_change()  # catch-up: covers changes made while (re)starting the watch
            logger.info(
                "Watching %s for changes (%s)",
                self._root,
                "polling" if self._force_polling else "native events",
            )
            try:
                async for changes in awatch(
                    self._root,
                    watch_filter=MarkdownChangeFilter(self._root),
                    debounce=self._debounce_ms,
                    stop_event=self._stop,
                    force_polling=self._force_polling,
                    poll_delay_ms=self._poll_delay_ms,
                    ignore_permission_denied=True,
                ):
                    logger.info("Detected %d document change(s); requesting re-index", len(changes))
                    self._on_change()
            except (FileNotFoundError, RuntimeError) as exc:
                # The directory vanished or the backend failed: wait, then watch again.
                logger.warning("Watching %s failed (%s); retrying", self._root, exc)
                await self._sleep_or_stop(self._missing_directory_retry_seconds)
            else:
                if not self._stop.is_set():  # the directory was removed while watching
                    await self._sleep_or_stop(self._missing_directory_retry_seconds)

    async def _resync(self) -> None:
        while not await self._sleep_or_stop(self._resync_seconds):
            logger.debug("Periodic resync: requesting re-index")
            self._on_change()

    async def _sleep_or_stop(self, seconds: float) -> bool:
        """Sleep, waking early on stop. Returns True if the watcher is stopping."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        return self._stop.is_set()
