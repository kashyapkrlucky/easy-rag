import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from watchfiles import Change

from easyrag.watcher import DocsWatcher, MarkdownChangeFilter

TIMEOUT = 10.0  # generous: CI filesystems can be slow


class Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


async def wait_until(condition: Callable[[], bool]) -> None:
    try:
        async with asyncio.timeout(TIMEOUT):
            # The condition is changed by watcher callbacks; there is no event to await.
            while not condition():  # noqa: ASYNC110
                await asyncio.sleep(0.05)
    except TimeoutError:
        raise AssertionError("condition not met in time") from None


@pytest.fixture(params=[False, True], ids=["native-events", "polling"])
def force_polling(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture
async def watched(
    tmp_path: Path, force_polling: bool
) -> AsyncIterator[tuple[Path, Counter, DocsWatcher]]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "existing.md").write_text("# Existing\n")
    counter = Counter()
    watcher = DocsWatcher(
        docs, counter, debounce_ms=200, force_polling=force_polling, poll_delay_ms=50,
        resync_seconds=0,
    )  # fmt: skip
    watcher.start()
    await wait_until(lambda: watcher.state == "watching")
    # Let the backend take its initial snapshot, then ignore the catch-up request and any
    # historical events (macOS FSEvents can report the fixture's own writes).
    await asyncio.sleep(1.0)
    counter.calls = 0
    try:
        yield docs, counter, watcher
    finally:
        await watcher.aclose()


async def test_creating_a_markdown_file_requests_a_run(
    watched: tuple[Path, Counter, DocsWatcher],
) -> None:
    docs, counter, _ = watched

    (docs / "new.md").write_text("# New\n")

    await wait_until(lambda: counter.calls >= 1)


async def test_modifying_and_deleting_request_runs(
    watched: tuple[Path, Counter, DocsWatcher],
) -> None:
    docs, counter, _ = watched

    (docs / "existing.md").write_text("# Existing\n\nEdited.\n")
    await wait_until(lambda: counter.calls >= 1)
    calls = counter.calls
    (docs / "existing.md").unlink()

    await wait_until(lambda: counter.calls > calls)


async def test_a_burst_of_writes_is_debounced(watched: tuple[Path, Counter, DocsWatcher]) -> None:
    docs, counter, _ = watched

    for i in range(20):
        (docs / f"bulk-{i}.md").write_text(f"# Bulk {i}\n")
    await wait_until(lambda: counter.calls >= 1)
    await asyncio.sleep(0.6)  # longer than the debounce window

    assert counter.calls <= 3  # 20 files → a handful of batches at most, not 20 requests


async def test_irrelevant_changes_are_ignored(watched: tuple[Path, Counter, DocsWatcher]) -> None:
    docs, counter, _ = watched

    (docs / "notes.txt").write_text("not markdown")
    (docs / ".hidden.md").write_text("# hidden")
    (docs / ".obsidian").mkdir()
    (docs / ".obsidian" / "workspace.md").write_text("# editor state")
    await asyncio.sleep(1.0)

    assert counter.calls == 0


async def test_watching_starts_with_a_catch_up_request(tmp_path: Path) -> None:
    counter = Counter()
    watcher = DocsWatcher(tmp_path, counter, resync_seconds=0)
    watcher.start()
    try:
        await wait_until(lambda: counter.calls >= 1)
    finally:
        await watcher.aclose()


async def test_periodic_resync_requests_runs_without_events(tmp_path: Path) -> None:
    counter = Counter()
    watcher = DocsWatcher(tmp_path, counter, resync_seconds=0.1)
    watcher.start()
    try:
        await wait_until(lambda: counter.calls >= 3)  # catch-up + at least two resyncs
    finally:
        await watcher.aclose()


async def test_waits_for_a_missing_directory_then_watches_it(tmp_path: Path) -> None:
    docs = tmp_path / "late-mount"
    counter = Counter()
    watcher = DocsWatcher(
        docs, counter, debounce_ms=100, resync_seconds=0, missing_directory_retry_seconds=0.1
    )
    watcher.start()
    try:
        await wait_until(lambda: watcher.state == "waiting_for_directory")
        docs.mkdir()
        await wait_until(lambda: watcher.state == "watching")
        await wait_until(lambda: counter.calls >= 1)  # catch-up once the directory exists
        await asyncio.sleep(1.0)
        calls = counter.calls
        (docs / "first.md").write_text("# First\n")
        await wait_until(lambda: counter.calls > calls)
    finally:
        await watcher.aclose()


async def test_close_stops_promptly_and_cleanly(tmp_path: Path) -> None:
    watcher = DocsWatcher(tmp_path, Counter(), resync_seconds=3600)
    watcher.start()
    await wait_until(lambda: watcher.state == "watching")

    started = time.monotonic()
    await watcher.aclose()

    assert time.monotonic() - started < 2.0
    assert watcher.state == "stopped"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/docs/guide.md", True),
        ("/docs/sub/API.MARKDOWN", True),
        ("/docs/notes.txt", False),
        ("/docs/.hidden.md", False),
        ("/docs/.git/HEAD.md", False),
        ("/docs/sub/.obsidian/x.md", False),
    ],
)
def test_filter_mirrors_the_loader_rules(path: str, expected: bool) -> None:
    assert MarkdownChangeFilter(Path("/docs"))(Change.modified, path) is expected
