import asyncio

import pytest

from easyrag.domain.errors import IndexIncompatibleError
from easyrag.embeddings.base import EmbeddingProviderUnavailableError
from easyrag.ingestion.coordinator import IndexingCoordinator, IndexRunInProgressError
from easyrag.ingestion.indexer import IndexReport

REPORT = IndexReport(documents_added=2, chunks_embedded=5)
INCOMPATIBLE = IndexIncompatibleError(
    stored_model="a", stored_dimension=8, configured_model="b", configured_dimension=8
)


class ScriptedIndexer:
    """Fails with the scripted exceptions in order, then succeeds."""

    def __init__(self, *failures: Exception, gate: asyncio.Event | None = None) -> None:
        self.failures = list(failures)
        self.calls = 0
        self.gate = gate

    async def run(self, *, rebuild: bool = False) -> IndexReport:
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        if self.failures:
            raise self.failures.pop(0)
        return REPORT


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def coordinator(
    indexer: ScriptedIndexer, sleep: RecordingSleep | None = None
) -> IndexingCoordinator:
    return IndexingCoordinator(
        indexer, initial_retry_seconds=5, max_retry_seconds=20, sleep=sleep or RecordingSleep()
    )


async def test_initial_status_is_idle() -> None:
    status = coordinator(ScriptedIndexer()).status()

    assert (status.state, status.last_report, status.last_error) == ("idle", None, None)


async def test_successful_run_is_recorded() -> None:
    subject = coordinator(ScriptedIndexer())

    subject.trigger()
    await subject.wait()

    status = subject.status()
    assert status.state == "succeeded"
    assert status.last_report == REPORT
    assert status.started_at is not None and status.finished_at is not None
    assert status.started_at <= status.finished_at


async def test_failed_run_is_recorded_without_raising() -> None:
    subject = coordinator(ScriptedIndexer(EmbeddingProviderUnavailableError("Ollama is down")))

    subject.trigger()
    await subject.wait()

    status = subject.status()
    assert status.state == "failed"
    assert status.last_error == "EmbeddingProviderUnavailableError: Ollama is down"


async def test_a_second_trigger_while_running_is_refused() -> None:
    gate = asyncio.Event()
    indexer = ScriptedIndexer(gate=gate)
    subject = coordinator(indexer)

    subject.trigger()
    await asyncio.sleep(0)
    assert subject.status().state == "running"
    with pytest.raises(IndexRunInProgressError):
        subject.trigger()

    gate.set()
    await subject.wait()
    assert indexer.calls == 1


async def test_startup_retries_with_exponential_backoff_until_success() -> None:
    sleep = RecordingSleep()
    down = EmbeddingProviderUnavailableError("model still downloading")
    indexer = ScriptedIndexer(down, down, down, NotADirectoryError("docs"))
    subject = coordinator(indexer, sleep)

    subject.start_until_success()
    await subject.wait()

    assert indexer.calls == 5
    assert sleep.delays == [5, 10, 20, 20]  # doubles, capped at max_retry_seconds
    assert subject.status().state == "succeeded"
    assert subject.status().attempts == 4


async def test_startup_stops_retrying_on_an_incompatible_index() -> None:
    sleep = RecordingSleep()
    indexer = ScriptedIndexer(INCOMPATIBLE)
    subject = coordinator(indexer, sleep)

    subject.start_until_success()
    await subject.wait()

    assert indexer.calls == 1
    assert sleep.delays == []
    status = subject.status()
    assert status.state == "failed"
    assert status.last_error is not None and "easyrag index --rebuild" in status.last_error


async def test_close_cancels_a_running_task() -> None:
    gate = asyncio.Event()  # never set: the run would block forever
    subject = coordinator(ScriptedIndexer(gate=gate))
    subject.trigger()
    await asyncio.sleep(0)

    await subject.aclose()

    assert not subject.running
    assert subject.status().last_error == "cancelled"
