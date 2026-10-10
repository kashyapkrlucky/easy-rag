import asyncio

import pytest

from easyrag.embeddings.base import EmbeddingProviderUnavailableError
from easyrag.ingestion.coordinator import IndexingCoordinator
from easyrag.ingestion.indexer import FailedDocument, IndexReport
from easyrag.observability.context import index_run_id
from easyrag.observability.metrics import Metrics

REPORT = IndexReport(
    documents_added=2,
    documents_updated=1,
    documents_unchanged=7,
    documents_deleted=1,
    chunks_embedded=5,
    chunks_reused=3,
    failed=(FailedDocument("bad.md", "x"),),
    duration_seconds=2.5,
)


def sample(metrics: Metrics, name: str, **labels: str) -> float | None:
    return metrics.registry.get_sample_value(name, labels)


def test_index_run_is_recorded() -> None:
    metrics = Metrics()

    metrics.record_index_run(REPORT)
    metrics.record_index_size(10, 42)

    assert sample(metrics, "easyrag_index_runs_total", result="succeeded") == 1
    assert sample(metrics, "easyrag_index_documents_total", change="added") == 2
    assert sample(metrics, "easyrag_index_documents_total", change="failed") == 1
    assert sample(metrics, "easyrag_index_chunks_total", source="embedded") == 5
    assert sample(metrics, "easyrag_index_chunks_total", source="reused") == 3
    assert sample(metrics, "easyrag_index_run_duration_seconds_sum") == 2.5
    assert sample(metrics, "easyrag_indexed_chunks") == 42
    assert (sample(metrics, "easyrag_index_last_success_timestamp_seconds") or 0) > 0


def test_registries_are_independent() -> None:
    first, second = Metrics(), Metrics()

    first.record_index_failure()

    assert sample(first, "easyrag_index_runs_total", result="failed") == 1
    assert sample(second, "easyrag_index_runs_total", result="failed") is None


class Indexer:
    def __init__(self, result: IndexReport | Exception) -> None:
        self.result = result
        self.run_ids: list[str | None] = []

    async def run(self, *, rebuild: bool = False) -> IndexReport:
        self.run_ids.append(index_run_id.get())
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_coordinator_reports_runs_and_binds_an_index_run_id() -> None:
    reports: list[IndexReport] = []

    async def on_report(report: IndexReport) -> None:
        reports.append(report)

    indexer = Indexer(REPORT)
    coordinator = IndexingCoordinator(indexer, on_report=on_report)

    coordinator.trigger()
    await coordinator.wait()
    coordinator.trigger()
    await coordinator.wait()

    assert reports == [REPORT, REPORT]
    assert all(indexer.run_ids) and len(set(indexer.run_ids)) == 2  # a new ID per run
    assert index_run_id.get() is None  # not leaked outside the run


async def test_coordinator_reports_failures() -> None:
    failures: list[Exception] = []
    coordinator = IndexingCoordinator(
        Indexer(EmbeddingProviderUnavailableError("down")), on_failure=failures.append
    )

    coordinator.trigger()
    await coordinator.wait()

    assert [type(f) for f in failures] == [EmbeddingProviderUnavailableError]


async def test_a_failing_observer_does_not_break_indexing(caplog: pytest.LogCaptureFixture) -> None:
    async def broken(report: IndexReport) -> None:
        raise RuntimeError("metrics backend exploded")

    coordinator = IndexingCoordinator(Indexer(REPORT), on_report=broken)

    coordinator.trigger()
    await coordinator.wait()
    await asyncio.sleep(0)

    assert coordinator.status().state == "succeeded"
    assert "Index report observer failed" in caplog.text
