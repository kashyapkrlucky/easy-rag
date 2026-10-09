import asyncio
from dataclasses import replace
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from easyrag.config import Settings
from easyrag.container import AppContainer, build_container
from easyrag.ingestion.coordinator import IndexingCoordinator
from easyrag.ingestion.indexer import FailedDocument, IndexReport
from easyrag.main import create_app
from easyrag.storage.index_store import IndexedDocument
from tests.support import SettingsFactory

REPORT = IndexReport(
    documents_added=1,
    documents_unchanged=4,
    chunks_embedded=3,
    chunks_reused=2,
    failed=(FailedDocument("bad.md", "input too long"),),
    duration_seconds=1.23456,
)
DOCUMENT = IndexedDocument(
    "guides/deployment.md", "Deployment", "a" * 64, 1234, 3, datetime(2026, 10, 10, tzinfo=UTC)
)


class StubIndexer:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.calls = 0

    async def run(self, *, rebuild: bool = False) -> IndexReport:
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        return REPORT


class StubStore:
    async def counts(self) -> tuple[int, int]:
        return 5, 12

    async def list_documents(self) -> list[IndexedDocument]:
        return [DOCUMENT]


def client_with(settings: Settings, indexer: StubIndexer) -> tuple[TestClient, IndexingCoordinator]:
    coordinator = IndexingCoordinator(indexer)

    def factory(settings: Settings) -> AppContainer:
        return replace(
            build_container(settings),
            indexing=coordinator,
            index_store=StubStore(),  # type: ignore[arg-type]
        )

    return TestClient(create_app(settings, container_factory=factory)), coordinator


def test_status_before_any_run(make_settings: SettingsFactory) -> None:
    client, _ = client_with(make_settings(), StubIndexer())

    with client:
        body = client.get("/api/v1/index/status").json()

    assert body["state"] == "idle"
    assert (body["documents"], body["chunks"], body["last_report"]) == (5, 12, None)


def test_trigger_runs_in_the_background_and_status_reports_it(
    make_settings: SettingsFactory,
) -> None:
    indexer = StubIndexer()
    client, coordinator = client_with(make_settings(), indexer)

    with client:
        response = client.post("/api/v1/index")
        client.portal.call(coordinator.wait)  # type: ignore[union-attr]
        body = client.get("/api/v1/index/status").json()

    assert (response.status_code, response.json()) == (202, {"status": "started"})
    assert indexer.calls == 1
    assert body["state"] == "succeeded"
    assert body["last_report"] == {
        "documents_added": 1,
        "documents_updated": 0,
        "documents_unchanged": 4,
        "documents_deleted": 0,
        "chunks_embedded": 3,
        "chunks_reused": 2,
        "failed": [{"document": "bad.md", "reason": "input too long"}],
        "skipped": [],
        "deletion_guarded": False,
        "duration_seconds": 1.235,
    }


def test_trigger_while_running_returns_409(make_settings: SettingsFactory) -> None:
    gate = asyncio.Event()
    client, coordinator = client_with(make_settings(), StubIndexer(gate))

    with client:
        first = client.post("/api/v1/index")
        second = client.post("/api/v1/index")
        client.portal.call(gate.set)  # type: ignore[union-attr]
        client.portal.call(coordinator.wait)  # type: ignore[union-attr]

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "index_running"


def test_documents_lists_the_index(make_settings: SettingsFactory) -> None:
    client, _ = client_with(make_settings(), StubIndexer())

    with client:
        body = client.get("/api/v1/documents").json()

    assert body == {
        "documents": [
            {
                "document": "guides/deployment.md",
                "title": "Deployment",
                "content_hash": "a" * 64,
                "size_bytes": 1234,
                "chunks": 3,
                "indexed_at": "2026-10-10T00:00:00Z",
            }
        ]
    }


def test_startup_indexing_is_started_when_enabled(make_settings: SettingsFactory) -> None:
    indexer = StubIndexer()
    client, coordinator = client_with(make_settings(index_on_startup=True), indexer)

    with client:
        client.portal.call(coordinator.wait)  # type: ignore[union-attr]

    assert indexer.calls == 1


def test_startup_indexing_can_be_disabled(make_settings: SettingsFactory) -> None:
    indexer = StubIndexer()
    client, _ = client_with(make_settings(index_on_startup=False), indexer)

    with client:
        client.get("/health/live")

    assert indexer.calls == 0
