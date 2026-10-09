from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from easyrag.embeddings.base import (
    EmbeddingProviderUnavailableError,
    EmbeddingResponseError,
    Vector,
)
from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.ingestion.chunker import Chunker, ChunkingConfig
from easyrag.ingestion.indexer import (
    DocumentToStore,
    Indexer,
    IndexIncompatibleError,
    pipeline_fingerprint,
)
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser


class InMemoryIndexStore:
    """IndexStore test double with the same compatibility semantics as Postgres."""

    def __init__(self) -> None:
        self.documents: dict[str, DocumentToStore] = {}
        self.state: tuple[str, int] | None = None
        self.resets = 0

    async def prepare(self, embedding_model: str, dimension: int) -> None:
        if self.state is None:
            self.state = (embedding_model, dimension)
        elif self.state != (embedding_model, dimension):
            raise IndexIncompatibleError(
                stored_model=self.state[0],
                stored_dimension=self.state[1],
                configured_model=embedding_model,
                configured_dimension=dimension,
            )

    async def save_document(self, document: DocumentToStore) -> None:
        self.documents[document.document.source_path] = document

    async def reset(self) -> None:
        self.documents.clear()
        self.state = None
        self.resets += 1


class RejectingEmbeddings(FakeEmbeddingProvider):
    """Fails for texts containing a marker, like a model rejecting one input."""

    def __init__(self, marker: str, error: Exception) -> None:
        super().__init__()
        self._marker, self._error = marker, error

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if any(self._marker in text for text in texts):
            raise self._error
        return await super().embed_documents(texts)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "guides").mkdir(parents=True)
    (root / "deployment.md").write_text(
        "# Deployment\n\nRun docker compose up.\n\n## Volumes\n\nMount ./docs read-only.\n"
    )
    (root / "guides" / "api.md").write_text("# API\n\nPOST /api/v1/chat answers questions.\n")
    (root / "empty.md").write_text("")
    return root


def make_indexer(
    docs: Path,
    store: InMemoryIndexStore,
    embeddings: FakeEmbeddingProvider | None = None,
    config: ChunkingConfig | None = None,
) -> Indexer:
    return Indexer(
        loader_factory=lambda: FilesystemMarkdownLoader(docs),
        parser=MarkdownParser(),
        chunker=Chunker(config),
        embeddings=embeddings or FakeEmbeddingProvider(dimension=16),
        store=store,
    )


async def test_indexes_every_document_with_its_chunks_and_vectors(docs: Path) -> None:
    store = InMemoryIndexStore()
    embeddings = FakeEmbeddingProvider(dimension=16)

    report = await make_indexer(docs, store, embeddings).run()

    assert sorted(store.documents) == ["deployment.md", "empty.md", "guides/api.md"]
    assert (report.documents_indexed, report.chunks_indexed) == (3, 3)
    assert report.failed == ()
    assert report.skipped == ()

    deployment = store.documents["deployment.md"]
    assert [c.heading_path for c in deployment.chunks] == [
        ("Deployment",),
        ("Deployment", "Volumes"),
    ]
    assert len(deployment.embeddings) == len(deployment.chunks)
    assert list(deployment.embeddings) == await FakeEmbeddingProvider(16).embed_documents(
        [c.embed_text for c in deployment.chunks]
    )
    assert deployment.embedding_model == embeddings.model_id
    assert store.state == (embeddings.model_id, 16)


async def test_document_without_content_is_stored_without_chunks(docs: Path) -> None:
    store = InMemoryIndexStore()

    await make_indexer(docs, store).run()

    assert store.documents["empty.md"].chunks == ()


async def test_rerunning_produces_identical_results(docs: Path) -> None:
    store = InMemoryIndexStore()
    indexer = make_indexer(docs, store)

    await indexer.run()
    first = dict(store.documents)
    await indexer.run()

    assert store.documents == first


async def test_incompatible_index_is_refused_before_any_embedding(docs: Path) -> None:
    store = InMemoryIndexStore()
    await make_indexer(docs, store, FakeEmbeddingProvider(16, model_id="model-a")).run()
    other = FakeEmbeddingProvider(16, model_id="model-b")

    with pytest.raises(IndexIncompatibleError, match="easyrag index --rebuild"):
        await make_indexer(docs, store, other).run()
    assert other.document_batches == []


async def test_rebuild_resets_and_reindexes_with_the_new_model(docs: Path) -> None:
    store = InMemoryIndexStore()
    await make_indexer(docs, store, FakeEmbeddingProvider(16, model_id="model-a")).run()

    report = await make_indexer(docs, store, FakeEmbeddingProvider(32, model_id="model-b")).run(
        rebuild=True
    )

    assert store.resets == 1
    assert store.state == ("model-b", 32)
    assert report.documents_indexed == 3
    assert all(len(v) == 32 for d in store.documents.values() for v in d.embeddings)


async def test_document_level_embedding_error_is_reported_and_others_continue(docs: Path) -> None:
    store = InMemoryIndexStore()
    embeddings = RejectingEmbeddings("POST /api/v1/chat", EmbeddingResponseError("input too long"))

    report = await make_indexer(docs, store, embeddings).run()

    assert sorted(store.documents) == ["deployment.md", "empty.md"]
    assert [(f.source_path, f.reason) for f in report.failed] == [
        ("guides/api.md", "input too long")
    ]


async def test_unavailable_provider_aborts_the_run_keeping_saved_documents(docs: Path) -> None:
    store = InMemoryIndexStore()
    # guides/api.md is processed last (sorted order); the model "goes down" there.
    embeddings = RejectingEmbeddings(
        "POST /api/v1/chat", EmbeddingProviderUnavailableError("Ollama is down")
    )

    with pytest.raises(EmbeddingProviderUnavailableError):
        await make_indexer(docs, store, embeddings).run()
    assert sorted(store.documents) == ["deployment.md", "empty.md"]


async def test_skipped_files_are_reported(docs: Path) -> None:
    (docs / "broken.md").write_bytes(b"\xff\xfe not utf-8")

    report = await make_indexer(docs, InMemoryIndexStore()).run()

    assert [(s.source_path, s.reason) for s in report.skipped] == [("broken.md", "not valid UTF-8")]


async def test_missing_docs_directory_fails_the_run_not_construction(tmp_path: Path) -> None:
    indexer = make_indexer(tmp_path / "missing", InMemoryIndexStore())  # constructing is fine

    with pytest.raises(NotADirectoryError):
        await indexer.run()


async def test_every_document_carries_the_pipeline_fingerprint(docs: Path) -> None:
    store = InMemoryIndexStore()
    embeddings = FakeEmbeddingProvider(16)
    config = ChunkingConfig(max_tokens=200, overlap_tokens=20)

    await make_indexer(docs, store, embeddings, config).run()

    expected = pipeline_fingerprint(
        max_tokens=200, overlap_tokens=20, embedding_model=embeddings.model_id, dimension=16
    )
    assert {d.pipeline_fingerprint for d in store.documents.values()} == {expected}


def test_pipeline_fingerprint_changes_with_every_input() -> None:
    base: dict[str, Any] = {
        "max_tokens": 400,
        "overlap_tokens": 50,
        "embedding_model": "m",
        "dimension": 8,
    }
    variants = [
        base,
        {**base, "max_tokens": 300},
        {**base, "overlap_tokens": 0},
        {**base, "embedding_model": "other"},
        {**base, "dimension": 16},
    ]

    fingerprints = [pipeline_fingerprint(**v) for v in variants]

    assert len(set(fingerprints)) == len(variants)
    assert pipeline_fingerprint(**base) == fingerprints[0]
