from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from easyrag.domain.errors import IndexIncompatibleError
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
    StoredDocumentState,
    embedding_input_hash,
    pipeline_fingerprint,
)
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser


class InMemoryIndexStore:
    """IndexStore test double with the same semantics as the Postgres store."""

    def __init__(self) -> None:
        self.documents: dict[str, DocumentToStore] = {}
        self.state: tuple[str, int] | None = None
        self.resets = 0
        self.saves: list[str] = []
        self.locked = False

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

    @asynccontextmanager
    async def lock(self) -> AsyncIterator[None]:
        assert not self.locked, "runs must not overlap"
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    async def document_states(self) -> Mapping[str, StoredDocumentState]:
        return {
            path: StoredDocumentState(d.document.content_hash, d.pipeline_fingerprint)
            for path, d in self.documents.items()
        }

    async def find_embeddings(
        self, input_hashes: Sequence[str], embedding_model: str
    ) -> Mapping[str, Vector]:
        found: dict[str, Vector] = {}
        for document in self.documents.values():
            if document.embedding_model != embedding_model:
                continue
            for chunk, vector in zip(document.chunks, document.embeddings, strict=True):
                if embedding_input_hash(chunk) in input_hashes:
                    found[embedding_input_hash(chunk)] = vector
        return found

    async def save_document(self, document: DocumentToStore) -> None:
        assert self.locked, "writes must happen under the lock"
        self.documents[document.document.source_path] = document
        self.saves.append(document.document.source_path)

    async def delete_documents(self, source_paths: Sequence[str]) -> None:
        assert self.locked, "writes must happen under the lock"
        for path in source_paths:
            del self.documents[path]

    async def reset(self) -> None:
        self.documents.clear()
        self.state = None
        self.resets += 1


class RejectingEmbeddings(FakeEmbeddingProvider):
    """Fails for texts containing a marker, like a model rejecting one input."""

    def __init__(self, marker: str, error: Exception) -> None:
        super().__init__(dimension=16)
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


def counts(report: Any) -> tuple[int, int, int, int]:
    return (
        report.documents_added,
        report.documents_updated,
        report.documents_unchanged,
        report.documents_deleted,
    )


# -- first run -------------------------------------------------------------------------


async def test_first_run_indexes_every_document(docs: Path) -> None:
    store = InMemoryIndexStore()
    embeddings = FakeEmbeddingProvider(dimension=16)

    report = await make_indexer(docs, store, embeddings).run()

    assert sorted(store.documents) == ["deployment.md", "empty.md", "guides/api.md"]
    assert counts(report) == (3, 0, 0, 0)
    assert (report.chunks_embedded, report.chunks_reused) == (3, 0)
    deployment = store.documents["deployment.md"]
    assert [c.heading_path for c in deployment.chunks] == [
        ("Deployment",),
        ("Deployment", "Volumes"),
    ]
    assert list(deployment.embeddings) == await FakeEmbeddingProvider(16).embed_documents(
        [c.embed_text for c in deployment.chunks]
    )
    assert store.state == (embeddings.model_id, 16)
    assert store.documents["empty.md"].chunks == ()


# -- incremental behaviour -------------------------------------------------------------


async def test_unchanged_corpus_makes_no_embedding_calls_and_no_writes(docs: Path) -> None:
    store = InMemoryIndexStore()
    embeddings = FakeEmbeddingProvider(dimension=16)
    indexer = make_indexer(docs, store, embeddings)
    await indexer.run()
    batches, saves = len(embeddings.document_batches), len(store.saves)

    report = await indexer.run()

    assert counts(report) == (0, 0, 3, 0)
    assert len(embeddings.document_batches) == batches
    assert len(store.saves) == saves


async def test_editing_one_section_re_embeds_only_that_section(docs: Path) -> None:
    store = InMemoryIndexStore()
    embeddings = FakeEmbeddingProvider(dimension=16)
    indexer = make_indexer(docs, store, embeddings)
    await indexer.run()
    embedded_before = embeddings.embedded_document_count

    text = (docs / "deployment.md").read_text().replace("read-only", "read-write")
    (docs / "deployment.md").write_text(text)
    report = await indexer.run()

    assert counts(report) == (0, 1, 2, 0)
    assert (report.chunks_embedded, report.chunks_reused) == (1, 1)
    assert embeddings.embedded_document_count == embedded_before + 1
    assert "read-write" in store.documents["deployment.md"].chunks[1].content


async def test_same_text_under_a_new_heading_gets_a_new_vector(docs: Path) -> None:
    # The embedded text includes the heading path, so the old vector must not be reused.
    store = InMemoryIndexStore()
    indexer = make_indexer(docs, store)
    await indexer.run()
    old_vector = store.documents["deployment.md"].embeddings[1]

    text = (docs / "deployment.md").read_text().replace("## Volumes", "## Mounts")
    (docs / "deployment.md").write_text(text)
    report = await indexer.run()

    assert report.chunks_embedded == 1
    assert store.documents["deployment.md"].embeddings[1] != old_vector


async def test_deleted_file_is_removed_from_the_index(docs: Path) -> None:
    store = InMemoryIndexStore()
    indexer = make_indexer(docs, store)
    await indexer.run()

    (docs / "guides" / "api.md").unlink()
    report = await indexer.run()

    assert counts(report) == (0, 0, 2, 1)
    assert "guides/api.md" not in store.documents


async def test_renamed_file_reuses_its_vectors(docs: Path) -> None:
    store = InMemoryIndexStore()
    embeddings = FakeEmbeddingProvider(dimension=16)
    indexer = make_indexer(docs, store, embeddings)
    await indexer.run()
    embedded_before = embeddings.embedded_document_count

    (docs / "guides" / "api.md").rename(docs / "guides" / "chat-api.md")
    report = await indexer.run()

    assert counts(report) == (1, 0, 2, 1)
    assert (report.chunks_embedded, report.chunks_reused) == (0, 1)
    assert embeddings.embedded_document_count == embedded_before
    assert "guides/chat-api.md" in store.documents
    assert "guides/api.md" not in store.documents


async def test_pipeline_change_re_indexes_every_document(docs: Path) -> None:
    store = InMemoryIndexStore()
    await make_indexer(docs, store, config=ChunkingConfig(max_tokens=400)).run()

    report = await make_indexer(docs, store, config=ChunkingConfig(max_tokens=300)).run()

    assert counts(report) == (0, 3, 0, 0)
    assert len({d.pipeline_fingerprint for d in store.documents.values()}) == 1


async def test_skipped_file_keeps_its_previously_indexed_version(docs: Path) -> None:
    store = InMemoryIndexStore()
    indexer = make_indexer(docs, store)
    await indexer.run()

    (docs / "guides" / "api.md").write_bytes(b"\xff\xfe not utf-8")  # mid-edit, say
    report = await indexer.run()

    assert [s.source_path for s in report.skipped] == ["guides/api.md"]
    assert report.documents_deleted == 0
    assert "chat answers questions" in store.documents["guides/api.md"].chunks[0].content


async def test_empty_docs_directory_does_not_wipe_the_index(docs: Path, tmp_path: Path) -> None:
    store = InMemoryIndexStore()
    await make_indexer(docs, store).run()
    empty = tmp_path / "empty-mount"
    empty.mkdir()

    report = await make_indexer(empty, store).run()

    assert report.deletion_guarded
    assert report.documents_deleted == 0
    assert len(store.documents) == 3


async def test_rebuild_empties_the_index_even_from_an_empty_directory(
    docs: Path, tmp_path: Path
) -> None:
    store = InMemoryIndexStore()
    await make_indexer(docs, store).run()
    empty = tmp_path / "empty"
    empty.mkdir()

    report = await make_indexer(empty, store).run(rebuild=True)

    assert store.documents == {}
    assert not report.deletion_guarded


# -- compatibility and failures --------------------------------------------------------


async def test_incompatible_index_is_refused_before_any_embedding(docs: Path) -> None:
    store = InMemoryIndexStore()
    await make_indexer(docs, store, FakeEmbeddingProvider(16, model_id="model-a")).run()
    other = FakeEmbeddingProvider(16, model_id="model-b")

    with pytest.raises(IndexIncompatibleError, match="easyrag index --rebuild"):
        await make_indexer(docs, store, other).run()
    assert other.document_batches == []


async def test_rebuild_switches_to_the_new_model(docs: Path) -> None:
    store = InMemoryIndexStore()
    await make_indexer(docs, store, FakeEmbeddingProvider(16, model_id="model-a")).run()

    report = await make_indexer(docs, store, FakeEmbeddingProvider(32, model_id="model-b")).run(
        rebuild=True
    )

    assert store.resets == 1
    assert store.state == ("model-b", 32)
    assert report.documents_added == 3
    assert all(len(v) == 32 for d in store.documents.values() for v in d.embeddings)


async def test_document_level_embedding_error_keeps_the_previous_version(docs: Path) -> None:
    store = InMemoryIndexStore()
    await make_indexer(docs, store).run()
    (docs / "guides" / "api.md").write_text("# API\n\nREJECT this new text.\n")
    embeddings = RejectingEmbeddings("REJECT", EmbeddingResponseError("input too long"))

    report = await make_indexer(docs, store, embeddings).run()

    assert [(f.source_path, f.reason) for f in report.failed] == [
        ("guides/api.md", "input too long")
    ]
    assert report.documents_deleted == 0
    assert "chat answers questions" in store.documents["guides/api.md"].chunks[0].content


async def test_unavailable_provider_aborts_the_run_keeping_saved_documents(docs: Path) -> None:
    store = InMemoryIndexStore()
    # guides/api.md is processed last (sorted order); the model "goes down" there.
    embeddings = RejectingEmbeddings(
        "POST /api/v1/chat", EmbeddingProviderUnavailableError("Ollama is down")
    )

    with pytest.raises(EmbeddingProviderUnavailableError):
        await make_indexer(docs, store, embeddings).run()
    assert sorted(store.documents) == ["deployment.md", "empty.md"]
    assert not store.locked  # the lock is released on failure


async def test_missing_docs_directory_fails_the_run_not_construction(tmp_path: Path) -> None:
    indexer = make_indexer(tmp_path / "missing", InMemoryIndexStore())

    with pytest.raises(NotADirectoryError):
        await indexer.run()


# -- fingerprint -----------------------------------------------------------------------


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
