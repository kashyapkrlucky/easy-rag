import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from easyrag.domain.errors import IndexIncompatibleError
from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.ingestion.chunker import Chunker
from easyrag.ingestion.indexer import Indexer
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.storage.index_store import PostgresIndexStore, vector_index_name

DIMENSION = 8


@pytest.fixture
async def db(migrated_database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "deployment.md").write_text(
        "---\ntitle: Deployment Guide\ntags: [ops]\n---\n"
        "# Deployment\n\nRun docker compose up.\n\n## Volumes\n\nMount ./docs read-only.\n"
    )
    (root / "api.md").write_text("# API\n\nPOST /api/v1/chat answers questions.\n")
    return root


def make_indexer(
    docs: Path, engine: AsyncEngine, embeddings: FakeEmbeddingProvider | None = None
) -> Indexer:
    return Indexer(
        loader_factory=lambda: FilesystemMarkdownLoader(docs),
        parser=MarkdownParser(),
        chunker=Chunker(),
        embeddings=embeddings or FakeEmbeddingProvider(DIMENSION, model_id="fake/a"),
        store=PostgresIndexStore(engine),
    )


async def rows(engine: AsyncEngine, sql: str) -> list[dict[str, Any]]:
    async with engine.connect() as connection:
        return [dict(row) for row in (await connection.execute(text(sql))).mappings()]


SNAPSHOT = """
    SELECT d.id AS document_id, d.source_path, d.content_hash, d.pipeline_fingerprint,
           c.id, c.ordinal, c.content_hash AS chunk_hash, c.embedding::text AS embedding
    FROM documents d JOIN chunks c ON c.document_id = d.id
    ORDER BY d.source_path, c.ordinal
"""


async def test_documents_chunks_and_vectors_are_stored(docs: Path, db: AsyncEngine) -> None:
    report = await make_indexer(docs, db).run()

    assert (report.documents_added, report.chunks_embedded) == (2, 3)
    documents = await rows(
        db, "SELECT source_path, title, front_matter FROM documents ORDER BY source_path"
    )
    assert documents == [
        {"source_path": "api.md", "title": "API", "front_matter": {}},
        {
            "source_path": "deployment.md",
            "title": "Deployment Guide",
            "front_matter": {"title": "Deployment Guide", "tags": ["ops"]},
        },
    ]
    chunks = await rows(
        db,
        "SELECT c.heading_path, c.start_line, c.end_line, c.embedding_model,"
        " vector_dims(c.embedding) AS dims, round(vector_norm(c.embedding)::numeric, 4) AS norm"
        " FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE d.source_path = 'deployment.md' ORDER BY c.ordinal",
    )
    assert chunks == [
        {"heading_path": ["Deployment"], "start_line": 7, "end_line": 7,
         "embedding_model": "fake/a", "dims": DIMENSION, "norm": 1},
        {"heading_path": ["Deployment", "Volumes"], "start_line": 11, "end_line": 11,
         "embedding_model": "fake/a", "dims": DIMENSION, "norm": 1},
    ]  # fmt: skip


async def test_stored_vectors_match_the_provider_output(docs: Path, db: AsyncEngine) -> None:
    embeddings = FakeEmbeddingProvider(DIMENSION, model_id="fake/a")
    await make_indexer(docs, db, embeddings).run()
    api = next(
        d for d in FilesystemMarkdownLoader(docs).load().documents if d.source_path == "api.md"
    )
    (chunk,) = Chunker().chunk(MarkdownParser().parse(api))
    (expected,) = await FakeEmbeddingProvider(DIMENSION).embed_documents([chunk.embed_text])

    async with db.connect() as connection:
        stored = (
            await connection.execute(
                text("SELECT embedding::text FROM chunks WHERE id = :id"), {"id": chunk.id}
            )
        ).scalar_one()
    actual = [float(x) for x in stored.strip("[]").split(",")]

    # pgvector stores float32: compare with single-precision tolerance.
    assert actual == pytest.approx(expected, abs=1e-6)


async def test_rerunning_produces_identical_rows(docs: Path, db: AsyncEngine) -> None:
    indexer = make_indexer(docs, db)

    await indexer.run()
    before = await rows(db, SNAPSHOT)
    await indexer.run()
    after = await rows(db, SNAPSHOT)

    assert after == before
    assert len(before) == 3


async def test_changed_document_is_replaced_without_orphans(docs: Path, db: AsyncEngine) -> None:
    indexer = make_indexer(docs, db)
    await indexer.run()
    (before,) = await rows(db, "SELECT id FROM documents WHERE source_path = 'api.md'")

    (docs / "api.md").write_text("# API\n\nNew text.\n\n## Auth\n\nUse an API key.\n")
    await indexer.run()

    (after,) = await rows(db, "SELECT id FROM documents WHERE source_path = 'api.md'")
    contents = await rows(
        db,
        "SELECT content FROM chunks WHERE document_id = "
        "(SELECT id FROM documents WHERE source_path = 'api.md') ORDER BY ordinal",
    )
    assert after == before  # document identity is stable across re-indexing
    assert [c["content"] for c in contents] == ["New text.", "Use an API key."]
    assert (await rows(db, "SELECT count(*) AS n FROM chunks"))[0]["n"] == 4


async def test_index_records_its_model_and_refuses_another(docs: Path, db: AsyncEngine) -> None:
    await make_indexer(docs, db).run()
    other = FakeEmbeddingProvider(DIMENSION, model_id="fake/b")

    state = await rows(db, "SELECT key, value FROM index_state ORDER BY key")
    with pytest.raises(IndexIncompatibleError):
        await make_indexer(docs, db, other).run()

    assert state == [
        {"key": "embedding_dimension", "value": str(DIMENSION)},
        {"key": "embedding_model", "value": "fake/a"},
    ]
    store = PostgresIndexStore(db)
    assert await store.compatibility("fake/a", DIMENSION) == "ok"
    assert await store.compatibility("fake/b", DIMENSION) == "misconfigured"
    assert await store.compatibility("fake/a", 16) == "misconfigured"
    assert other.document_batches == []


async def test_empty_index_is_compatible_with_anything(db: AsyncEngine) -> None:
    assert await PostgresIndexStore(db).compatibility("any/model", 1024) == "ok"


async def test_rebuild_switches_model_and_vector_index(docs: Path, db: AsyncEngine) -> None:
    await make_indexer(docs, db).run()
    other = FakeEmbeddingProvider(16, model_id="fake/b")

    report = await make_indexer(docs, db, other).run(rebuild=True)

    indexes = await rows(
        db, "SELECT indexname FROM pg_indexes WHERE indexname LIKE 'ix_chunks_embedding_hnsw_%'"
    )
    assert [i["indexname"] for i in indexes] == [vector_index_name(16)]
    dims = await rows(
        db, "SELECT DISTINCT vector_dims(embedding) AS d, embedding_model FROM chunks"
    )
    assert dims == [{"d": 16, "embedding_model": "fake/b"}]
    assert report.documents_indexed == 2


async def test_similarity_queries_can_use_the_hnsw_index(docs: Path, db: AsyncEngine) -> None:
    await make_indexer(docs, db).run()
    query = "[" + ",".join(["0.35"] * DIMENSION) + "]"

    async with db.connect() as connection:
        await connection.execute(text("SET enable_seqscan = off"))
        plan = (
            await connection.execute(
                text(
                    f"EXPLAIN SELECT id FROM chunks "
                    f"ORDER BY embedding::vector({DIMENSION}) <=> CAST(:q AS vector({DIMENSION})) "
                    f"LIMIT 5"
                ),
                {"q": query},
            )
        ).scalars()
        plan_text = "\n".join(plan)

    assert vector_index_name(DIMENSION) in plan_text, plan_text


# -- incremental indexing (Phase 15) ---------------------------------------------------


async def test_unchanged_second_run_embeds_and_writes_nothing(docs: Path, db: AsyncEngine) -> None:
    embeddings = FakeEmbeddingProvider(DIMENSION, model_id="fake/a")
    indexer = make_indexer(docs, db, embeddings)
    await indexer.run()
    before = await rows(db, SNAPSHOT)
    batches = len(embeddings.document_batches)

    report = await indexer.run()

    assert (report.documents_unchanged, report.documents_indexed) == (2, 0)
    assert len(embeddings.document_batches) == batches
    assert await rows(db, SNAPSHOT) == before  # not even rewritten


async def test_vectors_are_reused_by_embedding_input(docs: Path, db: AsyncEngine) -> None:
    embeddings = FakeEmbeddingProvider(DIMENSION, model_id="fake/a")
    indexer = make_indexer(docs, db, embeddings)
    await indexer.run()
    (docs / "api.md").rename(docs / "chat-api.md")

    report = await indexer.run()

    assert (report.documents_added, report.documents_deleted) == (1, 1)
    assert (report.chunks_embedded, report.chunks_reused) == (0, 1)
    hashes = await rows(db, "SELECT count(*) AS n FROM chunks WHERE embedding_input_hash IS NULL")
    assert hashes == [{"n": 0}]


async def test_removed_documents_are_deleted_with_their_chunks(docs: Path, db: AsyncEngine) -> None:
    indexer = make_indexer(docs, db)
    await indexer.run()
    (docs / "api.md").unlink()

    report = await indexer.run()

    assert report.documents_deleted == 1
    assert await rows(db, "SELECT source_path FROM documents") == [{"source_path": "deployment.md"}]
    assert (await rows(db, "SELECT count(*) AS n FROM chunks"))[0]["n"] == 2


async def test_concurrent_runs_are_serialised(docs: Path, db: AsyncEngine) -> None:
    first = make_indexer(docs, db, FakeEmbeddingProvider(DIMENSION, model_id="fake/a"))
    second = make_indexer(docs, db, FakeEmbeddingProvider(DIMENSION, model_id="fake/a"))

    reports = await asyncio.gather(first.run(), second.run())

    # One run indexed everything; the other waited for the lock and found nothing to do.
    assert sorted(r.documents_added for r in reports) == [0, 2]
    assert sorted(r.documents_unchanged for r in reports) == [0, 2]
    assert (await rows(db, "SELECT count(*) AS n FROM documents"))[0]["n"] == 2


async def test_store_lists_documents_and_counts(docs: Path, db: AsyncEngine) -> None:
    await make_indexer(docs, db).run()
    store = PostgresIndexStore(db)

    listed = await store.list_documents()

    assert [(d.source_path, d.title, d.chunk_count) for d in listed] == [
        ("api.md", "API", 1),
        ("deployment.md", "Deployment Guide", 2),
    ]
    assert await store.counts() == (2, 3)
