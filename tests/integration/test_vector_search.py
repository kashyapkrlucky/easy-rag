from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from easyrag.domain.errors import IndexIncompatibleError
from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.ingestion.chunker import Chunker
from easyrag.ingestion.indexer import Indexer
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.retrieval.vector import VectorRetriever
from easyrag.storage.index_store import PostgresIndexStore, vector_index_name
from easyrag.storage.vector_store import PostgresVectorStore, search_sql

DIMENSION = 256  # large enough that the lexical fake embedder rarely collides


@pytest.fixture
async def db(migrated_database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(DIMENSION, model_id="fake/lexical")


@pytest.fixture
async def indexed(tmp_path: Path, db: AsyncEngine, embeddings: FakeEmbeddingProvider) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deployment.md").write_text(
        "# Deployment\n\nStart everything with docker compose up.\n\n"
        "## Volumes\n\nMount the docs folder read only into the container.\n\n"
        "## Upgrades\n\nPull the new image and restart; migrations run automatically.\n"
    )
    (docs / "api.md").write_text(
        "# API\n\nThe chat endpoint answers questions with citations.\n\n"
        "## Authentication\n\nSend the API key in the Authorization header.\n"
    )
    (docs / "cooking.md").write_text("# Pasta\n\nBoil water, add salt, cook the pasta.\n")
    await Indexer(
        loader_factory=lambda: FilesystemMarkdownLoader(docs),
        parser=MarkdownParser(),
        chunker=Chunker(),
        embeddings=embeddings,
        store=PostgresIndexStore(db),
    ).run()
    return docs


def retriever(db: AsyncEngine, embeddings: FakeEmbeddingProvider) -> VectorRetriever:
    return VectorRetriever(embeddings=embeddings, search=PostgresVectorStore(db))


async def test_most_similar_section_ranks_first(
    indexed: Path, db: AsyncEngine, embeddings: FakeEmbeddingProvider
) -> None:
    results = await retriever(db, embeddings).retrieve("mount docs folder into the container")

    top = results[0]
    assert (top.source_path, top.heading_path, top.section) == (
        "deployment.md",
        ("Deployment", "Volumes"),
        "Volumes",
    )
    assert top.document_title == "Deployment"
    assert top.content == "Mount the docs folder read only into the container."
    assert (top.start_line, top.end_line) == (7, 7)


async def test_scores_are_cosine_similarities_in_descending_order(
    indexed: Path, db: AsyncEngine, embeddings: FakeEmbeddingProvider
) -> None:
    query = "send the api key in the authorization header"
    results = await retriever(db, embeddings).retrieve(query, top_k=50)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert len(results) == 6  # every chunk, when top_k exceeds the corpus
    # Score = 1 - cosine distance = dot product of unit vectors (float32 storage).
    (query_vector,) = [await embeddings.embed_query(query)]
    (top_vector,) = await embeddings.embed_documents(
        ["API > Authentication\n\nSend the API key in the Authorization header."]
    )
    expected = sum(a * b for a, b in zip(query_vector, top_vector, strict=True))
    assert results[0].score == pytest.approx(expected, abs=1e-5)


async def test_top_k_limits_results(
    indexed: Path, db: AsyncEngine, embeddings: FakeEmbeddingProvider
) -> None:
    assert len(await retriever(db, embeddings).retrieve("docker", top_k=2)) == 2


async def test_min_score_filters_after_top_k(
    indexed: Path, db: AsyncEngine, embeddings: FakeEmbeddingProvider
) -> None:
    unfiltered = await retriever(db, embeddings).retrieve("docker compose up", top_k=6)
    threshold = (unfiltered[0].score + unfiltered[1].score) / 2

    filtered = await retriever(db, embeddings).retrieve(
        "docker compose up", top_k=6, min_score=threshold
    )

    assert filtered == [r for r in unfiltered if r.score >= threshold]
    assert len(filtered) == 1


async def test_ties_are_broken_by_chunk_id(tmp_path: Path, db: AsyncEngine) -> None:
    # Same title + same text → identical embed_text → identical vectors and scores,
    # but distinct chunk IDs (they include the path).
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("c.md", "a.md", "b.md"):
        (docs / name).write_text("---\ntitle: Same\n---\nIdentical text.\n")
    embeddings = FakeEmbeddingProvider(DIMENSION, model_id="fake/lexical")
    await Indexer(
        loader_factory=lambda: FilesystemMarkdownLoader(docs),
        parser=MarkdownParser(),
        chunker=Chunker(),
        embeddings=embeddings,
        store=PostgresIndexStore(db),
    ).run()

    first = await retriever(db, embeddings).retrieve("identical text", top_k=3)
    second = await retriever(db, embeddings).retrieve("identical text", top_k=3)

    assert len({r.score for r in first}) == 1
    assert [r.chunk_id for r in first] == sorted(r.chunk_id for r in first)
    assert first == second


async def test_empty_index_returns_no_results(
    db: AsyncEngine, embeddings: FakeEmbeddingProvider
) -> None:
    assert await retriever(db, embeddings).retrieve("anything") == []


async def test_search_refuses_another_vector_space(indexed: Path, db: AsyncEngine) -> None:
    other = FakeEmbeddingProvider(DIMENSION, model_id="fake/other")

    with pytest.raises(IndexIncompatibleError, match="easyrag index --rebuild"):
        await retriever(db, other).retrieve("docker")


async def test_production_query_uses_the_hnsw_index(indexed: Path, db: AsyncEngine) -> None:
    query = "[" + ",".join(["0.0625"] * DIMENSION) + "]"

    async with db.connect() as connection:
        await connection.execute(text("SET enable_seqscan = off"))
        plan = "\n".join(
            (
                await connection.execute(
                    text("EXPLAIN " + search_sql(DIMENSION, with_min_score=True)),
                    {
                        "query": query,
                        "embedding_model": "fake/lexical",
                        "top_k": 5,
                        "min_score": 0.1,
                    },
                )
            ).scalars()
        )

    assert vector_index_name(DIMENSION) in plan, plan
