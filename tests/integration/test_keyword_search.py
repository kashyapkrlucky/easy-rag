"""Keyword search on real PostgreSQL full-text search, and hybrid fusion over real stores."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.ingestion.chunker import Chunker
from easyrag.ingestion.indexer import Indexer
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.retrieval.hybrid import HybridRetriever
from easyrag.retrieval.keyword import KeywordRetriever
from easyrag.retrieval.vector import VectorRetriever
from easyrag.storage.index_store import PostgresIndexStore
from easyrag.storage.keyword_store import PostgresKeywordStore
from easyrag.storage.migrate import alembic_config
from easyrag.storage.vector_store import PostgresVectorStore

EMBEDDINGS = FakeEmbeddingProvider(256, model_id="fake/lexical")


@pytest.fixture
async def db(migrated_database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def indexed(tmp_path: Path, db: AsyncEngine) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    # "model" is common; the identifier appears in exactly one section.
    (docs / "configuration.md").write_text(
        "# Configuration\n\n"
        "## Embedding model\n\nSet EASYRAG_EMBEDDING_MODEL to choose the embedding model.\n\n"
        "## Chat model\n\nThe chat model answers questions. The model runs in Ollama. "
        "Model choice matters; a larger model is slower.\n"
    )
    (docs / "deployment.md").write_text(
        "# Deployment\n\nRun docker compose up. Models are pulled by the model service.\n\n"
        "## Errors\n\nError E1042 means the database is unreachable.\n"
    )
    (docs / "scattered.md").write_text(
        "# Notes\n\nThe embedding is stored. Easyrag is the project. The model is local.\n"
    )
    await Indexer(
        loader_factory=lambda: FilesystemMarkdownLoader(docs),
        parser=MarkdownParser(),
        chunker=Chunker(),
        embeddings=EMBEDDINGS,
        store=PostgresIndexStore(db),
    ).run()
    return docs


def keyword(db: AsyncEngine) -> KeywordRetriever:
    return KeywordRetriever(search=PostgresKeywordStore(db))


async def test_search_vector_is_written_for_every_chunk(indexed: Path, db: AsyncEngine) -> None:
    async with db.connect() as connection:
        missing = (
            await connection.execute(
                text("SELECT count(*) FROM chunks WHERE search_vector IS NULL")
            )
        ).scalar_one()
        heading_weighted = (
            await connection.execute(
                text(
                    "SELECT count(*) FROM chunks "
                    "WHERE search_vector @@ to_tsquery('simple', 'errors:A')"
                )
            )
        ).scalar_one()

    assert missing == 0
    assert heading_weighted == 1  # the "Errors" heading is indexed with weight A


async def test_identifier_is_matched_as_a_phrase(indexed: Path, db: AsyncEngine) -> None:
    results = await keyword(db).retrieve("How do I configure EASYRAG_EMBEDDING_MODEL?")

    top = results[0]
    assert top.heading_path == ("Configuration", "Embedding model")
    assert top.keyword_score == top.score
    # The scattered note contains easyrag, embedding and model, but not as the phrase.
    scattered = [r for r in results if r.source_path == "scattered.md"]
    assert not scattered or scattered[0].score < top.score


async def test_rare_terms_outweigh_frequent_ones(indexed: Path, db: AsyncEngine) -> None:
    # Each candidate matches exactly one term. Without IDF this is a tie that raw frequency
    # breaks in favour of the chat section ("model" 4x); with IDF the rare E1042 must win.
    results = await keyword(db).retrieve("model E1042")

    assert results[0].heading_path == ("Deployment", "Errors")


async def test_natural_questions_are_not_and_queries(indexed: Path, db: AsyncEngine) -> None:
    # websearch_to_tsquery would require every word ("how", "do", "i", ...) and find nothing.
    results = await keyword(db).retrieve("How do I start the stack with docker compose?")

    assert results and results[0].source_path == "deployment.md"


async def test_scores_are_the_matched_share_of_query_weight(indexed: Path, db: AsyncEngine) -> None:
    results = await keyword(db).retrieve("E1042 unreachable")

    assert results[0].score == pytest.approx(1.0)  # every query term matched
    assert all(0 < r.score <= 1 for r in results)


async def test_unknown_terms_find_nothing(indexed: Path, db: AsyncEngine) -> None:
    assert await keyword(db).retrieve("kubernetes autoscaler") == []


async def test_keyword_search_uses_the_gin_index(indexed: Path, db: AsyncEngine) -> None:
    async with db.connect() as connection:
        await connection.execute(text("SET enable_seqscan = off"))
        plan = "\n".join(
            (
                await connection.execute(
                    text(
                        "EXPLAIN SELECT id FROM chunks "
                        "WHERE search_vector @@ phraseto_tsquery('simple', 'e1042')"
                    )
                )
            ).scalars()
        )

    assert "ix_chunks_search_vector" in plan, plan


async def test_hybrid_fuses_both_retrievers(indexed: Path, db: AsyncEngine) -> None:
    vector = VectorRetriever(embeddings=EMBEDDINGS, search=PostgresVectorStore(db))
    hybrid = HybridRetriever(vector=vector, keyword=keyword(db))

    results = await hybrid.retrieve("What does error E1042 mean?", top_k=3)

    top = results[0]
    assert top.heading_path == ("Deployment", "Errors")
    assert top.vector_score is not None and top.keyword_score is not None  # found by both
    assert results == await hybrid.retrieve("What does error E1042 mean?", top_k=3)


async def test_migration_backfills_existing_chunks(
    indexed: Path, db: AsyncEngine, migrated_database_url: str
) -> None:
    config = alembic_config(migrated_database_url)
    async with db.begin() as connection:
        await connection.execute(text("UPDATE chunks SET search_vector = NULL"))
    await db.dispose()  # release connections before running DDL

    command.downgrade(config, "0003")
    command.upgrade(config, "head")

    async with db.connect() as connection:
        missing = (
            await connection.execute(
                text("SELECT count(*) FROM chunks WHERE search_vector IS NULL")
            )
        ).scalar_one()
    assert missing == 0
