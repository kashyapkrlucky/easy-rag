from pathlib import Path

import pytest
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.ext.asyncio import create_async_engine

from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.evaluation.dataset import load_dataset
from easyrag.evaluation.runner import evaluate_retrieval, isolated_database, summarize_retrieval
from easyrag.ingestion.chunker import Chunker
from easyrag.ingestion.indexer import Indexer
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.retrieval.base import Retriever
from easyrag.retrieval.hybrid import HybridRetriever
from easyrag.retrieval.keyword import KeywordRetriever
from easyrag.retrieval.vector import VectorRetriever
from easyrag.storage.index_store import PostgresIndexStore
from easyrag.storage.keyword_store import PostgresKeywordStore
from easyrag.storage.vector_store import PostgresVectorStore

NIMBUS = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "nimbus"


def databases(url: str) -> set[str]:
    engine = create_engine(make_url(url))
    try:
        with engine.connect() as connection:
            return set(connection.execute(text("SELECT datname FROM pg_database")).scalars())
    finally:
        engine.dispose()


async def test_isolated_database_is_migrated_and_dropped(database_url: str) -> None:
    before = databases(database_url)

    async with isolated_database(database_url) as url:
        name = make_url(url).database
        assert name is not None and name.startswith("easyrag_eval_")
        assert name in databases(database_url)
        engine = create_engine(url)
        with engine.connect() as connection:
            tables = set(connection.execute(text("SELECT tablename FROM pg_tables")).scalars())
        engine.dispose()
        assert {"documents", "chunks", "index_state"} <= tables

    assert databases(database_url) == before


async def test_isolated_database_is_dropped_even_on_failure(database_url: str) -> None:
    before = databases(database_url)

    with pytest.raises(RuntimeError):
        async with isolated_database(database_url):
            raise RuntimeError("evaluation crashed")

    assert databases(database_url) == before


async def test_retrieval_evaluation_on_the_nimbus_corpus(database_url: str) -> None:
    dataset = load_dataset(NIMBUS)
    embeddings = FakeEmbeddingProvider(256, model_id="fake/lexical")
    async with isolated_database(database_url) as url:
        engine = create_async_engine(url)
        try:
            await Indexer(
                loader_factory=lambda: FilesystemMarkdownLoader(dataset.corpus_dir),
                parser=MarkdownParser(),
                chunker=Chunker(),
                embeddings=embeddings,
                store=PostgresIndexStore(engine),
            ).run()
            vector = VectorRetriever(embeddings=embeddings, search=PostgresVectorStore(engine))
            keyword = KeywordRetriever(search=PostgresKeywordStore(engine))
            configs: dict[str, Retriever] = {
                "vector": vector,
                "keyword": keyword,
                "hybrid": HybridRetriever(vector=vector, keyword=keyword),
            }

            rows = await evaluate_retrieval(configs, dataset.questions, top_k=5)
        finally:
            await engine.dispose()

    answerable = sum(1 for q in dataset.questions if q.answerable)
    assert len(rows) == 3 * answerable
    summary = summarize_retrieval(rows)
    assert set(summary) == {"vector", "keyword", "hybrid"}
    # Identifier questions are keyword search's strength, even with a lexical fake embedder.
    assert summary["keyword"]["by_kind"]["identifier"]["hit_at_k"] == 1.0
