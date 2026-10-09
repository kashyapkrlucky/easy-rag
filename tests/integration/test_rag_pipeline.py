"""The full pipeline over a real PostgreSQL + pgvector index, with fake models.

Index → retrieve (real SQL) → prompt → fake LLM that reports which sources it received.
"""

import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from easyrag.embeddings.fake import FakeEmbeddingProvider
from easyrag.ingestion.chunker import Chunker
from easyrag.ingestion.indexer import Indexer
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.llm.base import ChatMessage, GenerationOptions
from easyrag.llm.fake import FakeLLMProvider
from easyrag.rag.prompt import PromptBuilder
from easyrag.rag.service import RagService
from easyrag.retrieval.vector import VectorRetriever
from easyrag.storage.index_store import PostgresIndexStore
from easyrag.storage.vector_store import PostgresVectorStore


def report_sources(messages: Sequence[ChatMessage]) -> str:
    """A 'model' that answers with the documents it was shown, in prompt order."""
    return "Saw: " + ", ".join(re.findall(r'document="([^"]+)"', messages[-1].content))


@pytest.fixture
async def db(migrated_database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def rag(tmp_path: Path, db: AsyncEngine) -> RagService:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deployment.md").write_text(
        "# Deployment\n\nStart the stack with docker compose up and wait for readiness.\n"
    )
    (docs / "billing.md").write_text("# Billing\n\nInvoices are sent on the first of the month.\n")
    embeddings = FakeEmbeddingProvider(256, model_id="fake/lexical")
    await Indexer(
        loader_factory=lambda: FilesystemMarkdownLoader(docs),
        parser=MarkdownParser(),
        chunker=Chunker(),
        embeddings=embeddings,
        store=PostgresIndexStore(db),
    ).run()
    llm = FakeLLMProvider(responder=report_sources)
    return RagService(
        retriever=VectorRetriever(embeddings=embeddings, search=PostgresVectorStore(db)),
        llm=llm,
        prompt_builder=PromptBuilder(
            context_window=llm.context_window, max_output_tokens=256, max_context_tokens=2000
        ),
        generation_options=GenerationOptions(max_output_tokens=256),
    )


async def test_answer_is_generated_from_retrieved_documents(rag: RagService) -> None:
    answer = await rag.answer("How do I start the stack with docker compose?", top_k=2)

    assert answer.text == "Saw: deployment.md, billing.md"  # best match first
    assert [s.source_path for s in answer.sources] == ["deployment.md", "billing.md"]
    assert answer.sources[0].section == "Deployment"
    assert answer.trace.included_chunk_ids == tuple(s.chunk_id for s in answer.sources)


async def test_top_k_limits_the_evidence(rag: RagService) -> None:
    answer = await rag.answer("When are invoices sent?", top_k=1)

    assert answer.text == "Saw: billing.md"
    assert len(answer.sources) == 1
