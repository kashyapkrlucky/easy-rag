"""Composition root.

The only place that decides which concrete implementations the application uses.
Built once per application in the FastAPI lifespan and closed on shutdown.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx2
from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.config import Settings
from easyrag.embeddings.base import EmbeddingProvider
from easyrag.embeddings.ollama import OllamaEmbeddingProvider
from easyrag.health import CheckStatus, ReadinessCheck
from easyrag.ingestion.chunker import Chunker, ChunkingConfig
from easyrag.ingestion.indexer import Indexer
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.retrieval.base import Retriever
from easyrag.retrieval.vector import VectorRetriever
from easyrag.storage.database import create_engine, ping
from easyrag.storage.index_store import PostgresIndexStore
from easyrag.storage.migrate import SchemaStatus, schema_status
from easyrag.storage.vector_store import PostgresVectorStore

# Connecting should be fast even when a request (model load + inference) is slow.
_OLLAMA_CONNECT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    engine: AsyncEngine
    http_client: httpx2.AsyncClient
    embeddings: EmbeddingProvider
    indexer: Indexer
    retriever: Retriever
    # Reported by /health/ready, in this order.
    readiness_checks: Mapping[str, ReadinessCheck]

    async def aclose(self) -> None:
        await self.http_client.aclose()
        await self.engine.dispose()


ContainerFactory = Callable[[Settings], AppContainer]


def build_container(settings: Settings) -> AppContainer:
    engine = create_engine(settings)
    http_client = httpx2.AsyncClient(
        base_url=str(settings.ollama_base_url),
        timeout=httpx2.Timeout(
            settings.ollama_timeout_seconds, connect=_OLLAMA_CONNECT_TIMEOUT_SECONDS
        ),
    )
    embeddings = OllamaEmbeddingProvider(
        http_client,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )

    index_store = PostgresIndexStore(engine)
    indexer = Indexer(
        loader_factory=lambda: FilesystemMarkdownLoader(settings.docs_dir),
        parser=MarkdownParser(),
        chunker=Chunker(
            ChunkingConfig(
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
        ),
        embeddings=embeddings,
        store=index_store,
    )

    retriever = VectorRetriever(
        embeddings=embeddings,
        search=PostgresVectorStore(engine),
        default_top_k=settings.retrieval_top_k,
        default_min_score=settings.retrieval_min_score,
    )

    async def database() -> CheckStatus:
        await ping(engine, settings.health_check_timeout_seconds)
        return "ok"

    async def schema() -> CheckStatus:
        status = await schema_status(engine)
        return "ok" if status is SchemaStatus.CURRENT else "outdated"

    async def index() -> CheckStatus:
        return await index_store.compatibility(embeddings.model_id, embeddings.dimension)

    return AppContainer(
        settings=settings,
        engine=engine,
        http_client=http_client,
        embeddings=embeddings,
        indexer=indexer,
        retriever=retriever,
        readiness_checks={
            "database": database,
            "schema": schema,
            "embedding_model": embeddings.readiness,
            "index": index,
        },
    )
