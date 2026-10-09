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
from easyrag.ingestion.coordinator import IndexingCoordinator
from easyrag.ingestion.indexer import Indexer
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.ingestion.watcher import DocsWatcher
from easyrag.llm.base import GenerationOptions, LLMProvider
from easyrag.llm.ollama import OllamaLLMProvider
from easyrag.rag.prompt import PromptBuilder
from easyrag.rag.service import RagService
from easyrag.retrieval.base import Retriever
from easyrag.retrieval.hybrid import HybridRetriever
from easyrag.retrieval.keyword import KeywordRetriever
from easyrag.retrieval.vector import VectorRetriever
from easyrag.storage.database import create_engine, ping
from easyrag.storage.index_store import PostgresIndexStore
from easyrag.storage.keyword_store import PostgresKeywordStore
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
    llm: LLMProvider
    indexer: Indexer
    indexing: IndexingCoordinator
    watcher: DocsWatcher
    index_store: PostgresIndexStore
    # The configured retriever, and every mode by name (for per-request overrides).
    retriever: Retriever
    retrievers: Mapping[str, Retriever]
    rag: RagService
    # Reported by /health/ready, in this order.
    readiness_checks: Mapping[str, ReadinessCheck]

    async def aclose(self) -> None:
        # Stop event sources first, then indexing, then the clients indexing uses.
        await self.watcher.aclose()
        await self.indexing.aclose()
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

    llm = OllamaLLMProvider(
        http_client, model=settings.llm_model, context_window=settings.llm_context_window
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

    vector = VectorRetriever(
        embeddings=embeddings,
        search=PostgresVectorStore(engine),
        default_top_k=settings.retrieval_top_k,
        default_min_score=settings.retrieval_min_score,
    )
    keyword = KeywordRetriever(
        search=PostgresKeywordStore(engine), default_top_k=settings.retrieval_top_k
    )
    retrievers: dict[str, Retriever] = {
        "vector": vector,
        "keyword": keyword,
        "hybrid": HybridRetriever(
            vector=vector, keyword=keyword, default_top_k=settings.retrieval_top_k
        ),
    }
    retriever = retrievers[settings.retrieval_mode]

    rag = RagService(
        retriever=retriever,
        llm=llm,
        prompt_builder=PromptBuilder(
            context_window=llm.context_window,
            max_output_tokens=settings.llm_max_output_tokens,
            max_context_tokens=settings.rag_max_context_tokens,
        ),
        generation_options=GenerationOptions(
            temperature=settings.llm_temperature,
            seed=settings.llm_seed,
            max_output_tokens=settings.llm_max_output_tokens,
        ),
        ungrounded_policy=settings.rag_ungrounded_policy,
    )

    indexing = IndexingCoordinator(indexer)

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
        llm=llm,
        indexer=indexer,
        indexing=indexing,
        watcher=DocsWatcher(
            settings.docs_dir,
            indexing.request_run,
            debounce_ms=settings.watch_debounce_ms,
            force_polling=settings.watch_force_polling,
            resync_seconds=settings.watch_resync_seconds,
        ),
        index_store=index_store,
        retriever=retriever,
        retrievers=retrievers,
        rag=rag,
        readiness_checks={
            "database": database,
            "schema": schema,
            "embedding_model": embeddings.readiness,
            "llm_model": llm.readiness,
            "index": index,
        },
    )
