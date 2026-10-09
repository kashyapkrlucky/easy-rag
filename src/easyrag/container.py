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
from easyrag.storage.database import create_engine, ping
from easyrag.storage.migrate import SchemaStatus, schema_status

# Connecting should be fast even when a request (model load + inference) is slow.
_OLLAMA_CONNECT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    engine: AsyncEngine
    http_client: httpx2.AsyncClient
    embeddings: EmbeddingProvider
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

    async def database() -> CheckStatus:
        await ping(engine, settings.health_check_timeout_seconds)
        return "ok"

    async def schema() -> CheckStatus:
        status = await schema_status(engine)
        return "ok" if status is SchemaStatus.CURRENT else "outdated"

    return AppContainer(
        settings=settings,
        engine=engine,
        http_client=http_client,
        embeddings=embeddings,
        readiness_checks={
            "database": database,
            "schema": schema,
            "embedding_model": embeddings.readiness,
        },
    )
