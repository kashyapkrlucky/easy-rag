"""FastAPI application factory.

Run with:  uvicorn --factory easyrag.main:create_app
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from easyrag import __version__
from easyrag.api.errors import register_error_handlers
from easyrag.api.routes import chat, health, index, metrics, retrieve
from easyrag.config import Settings
from easyrag.container import ContainerFactory, build_container
from easyrag.observability.logging import configure_logging
from easyrag.observability.metrics import Metrics
from easyrag.observability.middleware import RequestContextMiddleware

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, container_factory: ContainerFactory = build_container
) -> FastAPI:
    """Build the application.

    ``settings`` defaults to values from the environment; ``container_factory`` lets tests
    substitute collaborators (e.g. fake readiness checks or providers).
    """
    resolved = settings if settings is not None else Settings()
    configure_logging(resolved.log_level, resolved.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("EasyRAG %s starting", __version__)
        container = container_factory(resolved)
        app.state.container = container
        if resolved.index_on_startup:
            container.indexing.start_until_success()
        if resolved.watch_enabled:
            container.watcher.start()
        if resolved.rerank_enabled:
            container.reranker.start_loading()
        try:
            yield
        finally:
            await container.aclose()
            logger.info("EasyRAG shut down")

    app = FastAPI(title="EasyRAG", version=__version__, lifespan=lifespan)

    def current_metrics() -> Metrics | None:
        container = getattr(app.state, "container", None)
        return container.metrics if container is not None else None

    app.add_middleware(RequestContextMiddleware, metrics=current_metrics)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(retrieve.router)
    app.include_router(chat.router)
    app.include_router(index.router)
    app.include_router(metrics.router)
    return app
