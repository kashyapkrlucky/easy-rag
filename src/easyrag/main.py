"""FastAPI application factory.

Run with:  uvicorn --factory easyrag.main:create_app
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from easyrag import __version__
from easyrag.api.errors import register_error_handlers
from easyrag.api.routes import health, retrieve
from easyrag.config import Settings
from easyrag.container import ContainerFactory, build_container
from easyrag.observability.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, container_factory: ContainerFactory = build_container
) -> FastAPI:
    """Build the application.

    ``settings`` defaults to values from the environment; ``container_factory`` lets tests
    substitute collaborators (e.g. fake readiness checks or providers).
    """
    resolved = settings if settings is not None else Settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("EasyRAG %s starting", __version__)
        container = container_factory(resolved)
        app.state.container = container
        try:
            yield
        finally:
            await container.aclose()
            logger.info("EasyRAG shut down")

    app = FastAPI(title="EasyRAG", version=__version__, lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(retrieve.router)
    return app
