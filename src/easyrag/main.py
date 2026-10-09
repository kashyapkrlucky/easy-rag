"""FastAPI application factory.

Run with:  uvicorn --factory easyrag.main:create_app
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from easyrag import __version__
from easyrag.api.routes import health
from easyrag.config import Settings
from easyrag.container import build_container
from easyrag.observability.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. ``settings`` defaults to values from the environment."""
    resolved = settings if settings is not None else Settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("EasyRAG %s starting", __version__)
        container = build_container(resolved)
        app.state.container = container
        try:
            yield
        finally:
            await container.aclose()
            logger.info("EasyRAG shut down")

    app = FastAPI(title="EasyRAG", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    return app
