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
from easyrag.observability.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("EasyRAG %s starting", __version__)
    yield
    logger.info("EasyRAG shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. ``settings`` defaults to values from the environment."""
    settings = settings if settings is not None else Settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="EasyRAG", version=__version__, lifespan=_lifespan)
    app.state.settings = settings
    app.include_router(health.router)
    return app
