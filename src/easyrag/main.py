"""FastAPI application factory.

Run with:  uvicorn --factory easyrag.main:create_app
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer

from easyrag import __version__
from easyrag.api.errors import register_error_handlers
from easyrag.api.middleware import (
    ApiKeyMiddleware,
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from easyrag.api.routes import chat, health, index, metrics, retrieve, ui
from easyrag.config import Settings
from easyrag.container import ContainerFactory, build_container
from easyrag.observability.logging import configure_logging
from easyrag.observability.metrics import Metrics
from easyrag.observability.middleware import RequestContextMiddleware

logger = logging.getLogger(__name__)

# Documentation only (the "Authorize" button in /docs): ApiKeyMiddleware enforces the key.
_API_KEY_SCHEMES = [
    Security(HTTPBearer(auto_error=False, description="Authorization: Bearer <key>")),
    Security(APIKeyHeader(name="X-API-Key", auto_error=False)),
]


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
        if resolved.api_key is None:
            logger.warning(
                "No API key set (EASYRAG_API_KEY): the API is unauthenticated. "
                "Do not expose it beyond localhost."
            )
        container = container_factory(resolved)
        app.state.container = container
        if resolved.ollama_pull_models:
            container.models.start()
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

    # The UI page and assets are public: they hold no data, and the API calls they make
    # need the key like any other client's.
    ui_paths = frozenset({"/"}) if resolved.ui_enabled else frozenset[str]()
    ui_prefixes = (ui.ASSETS_PATH + "/",) if resolved.ui_enabled else ()

    # add_middleware wraps: the last one added runs first. Order of execution:
    # request context -> security headers -> CORS -> API key -> body size limit -> routes.
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=resolved.max_request_bytes, metrics=current_metrics
    )
    if resolved.api_key is not None:
        public = {"/health/live", "/health/ready", app.docs_url, app.redoc_url, app.openapi_url}
        app.add_middleware(
            ApiKeyMiddleware,
            api_key=resolved.api_key.get_secret_value(),
            public_paths=frozenset(path for path in public if path) | ui_paths,
            public_prefixes=ui_prefixes,
            metrics=current_metrics,
        )
    if resolved.cors_allow_origins:
        # Header-based auth needs no cookies, hence no credentials.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )
    app.add_middleware(SecurityHeadersMiddleware, ui_paths=ui_paths, ui_prefixes=ui_prefixes)
    app.add_middleware(RequestContextMiddleware, metrics=current_metrics)
    register_error_handlers(app)

    protected = _API_KEY_SCHEMES if resolved.api_key is not None else []
    app.include_router(health.router)
    app.include_router(retrieve.router, dependencies=protected)
    app.include_router(chat.router, dependencies=protected)
    app.include_router(index.router, dependencies=protected)
    app.include_router(metrics.router)
    if resolved.ui_enabled:
        app.include_router(ui.router)
        app.mount(ui.ASSETS_PATH, ui.assets(), name="assets")
    return app
