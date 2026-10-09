"""Mapping of domain and infrastructure errors to HTTP responses.

Every error response has the same shape:

    {"error": {"code": "<machine-readable>", "message": "<human-readable>", "details": ...}}

Database errors and unexpected exceptions get generic messages (details are logged,
never returned), so connection strings and stack traces cannot leak to clients.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError

from easyrag.domain.errors import IndexIncompatibleError
from easyrag.embeddings.base import EmbeddingProviderUnavailableError, EmbeddingResponseError

logger = logging.getLogger(__name__)


def error_response(status_code: int, code: str, message: str, details: Any = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = jsonable_encoder(details)
    return JSONResponse(status_code=status_code, content={"error": error})


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT, "validation_error", "Invalid request", exc.errors()
    )


async def _embedding_unavailable(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("Embedding provider unavailable: %s", exc)
    return error_response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "embedding_unavailable",
        f"The embedding model is unavailable: {exc}",
    )


async def _embedding_response(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Embedding provider returned an invalid response: %s", exc)
    return error_response(
        status.HTTP_502_BAD_GATEWAY, "embedding_failed", f"The embedding model failed: {exc}"
    )


async def _index_incompatible(request: Request, exc: Exception) -> JSONResponse:
    logger.error("%s", exc)
    return error_response(status.HTTP_503_SERVICE_UNAVAILABLE, "index_incompatible", str(exc))


async def _database_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Database error (%s)", type(exc).__name__)  # message may contain the DSN
    return error_response(
        status.HTTP_503_SERVICE_UNAVAILABLE, "database_unavailable", "The database is unavailable"
    )


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "An unexpected error occurred"
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(EmbeddingProviderUnavailableError, _embedding_unavailable)
    app.add_exception_handler(EmbeddingResponseError, _embedding_response)
    app.add_exception_handler(IndexIncompatibleError, _index_incompatible)
    app.add_exception_handler(DBAPIError, _database_error)
    app.add_exception_handler(Exception, _unexpected_error)
