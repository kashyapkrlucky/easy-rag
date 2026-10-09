"""Health endpoints.

``/health/live``  — is the process up? Never touches dependencies.
``/health/ready`` — can the service do useful work? Checks each dependency with a timeout.
"""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from easyrag.api.dependencies import ContainerDep
from easyrag.storage.database import ping
from easyrag.storage.migrate import SchemaStatus, schema_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

CheckStatus = Literal["ok", "unavailable", "outdated"]


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckStatus]


@router.get("/live")
async def live() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    "/ready",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(container: ContainerDep, response: Response) -> ReadinessResponse:
    timeout = container.settings.health_check_timeout_seconds
    checks: dict[str, CheckStatus] = {}

    try:
        await ping(container.engine, timeout)
        checks["database"] = "ok"
    except Exception as exc:
        # Log the exception type only: messages can include connection details.
        logger.warning("Readiness: database unavailable (%s)", type(exc).__name__)
        checks["database"] = "unavailable"

    if checks["database"] == "ok":
        try:
            async with asyncio.timeout(timeout):
                schema = await schema_status(container.engine)
            checks["schema"] = "ok" if schema is SchemaStatus.CURRENT else "outdated"
        except Exception as exc:
            logger.warning("Readiness: schema check failed (%s)", type(exc).__name__)
            checks["schema"] = "unavailable"
    else:
        checks["schema"] = "unavailable"

    if checks["schema"] == "outdated":
        logger.warning("Readiness: database schema is not at the expected migration revision")

    if all(result == "ok" for result in checks.values()):
        return ReadinessResponse(status="ready", checks=checks)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="not_ready", checks=checks)
