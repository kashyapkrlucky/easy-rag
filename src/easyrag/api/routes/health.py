"""Health endpoints.

``/health/live``  — is the process up? Never touches dependencies.
``/health/ready`` — can the service do useful work? Checks each dependency with a timeout.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from easyrag.api.dependencies import ContainerDep
from easyrag.storage.database import ping

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

CheckStatus = Literal["ok", "unavailable"]


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
    checks: dict[str, CheckStatus] = {}

    try:
        await ping(container.engine, container.settings.health_check_timeout_seconds)
        checks["database"] = "ok"
    except Exception as exc:
        # Log the exception type only: messages can include connection details.
        logger.warning("Readiness: database unavailable (%s)", type(exc).__name__)
        checks["database"] = "unavailable"

    if all(result == "ok" for result in checks.values()):
        return ReadinessResponse(status="ready", checks=checks)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="not_ready", checks=checks)
