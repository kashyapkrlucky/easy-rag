"""Health endpoints.

``/health/live``  — is the process up? Never touches dependencies.
``/health/ready`` — can the service do useful work? Runs the registered readiness checks.
"""

from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from easyrag.api.dependencies import ContainerDep
from easyrag.health import CheckStatus, run_checks

router = APIRouter(prefix="/health", tags=["health"])


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
    checks = await run_checks(
        container.readiness_checks, container.settings.health_check_timeout_seconds
    )
    if all(result == "ok" for result in checks.values()):
        return ReadinessResponse(status="ready", checks=checks)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="not_ready", checks=checks)
