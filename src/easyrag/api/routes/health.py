"""Health endpoints.

``/health/live`` answers "is the process up?" and never touches dependencies.
Readiness (``/health/ready``) arrives with the first real dependency in Phase 3.
"""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/live")
async def live() -> LivenessResponse:
    return LivenessResponse()
