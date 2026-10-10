"""Prometheus scrape endpoint."""

from fastapi import APIRouter, Response

from easyrag.api.dependencies import ContainerDep
from easyrag.observability.metrics import CONTENT_TYPE

router = APIRouter(tags=["observability"])


@router.get("/metrics", include_in_schema=False)
async def metrics(container: ContainerDep) -> Response:
    return Response(container.metrics.render(), media_type=CONTENT_TYPE)
