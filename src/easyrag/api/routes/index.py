"""Index management: status, manual trigger, and the list of indexed documents."""

from fastapi import APIRouter, status

from easyrag.api.dependencies import ContainerDep
from easyrag.api.schemas import (
    DocumentResponse,
    DocumentsResponse,
    IndexStatusResponse,
    IndexTriggerResponse,
)

router = APIRouter(prefix="/api/v1", tags=["index"])


@router.get("/index/status")
async def index_status(container: ContainerDep) -> IndexStatusResponse:
    """The current or last indexing run, and the size of the index."""
    documents, chunks = await container.index_store.counts()
    watcher = container.watcher.state if container.settings.watch_enabled else "disabled"
    return IndexStatusResponse.from_domain(container.indexing.status(), documents, chunks, watcher)


@router.post(
    "/index",
    status_code=status.HTTP_202_ACCEPTED,
    responses={status.HTTP_409_CONFLICT: {"description": "A run is already in progress"}},
)
async def trigger_index(container: ContainerDep) -> IndexTriggerResponse:
    """Start an incremental indexing run in the background; poll /index/status."""
    container.indexing.trigger()  # IndexRunInProgressError → 409 (api/errors.py)
    return IndexTriggerResponse()


@router.get("/documents")
async def documents(container: ContainerDep) -> DocumentsResponse:
    """Every indexed document, with its chunk count."""
    return DocumentsResponse(
        documents=[
            DocumentResponse.from_domain(d) for d in await container.index_store.list_documents()
        ]
    )
