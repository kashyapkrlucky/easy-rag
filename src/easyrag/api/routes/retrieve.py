"""Retrieval without generation: for debugging retrieval quality and for evaluation."""

from fastapi import APIRouter

from easyrag.api.dependencies import ContainerDep
from easyrag.api.schemas import RetrievedChunkResponse, RetrieveRequest, RetrieveResponse

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest, container: ContainerDep) -> RetrieveResponse:
    """Return the chunks most similar to ``query``, best first."""
    chunks = await container.retriever.retrieve(
        request.query, top_k=request.top_k, min_score=request.min_score
    )
    return RetrieveResponse(
        query=request.query,
        embedding_model=container.embeddings.model_id,
        results=[RetrievedChunkResponse.from_domain(chunk) for chunk in chunks],
    )
