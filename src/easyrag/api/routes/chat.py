"""Question answering over the indexed documents."""

from fastapi import APIRouter

from easyrag.api.dependencies import ContainerDep
from easyrag.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest, container: ContainerDep) -> ChatResponse:
    """Answer ``question`` using only the indexed documents, with the sources used."""
    answer = await container.rag.answer(request.question, top_k=request.top_k)
    return ChatResponse.from_domain(answer)
