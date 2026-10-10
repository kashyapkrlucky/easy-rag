"""Question answering over the indexed documents."""

from fastapi import APIRouter

from easyrag.api.dependencies import ContainerDep
from easyrag.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


# Omits "trace" unless the client asked for it (it is the only field left unset).
@router.post("/chat", response_model_exclude_unset=True)
async def chat(request: ChatRequest, container: ContainerDep) -> ChatResponse:
    """Answer ``question`` using only the indexed documents, with the sources used."""
    with container.chat_slots.slot():  # 503 "busy" when the limit is reached
        answer = await container.rag.answer(request.question, top_k=request.top_k)
    return ChatResponse.from_domain(answer, debug=request.debug)
