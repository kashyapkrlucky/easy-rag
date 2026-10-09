"""Answers produced by the RAG pipeline, and the trace of how each was produced."""

from dataclasses import dataclass

from easyrag.domain.chunks import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RagTrace:
    """What happened while answering one question. Contains no document text."""

    prompt_version: str
    llm_model: str
    # (chunk_id, score) of every retrieved chunk, best first.
    retrieved: tuple[tuple[str, float], ...]
    # Chunks given to the model, and retrieved chunks left out by the token budget.
    included_chunk_ids: tuple[str, ...]
    excluded_chunk_ids: tuple[str, ...]
    prompt_tokens_estimate: int
    # As reported by the LLM backend, when available.
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    text: str
    # The chunks the model was shown, in the order they appeared in the prompt.
    sources: tuple[RetrievedChunk, ...]
    # True when generation stopped at the output-token limit: the answer is incomplete.
    truncated: bool
    trace: RagTrace
