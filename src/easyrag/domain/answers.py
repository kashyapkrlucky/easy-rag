"""Answers produced by the RAG pipeline, and the trace of how each was produced."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from easyrag.domain.chunks import RetrievedChunk


class Outcome(StrEnum):
    """How an answer came about (finer-grained than the client-facing status)."""

    ANSWERED = "answered"  # the reply cites at least one valid source
    REFUSED = "refused"  # the model replied with the refusal sentence
    NO_EVIDENCE = "no_evidence"  # nothing to show the model; it was not called
    UNGROUNDED = "ungrounded"  # the reply cites no valid source


# What clients see: was the question answered from the documents?
AnswerStatus = Literal["answered", "insufficient_evidence", "ungrounded"]


@dataclass(frozen=True, slots=True)
class RagTrace:
    """What happened while answering one question. Contains no document text."""

    prompt_version: str
    outcome: Outcome
    # (chunk_id, score) of every retrieved chunk, best first.
    retrieved: tuple[tuple[str, float], ...]
    # Chunks given to the model, and retrieved chunks left out by the token budget.
    included_chunk_ids: tuple[str, ...]
    excluded_chunk_ids: tuple[str, ...]
    # Chunks the answer cites, in citation order; out-of-range markers that were dropped.
    cited_chunk_ids: tuple[str, ...]
    invalid_citations: int
    # How sources were determined: from the model's [n] markers, or inferred server-side
    # from lexical support when the reply had no valid markers. None if nothing was cited.
    attribution: str | None
    prompt_tokens_estimate: int
    # None when the model was not called (no evidence).
    llm_model: str | None
    finish_reason: str | None
    # As reported by the LLM backend, when available.
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    text: str
    status: AnswerStatus
    # The cited chunks: in the answer text, "[k]" refers to sources[k - 1].
    sources: tuple[RetrievedChunk, ...]
    # True when generation stopped at the output-token limit: the answer is incomplete.
    truncated: bool
    trace: RagTrace

    @property
    def grounded(self) -> bool:
        """False only for answers returned despite citing no source (policy "flag")."""
        return self.status != "ungrounded"
