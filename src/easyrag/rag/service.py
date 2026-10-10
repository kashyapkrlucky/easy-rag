"""RAG orchestration: question → retrieve → grounded prompt → generate → verified answer.

Grounding is enforced in layers (ARCHITECTURE D12, D13):
1. Evidence gate: with nothing to show the model, answer with the refusal sentence
   without calling it.
2. Prompt contract: answer only from the sources and cite them as [n].
3. Verification: classify the reply (refusal / cited / uncited), keep only valid
   citations, and return exactly the cited sources. A cited reply must be supported by
   the sources it cites, sentence by sentence (``rag.verification``, D39). A reply without
   valid markers is attributed server-side by lexical support (``rag.attribution``). A
   reply that fails either is ungrounded, and is refused or flagged according to the
   ``UngroundedPolicy``.

Before generation, instruction-like text in the sources is redacted (``easyrag.injection``,
D39), and verification uses the text the model was actually shown.

Depends only on ports (``Retriever``, ``LLMProvider``) and pure helpers, never on
Ollama, pgvector, HTTP or FastAPI (enforced by tests/unit/test_architecture.py).
"""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Literal

from easyrag.domain.answers import Answer, AnswerStatus, Outcome, RagTrace
from easyrag.domain.chunks import RetrievedChunk
from easyrag.llm.base import Generation, GenerationOptions, LLMProvider
from easyrag.rag.attribution import attribute
from easyrag.rag.citations import apply_citations, is_refusal
from easyrag.rag.prompt import PROMPT_VERSION, REFUSAL, BuiltPrompt, PromptBuilder
from easyrag.rag.verification import verify_citations
from easyrag.retrieval.base import Retriever

logger = logging.getLogger(__name__)

# What to do with a reply that cites no valid source:
#   "refuse" — replace it with the refusal sentence (strict grounding);
#   "flag"   — return it with status "ungrounded" (grounded=false).
UngroundedPolicy = Literal["refuse", "flag"]

# Called with every answer (e.g. to record metrics). Must not raise; errors are logged.
AnswerObserver = Callable[[Answer], None]


class RagService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMProvider,
        prompt_builder: PromptBuilder,
        generation_options: GenerationOptions,
        ungrounded_policy: UngroundedPolicy = "refuse",
        on_answer: AnswerObserver | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._generation_options = generation_options
        self._ungrounded_policy = ungrounded_policy
        self._on_answer = on_answer

    async def answer(self, question: str, *, top_k: int | None = None) -> Answer:
        """Answer ``question`` from the indexed documents.

        Raises the retriever's and LLM provider's errors unchanged (e.g.
        ``EmbeddingProviderUnavailableError``, ``LLMProviderUnavailableError``).
        """
        question = question.strip()
        if not question:
            raise ValueError("question must not be blank")
        started = time.perf_counter()

        chunks = await self._retriever.retrieve(question, top_k=top_k)
        retrieved_at = time.perf_counter()
        prompt = self._prompt_builder.build(question, chunks)

        if prompt.redacted:
            logger.warning(
                "Removed instruction-like text (possible prompt injection) from %d source(s) "
                "before generation: %s",
                len(prompt.redacted),
                ", ".join(sorted({chunk.source_path for chunk in prompt.redacted})),
            )

        if not prompt.included:  # evidence gate: never ask the model without sources
            return self._finish(
                question, chunks, prompt, None, Outcome.NO_EVIDENCE, (), 0,
                text=REFUSAL, status="insufficient_evidence",
                timings=(started, retrieved_at, retrieved_at),
            )  # fmt: skip

        generation = await self._llm.generate(prompt.messages, self._generation_options)
        generated_at = time.perf_counter()
        timings = (started, retrieved_at, generated_at)

        if is_refusal(generation.text, REFUSAL):
            return self._finish(
                question, chunks, prompt, generation, Outcome.REFUSED, (), 0,
                text=REFUSAL, status="insufficient_evidence", timings=timings,
            )  # fmt: skip

        citations = apply_citations(generation.text, len(prompt.included))
        cited = tuple(prompt.included[number - 1] for number in citations.cited)
        invalid = len(citations.invalid)
        unverified = 0
        if cited:
            verification = verify_citations(generation.text, prompt.evidence)
            if verification.corrected:
                logger.info("Corrected %d misattributed citation(s)", len(verification.corrected))
                citations = apply_citations(verification.reply, len(prompt.included))
                cited = tuple(prompt.included[number - 1] for number in citations.cited)
            if verification.verified:
                return self._finish(
                    question, chunks, prompt, generation, Outcome.ANSWERED, cited, invalid,
                    text=citations.text, status="answered", timings=timings,
                    attribution="explicit",
                )  # fmt: skip
            unverified = len(verification.failures)
            logger.warning(
                "Cited answer failed verification: %d sentence(s) not supported by the "
                "sources they cite",
                unverified,
            )
            logger.debug("Unverified sentences: %r", verification.failures)  # model output
        else:
            inferred = attribute(citations.text, prompt.included, prompt.evidence)
            if inferred.supported:
                return self._finish(
                    question, chunks, prompt, generation, Outcome.ANSWERED, inferred.sources,
                    invalid, text=citations.text, status="answered", timings=timings,
                    attribution="inferred",
                )  # fmt: skip

        status: AnswerStatus
        if self._ungrounded_policy == "flag":
            text, status = citations.text, "ungrounded"
        else:
            text, status = REFUSAL, "insufficient_evidence"
        return self._finish(
            question, chunks, prompt, generation, Outcome.UNGROUNDED, (), invalid,
            text=text, status=status, timings=timings, unverified=unverified,
        )  # fmt: skip

    def _finish(
        self,
        question: str,
        retrieved: Sequence[RetrievedChunk],
        prompt: BuiltPrompt,
        generation: Generation | None,
        outcome: Outcome,
        cited: tuple[RetrievedChunk, ...],
        invalid_citations: int,
        *,
        text: str,
        status: AnswerStatus,
        timings: tuple[float, float, float],
        attribution: str | None = None,
        unverified: int = 0,
    ) -> Answer:
        started, retrieved_at, generated_at = timings
        trace = RagTrace(
            prompt_version=PROMPT_VERSION,
            outcome=outcome,
            retrieved=tuple((chunk.chunk_id, chunk.score) for chunk in retrieved),
            included_chunk_ids=tuple(chunk.chunk_id for chunk in prompt.included),
            excluded_chunk_ids=tuple(chunk.chunk_id for chunk in prompt.excluded),
            cited_chunk_ids=tuple(chunk.chunk_id for chunk in cited),
            invalid_citations=invalid_citations,
            attribution=attribution,
            prompt_tokens_estimate=prompt.tokens_estimate if generation else 0,
            llm_model=generation.model if generation else None,
            finish_reason=generation.finish_reason if generation else None,
            prompt_tokens=generation.prompt_tokens if generation else None,
            completion_tokens=generation.completion_tokens if generation else None,
            retrieval_seconds=retrieved_at - started,
            generation_seconds=generated_at - retrieved_at,
            total_seconds=time.perf_counter() - started,
            reranked=any(chunk.rerank_score is not None for chunk in retrieved),
            redacted_sources=len(prompt.redacted),
            unverified_sentences=unverified,
        )
        _log(trace)
        logger.debug("Question: %r", question)  # user content only at DEBUG
        if outcome is Outcome.UNGROUNDED:
            logger.warning(
                "Answer cited no valid source (policy=%s, invalid citations=%d)",
                self._ungrounded_policy,
                invalid_citations,
            )
        answer = Answer(
            question=question,
            text=text.strip(),
            status=status,
            sources=cited,
            truncated=generation is not None and generation.finish_reason == "length",
            trace=trace,
        )
        if self._on_answer is not None:
            try:
                self._on_answer(answer)
            except Exception:  # observability must never fail a request
                logger.exception("Answer observer failed")
        return answer


def _log(trace: RagTrace) -> None:
    logger.info(
        "RAG answer: outcome=%s prompt_version=%s model=%s retrieved=%d included=%d "
        "excluded=%d cited=%d attribution=%s invalid_citations=%d prompt_tokens=%s (est %d) "
        "completion_tokens=%s finish=%s retrieval=%.2fs generation=%.2fs total=%.2fs",
        trace.outcome,
        trace.prompt_version,
        trace.llm_model,
        len(trace.retrieved),
        len(trace.included_chunk_ids),
        len(trace.excluded_chunk_ids),
        len(trace.cited_chunk_ids),
        trace.attribution,
        trace.invalid_citations,
        trace.prompt_tokens,
        trace.prompt_tokens_estimate,
        trace.completion_tokens,
        trace.finish_reason,
        trace.retrieval_seconds,
        trace.generation_seconds,
        trace.total_seconds,
        extra={
            "rag": {
                "outcome": trace.outcome,
                "prompt_version": trace.prompt_version,
                "llm_model": trace.llm_model,
                "retrieved": len(trace.retrieved),
                "reranked": trace.reranked,
                "top_score": round(trace.retrieved[0][1], 6) if trace.retrieved else None,
                "included": len(trace.included_chunk_ids),
                "excluded": len(trace.excluded_chunk_ids),
                "cited": len(trace.cited_chunk_ids),
                "attribution": trace.attribution,
                "invalid_citations": trace.invalid_citations,
                "redacted_sources": trace.redacted_sources,
                "unverified_sentences": trace.unverified_sentences,
                "prompt_tokens": trace.prompt_tokens,
                "prompt_tokens_estimate": trace.prompt_tokens_estimate,
                "completion_tokens": trace.completion_tokens,
                "finish_reason": trace.finish_reason,
                "retrieval_ms": round(trace.retrieval_seconds * 1000, 1),
                "generation_ms": round(trace.generation_seconds * 1000, 1),
                "total_ms": round(trace.total_seconds * 1000, 1),
            }
        },
    )
