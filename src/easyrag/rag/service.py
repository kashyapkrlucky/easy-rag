"""RAG orchestration: question → retrieve → grounded prompt → generate → answer.

Depends only on ports (``Retriever``, ``LLMProvider``) and the prompt builder, never on
Ollama, pgvector, HTTP or FastAPI (enforced by tests/unit/test_architecture.py).
"""

import logging
import time

from easyrag.domain.answers import Answer, RagTrace
from easyrag.llm.base import GenerationOptions, LLMProvider
from easyrag.rag.prompt import PROMPT_VERSION, PromptBuilder
from easyrag.retrieval.base import Retriever

logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMProvider,
        prompt_builder: PromptBuilder,
        generation_options: GenerationOptions,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._generation_options = generation_options

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
        generation = await self._llm.generate(prompt.messages, self._generation_options)
        finished = time.perf_counter()

        trace = RagTrace(
            prompt_version=PROMPT_VERSION,
            llm_model=generation.model,
            retrieved=tuple((chunk.chunk_id, chunk.score) for chunk in chunks),
            included_chunk_ids=tuple(chunk.chunk_id for chunk in prompt.included),
            excluded_chunk_ids=tuple(chunk.chunk_id for chunk in prompt.excluded),
            prompt_tokens_estimate=prompt.tokens_estimate,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            finish_reason=generation.finish_reason,
            retrieval_seconds=retrieved_at - started,
            generation_seconds=finished - retrieved_at,
            total_seconds=finished - started,
        )
        _log(trace)
        logger.debug("Question: %r", question)  # user content only at DEBUG

        return Answer(
            question=question,
            text=generation.text.strip(),
            sources=prompt.included,
            truncated=generation.finish_reason == "length",
            trace=trace,
        )


def _log(trace: RagTrace) -> None:
    logger.info(
        "RAG answer: prompt_version=%s model=%s retrieved=%d included=%d excluded=%d "
        "prompt_tokens=%s (est %d) completion_tokens=%s finish=%s "
        "retrieval=%.2fs generation=%.2fs total=%.2fs",
        trace.prompt_version,
        trace.llm_model,
        len(trace.retrieved),
        len(trace.included_chunk_ids),
        len(trace.excluded_chunk_ids),
        trace.prompt_tokens,
        trace.prompt_tokens_estimate,
        trace.completion_tokens,
        trace.finish_reason,
        trace.retrieval_seconds,
        trace.generation_seconds,
        trace.total_seconds,
    )
