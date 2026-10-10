"""Prometheus metrics, held in a registry owned by one application instance.

A per-app ``CollectorRegistry`` (instead of prometheus_client's global one) keeps tests
and multiple app instances independent and makes the dependency explicit.
"""

import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from easyrag.domain.answers import Answer
from easyrag.ingestion.indexer import IndexReport

CONTENT_TYPE = CONTENT_TYPE_LATEST

# Seconds; spans fast API calls up to CPU-bound answer generation.
_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_INDEX_BUCKETS = (0.1, 0.5, 1, 5, 15, 30, 60, 120, 300, 600, 1800)
# Ranking scores: cosine and reranker probability (0-1) and RRF (~0.01-0.033).
_SCORE_BUCKETS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                  0.7, 0.8, 0.9, 1.0)  # fmt: skip
_TOKEN_BUCKETS = (64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384)
_COUNT_BUCKETS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.build_info = Gauge(
            "easyrag_build_info",
            "EasyRAG version and configured models (value is always 1)",
            ["version", "embedding_model", "llm_model"],
            registry=r,
        )
        self.http_requests = Counter(
            "easyrag_http_requests_total",
            "HTTP requests by method, route template and status code",
            ["method", "route", "status"],
            registry=r,
        )
        self.http_duration = Histogram(
            "easyrag_http_request_duration_seconds",
            "HTTP request duration by method and route template",
            ["method", "route"],
            buckets=_HTTP_BUCKETS,
            registry=r,
        )
        self.api_errors = Counter(
            "easyrag_api_errors_total", "API error responses by error code", ["code"], registry=r
        )
        self.index_runs = Counter(
            "easyrag_index_runs_total", "Indexing runs by result", ["result"], registry=r
        )
        self.index_documents = Counter(
            "easyrag_index_documents_total",
            "Documents processed by indexing runs, by change",
            ["change"],
            registry=r,
        )
        self.index_chunks = Counter(
            "easyrag_index_chunks_total",
            "Chunks written by indexing runs, by vector source",
            ["source"],
            registry=r,
        )
        self.index_duration = Histogram(
            "easyrag_index_run_duration_seconds",
            "Duration of successful indexing runs",
            buckets=_INDEX_BUCKETS,
            registry=r,
        )
        self.index_last_success = Gauge(
            "easyrag_index_last_success_timestamp_seconds",
            "Unix time of the last successful indexing run",
            registry=r,
        )
        self.indexed_documents = Gauge(
            "easyrag_indexed_documents", "Documents currently in the index", registry=r
        )
        self.indexed_chunks = Gauge(
            "easyrag_indexed_chunks", "Chunks currently in the index", registry=r
        )

        # Question answering (recorded from each Answer's RagTrace).
        self.rag_answers = Counter(
            "easyrag_rag_answers_total", "Answers by pipeline outcome", ["outcome"], registry=r
        )
        self.rag_stage_duration = Histogram(
            "easyrag_rag_stage_duration_seconds",
            "Time spent per RAG stage (retrieval, generation, total)",
            ["stage"],
            buckets=_HTTP_BUCKETS,
            registry=r,
        )
        self.llm_tokens = Counter(
            "easyrag_llm_tokens_total",
            "Tokens processed by the chat model, as reported by the backend",
            ["kind"],
            registry=r,
        )
        self.rag_prompt_tokens = Histogram(
            "easyrag_rag_prompt_tokens",
            "Prompt size per generation (backend count, else the estimate)",
            buckets=_TOKEN_BUCKETS,
            registry=r,
        )
        self.rag_chunks = Histogram(
            "easyrag_rag_chunks",
            "Chunks per answer: retrieved, included in the prompt, cited",
            ["stage"],
            buckets=_COUNT_BUCKETS,
            registry=r,
        )
        self.retrieval_top_score = Histogram(
            "easyrag_retrieval_top_score",
            "Score of the best retrieved chunk per question, by retrieval configuration",
            ["config"],
            buckets=_SCORE_BUCKETS,
            registry=r,
        )
        self.rag_citations = Counter(
            "easyrag_rag_cited_answers_total",
            "Answers with sources, by how sources were determined (explicit / inferred)",
            ["attribution"],
            registry=r,
        )
        self.rag_invalid_citations = Counter(
            "easyrag_rag_invalid_citations_total",
            "Citation markers dropped because they referred to no source",
            registry=r,
        )
        self.rag_truncated = Counter(
            "easyrag_rag_truncated_answers_total",
            "Answers cut off by the output-token limit",
            registry=r,
        )
        # Model backend (Ollama) HTTP calls, per attempt.
        self.provider_duration = Histogram(
            "easyrag_provider_request_duration_seconds",
            "Model backend requests by endpoint and outcome (status code, timeout, "
            "connect_error, error)",
            ["operation", "outcome"],
            buckets=_HTTP_BUCKETS,
            registry=r,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def record_index_run(self, report: IndexReport) -> None:
        self.index_runs.labels(result="succeeded").inc()
        for change, count in (
            ("added", report.documents_added),
            ("updated", report.documents_updated),
            ("unchanged", report.documents_unchanged),
            ("deleted", report.documents_deleted),
            ("failed", len(report.failed)),
            ("skipped", len(report.skipped)),
        ):
            self.index_documents.labels(change=change).inc(count)
        self.index_chunks.labels(source="embedded").inc(report.chunks_embedded)
        self.index_chunks.labels(source="reused").inc(report.chunks_reused)
        self.index_duration.observe(report.duration_seconds)
        self.index_last_success.set(time.time())

    def record_index_failure(self) -> None:
        self.index_runs.labels(result="failed").inc()

    def record_index_size(self, documents: int, chunks: int) -> None:
        self.indexed_documents.set(documents)
        self.indexed_chunks.set(chunks)

    def record_answer(self, answer: Answer, *, retrieval_config: str) -> None:
        trace = answer.trace
        self.rag_answers.labels(outcome=trace.outcome).inc()
        self.rag_stage_duration.labels(stage="retrieval").observe(trace.retrieval_seconds)
        self.rag_stage_duration.labels(stage="total").observe(trace.total_seconds)
        if trace.llm_model is not None:  # the model was called
            self.rag_stage_duration.labels(stage="generation").observe(trace.generation_seconds)
            self.rag_prompt_tokens.observe(trace.prompt_tokens or trace.prompt_tokens_estimate)
            if trace.prompt_tokens is not None:
                self.llm_tokens.labels(kind="prompt").inc(trace.prompt_tokens)
            if trace.completion_tokens is not None:
                self.llm_tokens.labels(kind="completion").inc(trace.completion_tokens)
        self.rag_chunks.labels(stage="retrieved").observe(len(trace.retrieved))
        self.rag_chunks.labels(stage="included").observe(len(trace.included_chunk_ids))
        self.rag_chunks.labels(stage="cited").observe(len(trace.cited_chunk_ids))
        if trace.retrieved:
            self.retrieval_top_score.labels(config=retrieval_config).observe(trace.retrieved[0][1])
        if trace.attribution is not None:
            self.rag_citations.labels(attribution=trace.attribution).inc()
        self.rag_invalid_citations.inc(trace.invalid_citations)
        if answer.truncated:
            self.rag_truncated.inc()

    def record_provider_request(self, operation: str, outcome: str, seconds: float) -> None:
        self.provider_duration.labels(operation=operation, outcome=outcome).observe(seconds)
