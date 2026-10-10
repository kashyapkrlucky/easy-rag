"""Prometheus metrics, held in a registry owned by one application instance.

A per-app ``CollectorRegistry`` (instead of prometheus_client's global one) keeps tests
and multiple app instances independent and makes the dependency explicit.
"""

import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from easyrag.ingestion.indexer import IndexReport

CONTENT_TYPE = CONTENT_TYPE_LATEST

# Seconds; spans fast API calls up to CPU-bound answer generation.
_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_INDEX_BUCKETS = (0.1, 0.5, 1, 5, 15, 30, 60, 120, 300, 600, 1800)


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
