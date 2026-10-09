"""Request and response models of the public HTTP API (v1)."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from easyrag.domain.answers import Answer
from easyrag.domain.chunks import RetrievedChunk
from easyrag.ingestion.coordinator import IndexStatus
from easyrag.ingestion.watcher import WatcherState
from easyrag.retrieval.vector import MAX_TOP_K
from easyrag.storage.index_store import IndexedDocument

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


RetrievalMode = Literal["vector", "keyword", "hybrid"]


class RetrieveRequest(BaseModel):
    query: Query
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    min_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Cosine threshold for vector results"
    )
    mode: RetrievalMode | None = Field(
        default=None, description="Override EASYRAG_RETRIEVAL_MODE for this request"
    )


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document: str = Field(description="Path of the source document, relative to the docs root")
    title: str = Field(description="Document title")
    section: str = Field(description="Deepest heading, or the document title for a preamble")
    heading_path: list[str]
    anchor: str = Field(description="URL fragment of the section ('' for a preamble)")
    lines: tuple[int, int] = Field(description="1-based, inclusive line range in the document")
    score: float = Field(
        description="Ranking score of the mode used (cosine / keyword share / RRF); "
        "compare only within one mode"
    )
    vector_score: float | None = Field(description="Cosine similarity, if found by vector")
    keyword_score: float | None = Field(description="Matched IDF share, if found by keyword")
    content: str

    @classmethod
    def from_domain(cls, chunk: RetrievedChunk) -> "RetrievedChunkResponse":
        return cls(
            chunk_id=chunk.chunk_id,
            document=chunk.source_path,
            title=chunk.document_title,
            section=chunk.section,
            heading_path=list(chunk.heading_path),
            anchor=chunk.section_anchor,
            lines=(chunk.start_line, chunk.end_line),
            score=round(chunk.score, 6),
            vector_score=None if chunk.vector_score is None else round(chunk.vector_score, 6),
            keyword_score=None if chunk.keyword_score is None else round(chunk.keyword_score, 6),
            content=chunk.content,
        )


class RetrieveResponse(BaseModel):
    query: str
    mode: RetrievalMode
    embedding_model: str
    results: list[RetrievedChunkResponse]


class ChatRequest(BaseModel):
    question: Query
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class SourceResponse(BaseModel):
    document: str = Field(description="Path of the source document, relative to the docs root")
    title: str = Field(description="Document title")
    section: str = Field(description="Deepest heading, or the document title for a preamble")
    heading_path: list[str]
    anchor: str = Field(description="URL fragment of the section ('' for a preamble)")
    lines: tuple[int, int] = Field(description="1-based, inclusive line range in the document")
    score: float = Field(description="Retrieval ranking score (see /retrieve)")
    chunk_id: str

    @classmethod
    def from_domain(cls, chunk: RetrievedChunk) -> "SourceResponse":
        return cls(
            document=chunk.source_path,
            title=chunk.document_title,
            section=chunk.section,
            heading_path=list(chunk.heading_path),
            anchor=chunk.section_anchor,
            lines=(chunk.start_line, chunk.end_line),
            score=round(chunk.score, 6),
            chunk_id=chunk.chunk_id,
        )


class ChatResponse(BaseModel):
    answer: str = Field(description='Answer text; "[k]" cites sources[k - 1]')
    status: Literal["answered", "insufficient_evidence", "ungrounded"] = Field(
        description="answered: cites the documents; insufficient_evidence: the documents do "
        "not answer the question; ungrounded: the answer cites no source (unverified)"
    )
    grounded: bool = Field(description="False only when status is 'ungrounded'")
    truncated: bool = Field(description="True if the answer was cut off by the output limit")
    sources: list[SourceResponse] = Field(description="The cited document sections, in order")

    @classmethod
    def from_domain(cls, answer: Answer) -> "ChatResponse":
        return cls(
            answer=answer.text,
            status=answer.status,
            grounded=answer.grounded,
            truncated=answer.truncated,
            sources=[SourceResponse.from_domain(chunk) for chunk in answer.sources],
        )


class IndexReportResponse(BaseModel):
    documents_added: int
    documents_updated: int
    documents_unchanged: int
    documents_deleted: int
    chunks_embedded: int
    chunks_reused: int
    failed: list[dict[str, str]] = Field(description="[{document, reason}]")
    skipped: list[dict[str, str]] = Field(description="[{document, reason}]")
    deletion_guarded: bool = Field(
        description="True if removed documents were kept because the docs directory was empty"
    )
    duration_seconds: float


class IndexStatusResponse(BaseModel):
    state: Literal["idle", "running", "succeeded", "failed"]
    started_at: datetime | None
    finished_at: datetime | None
    last_error: str | None
    attempts: int = Field(description="Consecutive failed startup attempts while waiting")
    pending: bool = Field(description="Another run will follow the current one")
    watcher: Literal["disabled", "starting", "watching", "waiting_for_directory", "stopped"]
    last_report: IndexReportResponse | None
    documents: int = Field(description="Documents currently in the index")
    chunks: int = Field(description="Chunks currently in the index")

    @classmethod
    def from_domain(
        cls, status: IndexStatus, documents: int, chunks: int, watcher: WatcherState
    ) -> "IndexStatusResponse":
        report = status.last_report
        return cls(
            state=status.state,
            started_at=status.started_at,
            finished_at=status.finished_at,
            last_error=status.last_error,
            attempts=status.attempts,
            pending=status.pending,
            watcher=watcher,
            documents=documents,
            chunks=chunks,
            last_report=None
            if report is None
            else IndexReportResponse(
                documents_added=report.documents_added,
                documents_updated=report.documents_updated,
                documents_unchanged=report.documents_unchanged,
                documents_deleted=report.documents_deleted,
                chunks_embedded=report.chunks_embedded,
                chunks_reused=report.chunks_reused,
                failed=[{"document": f.source_path, "reason": f.reason} for f in report.failed],
                skipped=[{"document": f.source_path, "reason": f.reason} for f in report.skipped],
                deletion_guarded=report.deletion_guarded,
                duration_seconds=round(report.duration_seconds, 3),
            ),
        )


class IndexTriggerResponse(BaseModel):
    status: Literal["started"] = "started"


class DocumentResponse(BaseModel):
    document: str
    title: str
    content_hash: str
    size_bytes: int
    chunks: int
    indexed_at: datetime

    @classmethod
    def from_domain(cls, document: IndexedDocument) -> "DocumentResponse":
        return cls(
            document=document.source_path,
            title=document.title,
            content_hash=document.content_hash,
            size_bytes=document.size_bytes,
            chunks=document.chunk_count,
            indexed_at=document.indexed_at,
        )


class DocumentsResponse(BaseModel):
    documents: list[DocumentResponse]
