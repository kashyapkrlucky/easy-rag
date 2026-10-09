"""Request and response models of the public HTTP API (v1)."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from easyrag.domain.answers import Answer
from easyrag.domain.chunks import RetrievedChunk
from easyrag.retrieval.vector import MAX_TOP_K

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class RetrieveRequest(BaseModel):
    query: Query
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document: str = Field(description="Path of the source document, relative to the docs root")
    title: str = Field(description="Document title")
    section: str = Field(description="Deepest heading, or the document title for a preamble")
    heading_path: list[str]
    anchor: str = Field(description="URL fragment of the section ('' for a preamble)")
    lines: tuple[int, int] = Field(description="1-based, inclusive line range in the document")
    score: float = Field(description="Cosine similarity; higher is more similar")
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
            content=chunk.content,
        )


class RetrieveResponse(BaseModel):
    query: str
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
    score: float = Field(description="Retrieval cosine similarity")
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
    answer: str
    truncated: bool = Field(description="True if the answer was cut off by the output limit")
    sources: list[SourceResponse] = Field(description="The document sections given to the model")

    @classmethod
    def from_domain(cls, answer: Answer) -> "ChatResponse":
        return cls(
            answer=answer.text,
            truncated=answer.truncated,
            sources=[SourceResponse.from_domain(chunk) for chunk in answer.sources],
        )
