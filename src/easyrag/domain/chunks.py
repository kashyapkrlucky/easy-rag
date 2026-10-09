"""Chunks: the unit of embedding, retrieval and citation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    # Deterministic 32-hex-character ID (see easyrag.ingestion.chunker.chunk_id).
    id: str
    source_path: str
    # Position of the chunk within its document, starting at 0.
    ordinal: int
    heading_path: tuple[str, ...]
    section_anchor: str
    # Verbatim source text (lines start_line..end_line of the file); shown to the LLM.
    content: str
    # Contextual header + content; what the embedding model receives.
    embed_text: str
    # SHA-256 (lowercase hex) of ``content``; used to reuse embeddings of unchanged text.
    content_hash: str
    # 1-based, inclusive line range of ``content`` in the source file.
    start_line: int
    end_line: int
    # estimate_tokens(embed_text)
    token_estimate: int


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with what citations need."""

    chunk_id: str
    source_path: str
    document_title: str
    heading_path: tuple[str, ...]
    section_anchor: str
    content: str
    start_line: int
    end_line: int
    # Ranking score of the retriever that produced this result (higher is better):
    # cosine similarity (vector), matched IDF share in [0, 1] (keyword), or the
    # Reciprocal Rank Fusion score (hybrid). Compare scores only within one mode.
    score: float
    # Component scores, when the chunk was found by that method.
    vector_score: float | None = None
    keyword_score: float | None = None

    @property
    def section(self) -> str:
        """Human-readable section name: the deepest heading, or the title for a preamble."""
        return self.heading_path[-1] if self.heading_path else self.document_title
