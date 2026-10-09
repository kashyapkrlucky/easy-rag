"""Indexing: docs directory → parse → chunk → embed → store.

Phase 9 scope: a full, idempotent run. Every run re-embeds and re-stores every loaded
document; each document is replaced atomically, so re-running yields identical rows.
Incremental behaviour (skipping unchanged documents, reusing embeddings, deleting
removed files, an advisory lock) arrives in Phase 15.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from easyrag.domain.chunks import Chunk
from easyrag.domain.documents import ParsedDocument, SkippedFile
from easyrag.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingProviderUnavailableError,
    Vector,
)
from easyrag.ingestion.chunker import CHUNKER_VERSION, Chunker
from easyrag.ingestion.loader import DocumentLoader
from easyrag.ingestion.markdown import PARSER_VERSION, MarkdownParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentToStore:
    document: ParsedDocument
    size_bytes: int
    pipeline_fingerprint: str
    chunks: tuple[Chunk, ...]
    embeddings: tuple[Vector, ...]
    embedding_model: str


class IndexStore(Protocol):
    async def prepare(self, embedding_model: str, dimension: int) -> None:
        """Record or verify the index's embedding model and dimension; create vector indexes.

        Raises ``IndexIncompatibleError`` if the index was built differently.
        """
        ...

    async def save_document(self, document: DocumentToStore) -> None:
        """Atomically insert or replace one document and all of its chunks."""
        ...

    async def reset(self) -> None:
        """Delete every document, chunk and recorded embedding setting."""
        ...


@dataclass(frozen=True, slots=True)
class FailedDocument:
    source_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class IndexReport:
    documents_indexed: int
    chunks_indexed: int
    failed: tuple[FailedDocument, ...] = field(default=())
    skipped: tuple[SkippedFile, ...] = field(default=())
    duration_seconds: float = 0.0


LoaderFactory = Callable[[], DocumentLoader]


def pipeline_fingerprint(
    *, max_tokens: int, overlap_tokens: int, embedding_model: str, dimension: int
) -> str:
    """Identifies everything besides content that determines a document's stored chunks.

    A document whose stored fingerprint differs must be re-indexed (ARCHITECTURE D8).
    """
    settings = {
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "chunk_max_tokens": max_tokens,
        "chunk_overlap_tokens": overlap_tokens,
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


class Indexer:
    def __init__(
        self,
        *,
        loader_factory: LoaderFactory,
        parser: MarkdownParser,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        store: IndexStore,
    ) -> None:
        # A factory, so that a missing docs directory fails the index run, not app startup.
        self._loader_factory = loader_factory
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store

    async def run(self, *, rebuild: bool = False) -> IndexReport:
        """Index every document. ``rebuild`` first deletes the existing index.

        Raises ``IndexIncompatibleError`` (before any work) when the index was built with
        another embedding model, and ``EmbeddingProviderUnavailableError`` when the model
        cannot be reached (documents already saved stay saved).
        """
        started = time.perf_counter()
        model_id, dimension = self._embeddings.model_id, self._embeddings.dimension

        if rebuild:
            logger.info("Rebuild requested: deleting the existing index")
            await self._store.reset()
        await self._store.prepare(model_id, dimension)

        loaded = self._loader_factory().load()
        fingerprint = pipeline_fingerprint(
            max_tokens=self._chunker.config.max_tokens,
            overlap_tokens=self._chunker.config.overlap_tokens,
            embedding_model=model_id,
            dimension=dimension,
        )
        logger.info(
            "Indexing %d documents (%d skipped) with %s",
            len(loaded.documents),
            len(loaded.skipped),
            model_id,
        )

        indexed = chunk_count = 0
        failed: list[FailedDocument] = []
        for source in loaded.documents:
            parsed = self._parser.parse(source)
            chunks = self._chunker.chunk(parsed)
            try:
                vectors = await self._embeddings.embed_documents([c.embed_text for c in chunks])
            except EmbeddingProviderUnavailableError:
                raise  # every remaining document would fail the same way
            except EmbeddingError as exc:
                logger.warning("Failed to embed %s: %s", source.source_path, exc)
                failed.append(FailedDocument(source.source_path, str(exc)))
                continue

            await self._store.save_document(
                DocumentToStore(
                    document=parsed,
                    size_bytes=source.size_bytes,
                    pipeline_fingerprint=fingerprint,
                    chunks=chunks,
                    embeddings=tuple(vectors),
                    embedding_model=model_id,
                )
            )
            indexed += 1
            chunk_count += len(chunks)
            logger.debug("Indexed %s (%d chunks)", source.source_path, len(chunks))

        report = IndexReport(
            documents_indexed=indexed,
            chunks_indexed=chunk_count,
            failed=tuple(failed),
            skipped=loaded.skipped,
            duration_seconds=time.perf_counter() - started,
        )
        logger.info(
            "Indexed %d documents (%d chunks) in %.1fs; %d failed, %d skipped",
            report.documents_indexed,
            report.chunks_indexed,
            report.duration_seconds,
            len(report.failed),
            len(report.skipped),
        )
        return report
