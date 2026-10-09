"""Incremental indexing: docs directory → parse → chunk → embed → store (ARCHITECTURE D8-D10).

A run:
1. verifies the index matches the configured embedding model (``prepare``);
2. takes the store's writer lock, so runs from the API, CLI and watcher never interleave;
3. skips documents whose content hash *and* pipeline fingerprint are unchanged;
4. re-indexes the rest, reusing stored vectors whose embedding input is identical and
   embedding only new inputs;
5. deletes documents whose files are gone — never files the loader merely skipped, and
   never every document just because the docs directory came back empty.
"""

import dataclasses
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
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
from easyrag.ingestion.hashing import content_hash
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


@dataclass(frozen=True, slots=True)
class StoredDocumentState:
    content_hash: str
    pipeline_fingerprint: str


class IndexStore(Protocol):
    async def prepare(self, embedding_model: str, dimension: int) -> None:
        """Record or verify the index's embedding model and dimension; create vector indexes.

        Raises ``IndexIncompatibleError`` if the index was built differently.
        """
        ...

    def lock(self) -> AbstractAsyncContextManager[None]:
        """Exclusive writer lock for the duration of a run (waits for other writers)."""
        ...

    async def document_states(self) -> Mapping[str, StoredDocumentState]:
        """Indexed documents by source path."""
        ...

    async def find_embeddings(
        self, input_hashes: Sequence[str], embedding_model: str
    ) -> Mapping[str, Vector]:
        """Stored vectors for these embedding-input hashes, produced by ``embedding_model``."""
        ...

    async def save_document(self, document: DocumentToStore) -> None:
        """Atomically insert or replace one document and all of its chunks."""
        ...

    async def delete_documents(self, source_paths: Sequence[str]) -> None:
        """Delete documents (and, by cascade, their chunks)."""
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
    documents_added: int = 0
    documents_updated: int = 0
    documents_unchanged: int = 0
    documents_deleted: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    failed: tuple[FailedDocument, ...] = field(default=())
    skipped: tuple[SkippedFile, ...] = field(default=())
    # True if removed documents were *not* deleted because the docs directory was empty.
    deletion_guarded: bool = False
    duration_seconds: float = 0.0

    @property
    def documents_indexed(self) -> int:
        return self.documents_added + self.documents_updated


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


def embedding_input_hash(chunk: Chunk) -> str:
    """Key for vector reuse: the exact text the embedding model receives."""
    return content_hash(chunk.embed_text)


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
        """Bring the index in line with the docs directory. ``rebuild`` first empties it.

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

        async with self._store.lock():
            report = await self._run_locked(model_id, dimension)

        report = dataclasses.replace(report, duration_seconds=time.perf_counter() - started)
        logger.info(
            "Index run: %d added, %d updated, %d unchanged, %d deleted; %d chunks embedded, "
            "%d reused; %d failed, %d skipped%s (%.1fs)",
            report.documents_added,
            report.documents_updated,
            report.documents_unchanged,
            report.documents_deleted,
            report.chunks_embedded,
            report.chunks_reused,
            len(report.failed),
            len(report.skipped),
            "; deletions withheld (empty docs directory)" if report.deletion_guarded else "",
            report.duration_seconds,
        )
        return report

    async def _run_locked(self, model_id: str, dimension: int) -> IndexReport:
        loaded = self._loader_factory().load()
        stored = await self._store.document_states()
        fingerprint = pipeline_fingerprint(
            max_tokens=self._chunker.config.max_tokens,
            overlap_tokens=self._chunker.config.overlap_tokens,
            embedding_model=model_id,
            dimension=dimension,
        )

        added = updated = unchanged = embedded = reused = 0
        failed: list[FailedDocument] = []
        for source in loaded.documents:
            previous = stored.get(source.source_path)
            if previous == StoredDocumentState(source.content_hash, fingerprint):
                unchanged += 1
                continue

            parsed = self._parser.parse(source)
            chunks = self._chunker.chunk(parsed)
            try:
                vectors, new = await self._vectors(chunks, model_id)
            except EmbeddingProviderUnavailableError:
                raise  # every remaining document would fail the same way
            except EmbeddingError as exc:
                logger.warning("Failed to embed %s: %s", source.source_path, exc)
                failed.append(FailedDocument(source.source_path, str(exc)))
                continue  # the previously indexed version, if any, stays

            await self._store.save_document(
                DocumentToStore(
                    document=parsed,
                    size_bytes=source.size_bytes,
                    pipeline_fingerprint=fingerprint,
                    chunks=chunks,
                    embeddings=vectors,
                    embedding_model=model_id,
                )
            )
            if previous is None:
                added += 1
            else:
                updated += 1
            embedded += new
            reused += len(chunks) - new

        present = {d.source_path for d in loaded.documents}
        present |= {s.source_path for s in loaded.skipped}
        removed = sorted(set(stored) - present)
        guarded = bool(removed) and not present  # empty docs directory: likely a mount mistake
        if guarded:
            logger.warning(
                "The docs directory is empty but %d documents are indexed: not deleting them. "
                "Check EASYRAG_DOCS_DIR / the mount, or run `easyrag index --rebuild`.",
                len(removed),
            )
        elif removed:
            await self._store.delete_documents(removed)

        return IndexReport(
            documents_added=added,
            documents_updated=updated,
            documents_unchanged=unchanged,
            documents_deleted=0 if guarded else len(removed),
            chunks_embedded=embedded,
            chunks_reused=reused,
            failed=tuple(failed),
            skipped=loaded.skipped,
            deletion_guarded=guarded,
        )

    async def _vectors(
        self, chunks: Sequence[Chunk], model_id: str
    ) -> tuple[tuple[Vector, ...], int]:
        """Vectors for ``chunks`` (reused where possible) and how many were newly embedded."""
        if not chunks:
            return (), 0
        hashes = [embedding_input_hash(chunk) for chunk in chunks]
        known = dict(await self._store.find_embeddings(sorted(set(hashes)), model_id))
        texts = {hash_: chunk.embed_text for hash_, chunk in zip(hashes, chunks, strict=True)}
        missing = sorted(set(hashes) - set(known))
        if missing:
            vectors = await self._embeddings.embed_documents([texts[h] for h in missing])
            known.update(zip(missing, vectors, strict=True))
        return tuple(known[h] for h in hashes), len(missing)
