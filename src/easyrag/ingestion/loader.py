"""Document discovery and loading.

``FilesystemMarkdownLoader`` walks a docs root and returns every Markdown file as a
``SourceDocument`` with normalised text and a content hash. Files that cannot be
loaded are reported in ``LoadResult.skipped`` rather than aborting the whole run.
"""

import logging
import os
import unicodedata
from collections.abc import Iterator
from pathlib import Path, PurePath
from typing import Protocol

from easyrag.domain.documents import LoadResult, SkippedFile, SourceDocument
from easyrag.ingestion.hashing import content_hash, normalize_text

logger = logging.getLogger(__name__)

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
DEFAULT_MAX_BYTES = 5 * 1024 * 1024


class DocumentLoader(Protocol):
    def load(self) -> LoadResult: ...


class _SkipError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FilesystemMarkdownLoader:
    """Loads ``*.md`` / ``*.markdown`` files below ``root``.

    Rules:
    - hidden files and directories (name starts with ".") are ignored;
    - symlinked files are followed only if they resolve inside ``root``;
      symlinked directories are not traversed (avoids cycles and escapes);
    - files larger than ``max_bytes`` or not valid UTF-8 are skipped.

    Results are sorted by ``source_path`` so loading is deterministic.
    """

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        resolved = root.resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(f"Docs directory does not exist or is not a directory: {root}")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._root = resolved
        self._max_bytes = max_bytes

    @property
    def root(self) -> Path:
        return self._root

    def load(self) -> LoadResult:
        documents: list[SourceDocument] = []
        skipped: list[SkippedFile] = []

        for path in self._candidates(skipped):
            source_path = self._source_path(path)
            try:
                documents.append(self._read(path, source_path))
            except _SkipError as skip:
                logger.warning("Skipping %s: %s", source_path, skip.reason)
                skipped.append(SkippedFile(source_path=source_path, reason=skip.reason))

        documents.sort(key=lambda document: document.source_path)
        skipped.sort(key=lambda file: file.source_path)
        return LoadResult(documents=tuple(documents), skipped=tuple(skipped))

    def _candidates(self, skipped: list[SkippedFile]) -> Iterator[Path]:
        def on_error(error: OSError) -> None:
            source_path = self._source_path(Path(str(error.filename)))
            reason = f"directory not readable ({error.strerror})"
            logger.warning("Skipping %s: %s", source_path, reason)
            skipped.append(SkippedFile(source_path=source_path, reason=reason))

        for directory, subdirectories, filenames in os.walk(
            self._root, onerror=on_error, followlinks=False
        ):
            subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
            for name in filenames:
                if name.startswith(".") or PurePath(name).suffix.lower() not in MARKDOWN_SUFFIXES:
                    continue
                yield Path(directory) / name

    def _source_path(self, path: Path) -> str:
        relative = path.relative_to(self._root).as_posix()
        # macOS may report decomposed (NFD) names; one file must have one identity.
        return unicodedata.normalize("NFC", relative)

    def _read(self, path: Path, source_path: str) -> SourceDocument:
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except OSError as exc:
                raise _SkipError("broken symlink") from exc
            if not target.is_relative_to(self._root):
                raise _SkipError("symlink points outside the docs directory")

        try:
            if path.stat().st_size > self._max_bytes:
                raise _SkipError(f"larger than {self._max_bytes} bytes")
            raw = path.read_bytes()
        except OSError as exc:
            raise _SkipError(f"not readable ({exc.strerror})") from exc

        if len(raw) > self._max_bytes:  # the file grew between stat() and read
            raise _SkipError(f"larger than {self._max_bytes} bytes")

        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _SkipError("not valid UTF-8") from exc

        text = normalize_text(decoded)
        return SourceDocument(
            source_path=source_path,
            text=text,
            content_hash=content_hash(text),
            size_bytes=len(raw),
        )
