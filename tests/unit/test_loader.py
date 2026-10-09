import os
import unicodedata
from pathlib import Path

import pytest

from easyrag.ingestion.hashing import content_hash
from easyrag.ingestion.loader import FilesystemMarkdownLoader


def write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    return root


def test_discovers_markdown_recursively_in_sorted_order(docs: Path) -> None:
    write(docs, "zeta.md", "# Z")
    write(docs, "guides/deployment.md", "# D")
    write(docs, "guides/api.MARKDOWN", "# A")
    write(docs, "alpha.md", "# A")

    result = FilesystemMarkdownLoader(docs).load()

    assert [d.source_path for d in result.documents] == [
        "alpha.md",
        "guides/api.MARKDOWN",
        "guides/deployment.md",
        "zeta.md",
    ]
    assert result.skipped == ()


def test_ignores_non_markdown_and_hidden_entries(docs: Path) -> None:
    write(docs, "notes.txt", "text")
    write(docs, "image.png", b"\x89PNG")
    write(docs, ".hidden.md", "# Hidden")
    write(docs, ".obsidian/workspace.md", "# Config")
    write(docs, "visible.md", "# Visible")

    result = FilesystemMarkdownLoader(docs).load()

    assert [d.source_path for d in result.documents] == ["visible.md"]


def test_document_text_is_normalised_and_hashed(docs: Path) -> None:
    raw = "\ufeff# Title\r\nBody\r\n".encode()
    write(docs, "a.md", raw)

    (document,) = FilesystemMarkdownLoader(docs).load().documents

    assert document.text == "# Title\nBody\n"
    assert document.content_hash == content_hash("# Title\nBody\n")
    assert document.size_bytes == len(raw)


def test_loading_is_deterministic(docs: Path) -> None:
    write(docs, "a.md", "# A\n\ntext")
    write(docs, "b/c.md", "# C\n\ntext")
    loader = FilesystemMarkdownLoader(docs)

    assert loader.load() == loader.load()


def test_invalid_utf8_is_skipped_with_reason(docs: Path) -> None:
    write(docs, "good.md", "# Good")
    write(docs, "bad.md", b"# Bad \xff\xfe")

    result = FilesystemMarkdownLoader(docs).load()

    assert [d.source_path for d in result.documents] == ["good.md"]
    assert [(s.source_path, s.reason) for s in result.skipped] == [("bad.md", "not valid UTF-8")]


def test_oversized_file_is_skipped(docs: Path) -> None:
    write(docs, "big.md", "x" * 101)
    write(docs, "small.md", "x" * 100)

    result = FilesystemMarkdownLoader(docs, max_bytes=100).load()

    assert [d.source_path for d in result.documents] == ["small.md"]
    assert result.skipped[0].source_path == "big.md"
    assert "larger than 100 bytes" in result.skipped[0].reason


def test_symlink_inside_root_is_followed(docs: Path) -> None:
    target = write(docs, "real.md", "# Real")
    (docs / "alias.md").symlink_to(target)

    result = FilesystemMarkdownLoader(docs).load()

    assert [d.source_path for d in result.documents] == ["alias.md", "real.md"]


def test_symlink_escaping_root_is_skipped(docs: Path, tmp_path: Path) -> None:
    outside = write(tmp_path, "secret.md", "# Outside the docs root")
    (docs / "escape.md").symlink_to(outside)

    result = FilesystemMarkdownLoader(docs).load()

    assert result.documents == ()
    assert [(s.source_path, s.reason) for s in result.skipped] == [
        ("escape.md", "symlink points outside the docs directory")
    ]


def test_broken_symlink_is_skipped(docs: Path) -> None:
    (docs / "dangling.md").symlink_to(docs / "missing.md")

    result = FilesystemMarkdownLoader(docs).load()

    assert [(s.source_path, s.reason) for s in result.skipped] == [
        ("dangling.md", "broken symlink")
    ]


def test_symlinked_directories_are_not_traversed(docs: Path, tmp_path: Path) -> None:
    write(tmp_path, "elsewhere/outside.md", "# Outside")
    (docs / "linked").symlink_to(tmp_path / "elsewhere", target_is_directory=True)

    assert FilesystemMarkdownLoader(docs).load().documents == ()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_unreadable_file_is_skipped(docs: Path) -> None:
    locked = write(docs, "locked.md", "# Locked")
    locked.chmod(0)
    try:
        result = FilesystemMarkdownLoader(docs).load()
    finally:
        locked.chmod(0o644)

    assert result.skipped[0].source_path == "locked.md"
    assert result.skipped[0].reason.startswith("not readable")


def test_source_paths_are_nfc_normalised(docs: Path) -> None:
    write(docs, unicodedata.normalize("NFD", "café.md"), "# Café")

    (document,) = FilesystemMarkdownLoader(docs).load().documents

    assert document.source_path == unicodedata.normalize("NFC", "café.md")


def test_missing_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        FilesystemMarkdownLoader(tmp_path / "does-not-exist")
