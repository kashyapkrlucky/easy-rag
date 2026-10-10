"""The shipped examples stay usable: the corpus indexes, the README's expected answers are
in it, and the clients are valid programs."""

import py_compile
import shutil
import subprocess
from pathlib import Path

import pytest

from easyrag.ingestion.chunker import Chunker
from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser
from easyrag.injection import contains_instructions

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_example_corpus_loads_and_chunks() -> None:
    loaded = FilesystemMarkdownLoader(EXAMPLES / "docs").load()
    parser, chunker = MarkdownParser(), Chunker()

    paths = sorted(document.source_path for document in loaded.documents)
    sections = {
        chunk.section_anchor
        for document in loaded.documents
        for chunk in chunker.chunk(parser.parse(document))
    }

    assert paths == [
        "deployment.md", "faq.md", "getting-started.md", "runbooks/database-failover.md",
    ]  # fmt: skip
    assert loaded.skipped == ()
    assert {"rollback", "configuration", "why-was-my-sms-not-delivered", "steps"} <= sections


@pytest.mark.parametrize(
    "fact",
    [
        "make rollback ENV=production",
        "Tuesdays and Thursdays",
        "quiet hours",
        "Five times",
        "above 60 seconds",
        "`22:00-07:00`",
        "`twilio` or `vonage`",
    ],
)
def test_readme_answers_are_in_the_corpus(fact: str) -> None:
    corpus = " ".join(
        " ".join(path.read_text(encoding="utf-8").split())
        for path in (EXAMPLES / "docs").rglob("*.md")
    )

    assert fact.lower() in corpus.lower()


def test_example_corpus_has_no_instruction_like_text() -> None:
    for path in (EXAMPLES / "docs").rglob("*.md"):
        assert not contains_instructions(path.read_text(encoding="utf-8")), path.name


def test_python_client_compiles(tmp_path: Path) -> None:
    py_compile.compile(
        str(EXAMPLES / "clients" / "ask.py"), cfile=str(tmp_path / "ask.pyc"), doraise=True
    )


def test_shell_client_is_valid_posix_sh() -> None:
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("no sh")
    result = subprocess.run(
        [sh, "-n", str(EXAMPLES / "clients" / "ask.sh")], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
