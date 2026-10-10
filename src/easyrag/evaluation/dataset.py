"""Evaluation datasets: a Markdown corpus plus labelled questions (``questions.yaml``).

Layout::

    <dataset>/
        corpus/**/*.md
        questions.yaml

Loading validates the labels against the corpus itself: every expected document must
exist and every expected section must match a real heading, so a typo in a label fails
loudly instead of silently scoring zero.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, Field, model_validator

from easyrag.ingestion.loader import FilesystemMarkdownLoader
from easyrag.ingestion.markdown import MarkdownParser

QuestionKind = Literal["factual", "paraphrase", "identifier", "unanswerable", "injection"]
ANSWERABLE_KINDS: frozenset[str] = frozenset({"factual", "paraphrase", "identifier", "injection"})


def _alternatives(value: object) -> object:
    """A fact may be written as one string or a list of acceptable alternatives."""
    if isinstance(value, list):
        return [item if isinstance(item, list) else [item] for item in value]
    return value


class ExpectedSource(BaseModel):
    document: str
    # Matches when it occurs in the chunk's heading path ("A > B > C").
    section: str


class EvalQuestion(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    kind: QuestionKind
    question: str = Field(min_length=3)
    expected: list[ExpectedSource] = Field(default_factory=list)
    facts: Annotated[list[list[str]], BeforeValidator(_alternatives)] = Field(default_factory=list)
    injected: str | None = None

    @property
    def answerable(self) -> bool:
        return self.kind in ANSWERABLE_KINDS

    @model_validator(mode="after")
    def _labels_match_kind(self) -> "EvalQuestion":
        if self.answerable and not (self.expected and self.facts):
            raise ValueError(f"{self.id}: answerable questions need `expected` and `facts`")
        if not self.answerable and (self.expected or self.facts):
            raise ValueError(f"{self.id}: unanswerable questions must not have labels")
        if (self.kind == "injection") != (self.injected is not None):
            raise ValueError(f"{self.id}: `injected` is required for (and only for) injection")
        if any(not alternatives or not all(alternatives) for alternatives in self.facts):
            raise ValueError(f"{self.id}: facts must be non-empty strings")
        return self


class DatasetFile(BaseModel):
    name: str
    description: str = ""
    questions: list[EvalQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "DatasetFile":
        ids = [q.id for q in self.questions]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate question ids: {duplicates}")
        return self


@dataclass(frozen=True, slots=True)
class EvalDataset:
    name: str
    description: str
    root: Path
    corpus_dir: Path
    questions: tuple[EvalQuestion, ...]


class DatasetError(ValueError):
    """The dataset is malformed or its labels do not match its corpus."""


def load_dataset(root: Path) -> EvalDataset:
    corpus_dir = root / "corpus"
    questions_file = root / "questions.yaml"
    if not corpus_dir.is_dir() or not questions_file.is_file():
        raise DatasetError(f"{root} must contain corpus/ and questions.yaml")
    try:
        parsed = DatasetFile.model_validate(yaml.safe_load(questions_file.read_text("utf-8")))
    except (yaml.YAMLError, ValueError) as exc:
        raise DatasetError(f"{questions_file}: {exc}") from exc

    _validate_labels(parsed.questions, corpus_dir)
    return EvalDataset(
        name=parsed.name,
        description=parsed.description.strip(),
        root=root,
        corpus_dir=corpus_dir,
        questions=tuple(parsed.questions),
    )


def _validate_labels(questions: list[EvalQuestion], corpus_dir: Path) -> None:
    parser = MarkdownParser()
    headings: dict[str, list[str]] = {}
    for document in FilesystemMarkdownLoader(corpus_dir).load().documents:
        parsed = parser.parse(document)
        headings[document.source_path] = [" > ".join(s.heading_path) for s in parsed.sections]

    problems = [
        f"{q.id}: no section matching {e.section!r} in {e.document}"
        if e.document in headings
        else f"{q.id}: unknown document {e.document!r}"
        for q in questions
        for e in q.expected
        if e.document not in headings or not any(e.section in h for h in headings[e.document])
    ]
    if problems:
        raise DatasetError("labels do not match the corpus:\n  " + "\n  ".join(problems))
