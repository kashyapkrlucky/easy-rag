from pathlib import Path
from textwrap import dedent

import pytest

from easyrag.evaluation.dataset import DatasetError, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_dataset(tmp_path: Path, questions_yaml: str) -> Path:
    root = tmp_path / "ds"
    (root / "corpus").mkdir(parents=True)
    (root / "corpus" / "guide.md").write_text("# Guide\n\n## Install\n\nRun it.\n")
    (root / "questions.yaml").write_text(dedent(questions_yaml))
    return root


VALID = """
    name: tiny
    questions:
      - id: install
        kind: factual
        question: How do I install it?
        expected: [{document: guide.md, section: Install}]
        facts: ["run it", ["execute", "run"]]
      - id: price
        kind: unanswerable
        question: What does it cost?
"""


def test_valid_dataset_loads_with_normalised_facts(tmp_path: Path) -> None:
    dataset = load_dataset(make_dataset(tmp_path, VALID))

    install, price = dataset.questions
    assert dataset.name == "tiny"
    assert install.facts == [["run it"], ["execute", "run"]]
    assert install.answerable and not price.answerable


def test_the_shipped_nimbus_dataset_is_valid() -> None:
    dataset = load_dataset(PROJECT_ROOT / "eval" / "datasets" / "nimbus")

    assert len(dataset.questions) == 30
    assert {q.kind for q in dataset.questions} == {
        "factual", "paraphrase", "identifier", "unanswerable", "injection"
    }  # fmt: skip


@pytest.mark.parametrize(
    ("questions_yaml", "message"),
    [
        (VALID.replace("guide.md", "missing.md"), "unknown document 'missing.md'"),
        (VALID.replace("section: Install", "section: Uninstall"), "no section matching"),
        (VALID.replace("kind: unanswerable", "kind: factual"), "need `expected` and `facts`"),
        (VALID.replace("id: price", "id: install"), "duplicate question ids"),
        (VALID.replace("id: install", "id: Not Valid"), "id"),
        (VALID.replace("kind: factual", "kind: injection"), "`injected` is required"),
        ("name: x\nquestions: []\n", "questions"),
        ("name: [unclosed", "questions.yaml"),
    ],
)
def test_invalid_datasets_are_rejected(tmp_path: Path, questions_yaml: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_dataset(make_dataset(tmp_path, questions_yaml))


def test_missing_layout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"corpus/ and questions\.yaml"):
        load_dataset(tmp_path)
