from collections.abc import Sequence

import pytest

from easyrag.domain.chunks import RetrievedChunk
from easyrag.retrieval.keyword import MAX_TERMS, KeywordRetriever, keyword_terms


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How do I set EASYRAG_EMBEDDING_MODEL?", ("easyrag_embedding_model", "set")),
        ("What does nomic-embed-text:v1.5 produce?", ("nomic-embed-text:v1.5", "produce")),
        ("Is v1.5 supported on port 8000?", ("v1.5", "supported", "port", "8000")),
        ("Where is /api/v1/chat documented?", ("api/v1/chat", "documented")),
        ("What is the escalation policy?", ("escalation", "policy")),
        ("Docker docker DOCKER", ("docker",)),  # case-insensitive de-duplication
        ("How do I do it?", ()),  # only stopwords
        ("Erklärung der Einbettung", ("erklärung", "der", "einbettung")),  # non-English kept
    ],
)
def test_keyword_terms(query: str, expected: tuple[str, ...]) -> None:
    assert keyword_terms(query) == expected


def test_terms_are_capped() -> None:
    query = " ".join(f"word{i}" for i in range(40))

    assert len(keyword_terms(query)) == MAX_TERMS


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int]] = []

    async def search(self, terms: Sequence[str], *, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((tuple(terms), top_k))
        return []


async def test_retriever_passes_terms_and_top_k() -> None:
    search = RecordingSearch()

    await KeywordRetriever(search=search, default_top_k=7).retrieve("Set EASYRAG_DOCS_DIR")
    await KeywordRetriever(search=search).retrieve("docker", top_k=3, min_score=0.9)

    assert search.calls == [(("easyrag_docs_dir", "set"), 7), (("docker",), 3)]


async def test_stopword_only_queries_do_not_search() -> None:
    search = RecordingSearch()

    assert await KeywordRetriever(search=search).retrieve("how do I do it") == []
    assert search.calls == []


@pytest.mark.parametrize("top_k", [0, 51])
async def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError):
        await KeywordRetriever(search=RecordingSearch()).retrieve("docker", top_k=top_k)
