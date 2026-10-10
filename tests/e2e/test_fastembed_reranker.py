"""The real cross-encoder (downloads ~80 MB once). Opt-in: EASYRAG_TEST_RERANKER=1."""

import os
from pathlib import Path

import pytest

from easyrag.domain.chunks import RetrievedChunk
from easyrag.rerankers.fastembed import FastEmbedReranker

pytestmark = pytest.mark.skipif(
    not os.environ.get("EASYRAG_TEST_RERANKER"), reason="EASYRAG_TEST_RERANKER is not set"
)
CACHE = Path(os.environ.get("EASYRAG_TEST_MODEL_CACHE", ".cache/easyrag/models"))


def chunk(chunk_id: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, "d.md", "Docs", ("Docs",), "", content, 1, 1, 0.5)


async def test_real_cross_encoder_ranks_the_relevant_passage_first() -> None:
    reranker = FastEmbedReranker("Xenova/ms-marco-MiniLM-L-6-v2", CACHE)
    await reranker.load()
    assert reranker.state == "ready"

    results = await reranker.rerank(
        "How long does support take to respond?",
        [
            chunk("billing", "Invoices are sent on the first day of each month."),
            chunk("support", "Support requests are answered within one business day."),
            chunk("deploy", "Run docker compose up to start the stack."),
        ],
        top_n=3,
    )

    assert results[0].chunk_id == "support"
    assert results[0].score > results[-1].score
    assert all(0 < r.score < 1 for r in results)
