"""Keyword (lexical) retrieval: exact terms that dense retrieval tends to miss.

Identifiers such as ``EASYRAG_EMBEDDING_MODEL``, ``nomic-embed-text:v1.5`` or error
codes are kept whole and searched as phrases; other words are kept unless they are
stopwords. Scoring (IDF-weighted term coverage) happens in the store.
"""

import re
from collections.abc import Sequence
from typing import Protocol

from easyrag.domain.chunks import RetrievedChunk
from easyrag.retrieval.vector import MAX_TOP_K
from easyrag.stopwords import STOPWORDS

MAX_TERMS = 16

# Word characters joined by _ . - : / — e.g. EASYRAG_DOCS_DIR, v1.5, nomic-embed-text:v1.5
_IDENTIFIER = re.compile(r"[^\W_]+(?:[_.:/-][^\W_]+)+")
_WORD = re.compile(r"[^\W_]+")


def is_identifier(term: str) -> bool:
    """True for terms searched as exact phrases (EASYRAG_DOCS_DIR, v1.5, a-b:c)."""
    return _IDENTIFIER.fullmatch(term) is not None


def keyword_terms(query: str) -> tuple[str, ...]:
    """Search terms for ``query``: identifiers first (as phrases), then content words."""
    terms: list[str] = [match.lower() for match in _IDENTIFIER.findall(query)]
    for word in _WORD.findall(_IDENTIFIER.sub(" ", query)):
        word = word.lower()
        if word in STOPWORDS or (len(word) < 2 and not word.isdigit()):
            continue
        terms.append(word)
    return tuple(dict.fromkeys(terms))[:MAX_TERMS]


class KeywordSearch(Protocol):
    async def search(self, terms: Sequence[str], *, top_k: int) -> list[RetrievedChunk]:
        """Chunks matching any term, best first; ``score`` is the matched IDF share (0..1)."""
        ...


class KeywordRetriever:
    def __init__(self, *, search: KeywordSearch, default_top_k: int = 5) -> None:
        if not 1 <= default_top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        self._search = search
        self._default_top_k = default_top_k

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        """``min_score`` is a cosine-similarity threshold and does not apply here."""
        top_k = self._default_top_k if top_k is None else top_k
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        if not query.strip():
            raise ValueError("query must not be blank")
        terms = keyword_terms(query)
        return await self._search.search(terms, top_k=top_k) if terms else []
