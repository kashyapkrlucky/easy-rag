"""PostgreSQL full-text implementation of ``KeywordSearch`` (ARCHITECTURE D15).

PostgreSQL's ``ts_rank`` has no notion of term rarity, so a chunk repeating a common word
would outrank one containing a rare identifier. Instead each query term gets a BM25-style
IDF weight computed from document frequencies in this corpus, and a chunk's score is the
share of the query's total weight it matches:

    weight(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(c)  = Σ weight(t) · [c matches t]  /  Σ weight(t)        ∈ [0, 1]

Each term is parsed by PostgreSQL itself, bound as a parameter: identifiers with
``phraseto_tsquery('simple', …)`` (exact, adjacent words), ordinary words with
``plainto_tsquery('english', …)`` (stemmed: "deploy" matches "deployment"). Documents
carry both forms (migration 0004).
"""

import math
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from easyrag.domain.chunks import RetrievedChunk
from easyrag.retrieval.keyword import is_identifier


def _queries(terms: Sequence[str]) -> list[str]:
    return [
        f"phraseto_tsquery('simple', :t{i})"
        if is_identifier(term)
        else f"plainto_tsquery('english', :t{i})"
        for i, term in enumerate(terms)
    ]


class PostgresKeywordStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def search(self, terms: Sequence[str], *, top_k: int) -> list[RetrievedChunk]:
        if not terms:
            return []
        parameters: dict[str, object] = {f"t{i}": term for i, term in enumerate(terms)}
        queries = _queries(terms)

        frequencies = text(
            "SELECT count(*) AS n, "
            + ", ".join(
                f"count(*) FILTER (WHERE search_vector @@ {queries[i]}) AS df{i}"
                for i in range(len(terms))
            )
            + " FROM chunks"
        )
        async with self._engine.connect() as connection:
            row = (await connection.execute(frequencies, parameters)).one()
            total = int(row.n)
            document_frequencies = [int(row[i + 1]) for i in range(len(terms))]
            weights = [_idf(total, df) for df in document_frequencies]
            matchable = [i for i, df in enumerate(document_frequencies) if df > 0]
            if not matchable:
                return []

            parameters |= {f"w{i}": weights[i] for i in matchable}
            parameters |= {"total_weight": sum(weights), "top_k": top_k}
            matched = " + ".join(
                f"CASE WHEN c.search_vector @@ {queries[i]} THEN :w{i} ELSE 0 END"
                for i in matchable
            )
            any_term = " || ".join(queries[i] for i in matchable)
            query = text(
                f"""
                SELECT c.id AS chunk_id, d.source_path, d.title AS document_title,
                       c.heading_path, c.section_anchor, c.content, c.start_line, c.end_line,
                       ({matched}) / :total_weight AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.search_vector @@ ({any_term})
                ORDER BY score DESC, ts_rank_cd(c.search_vector, {any_term}) DESC, c.id
                LIMIT :top_k
                """
            )
            rows = await connection.execute(query, parameters)
            return [
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    source_path=row.source_path,
                    document_title=row.document_title,
                    heading_path=tuple(row.heading_path),
                    section_anchor=row.section_anchor,
                    content=row.content,
                    start_line=row.start_line,
                    end_line=row.end_line,
                    score=float(row.score),
                    keyword_score=float(row.score),
                )
                for row in rows
            ]


def _idf(total: int, document_frequency: int) -> float:
    """BM25 inverse document frequency (always positive)."""
    return math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
