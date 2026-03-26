"""Stage 1 searchers: Dense, Sparse, and Exact.

Each searcher takes an asyncpg connection and returns a list of raw result dicts
that the fusion stage consumes.  They are stateless — no pooling or caching here;
the engine is responsible for acquiring connections.

DenseSearcher   — pgvector cosine similarity (HNSW index)
SparseSearcher  — PostgreSQL tsvector full-text search (GIN index)
ExactSearcher   — citation / case name exact / trigram match
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ── SQL Templates ──────────────────────────────────────────────────────────────

_DENSE_SQL = """
SELECT
    s.id::text      AS segment_id,
    s.case_id::text AS case_id,
    s.segment_type::text AS segment_type,
    s.content,
    s.retrieval_weight,
    s.opinion_type::text AS opinion_type,
    c.court::text   AS court,
    c.year,
    (s.embedding <-> $1::vector) AS distance
FROM case_segments s
JOIN cases c ON c.id = s.case_id
WHERE s.embedding IS NOT NULL
  AND c.status != 'overruled'
  {court_filter}
  {year_filter}
ORDER BY distance
LIMIT $2
"""

_SPARSE_SQL = """
SELECT
    s.id::text      AS segment_id,
    s.case_id::text AS case_id,
    s.segment_type::text AS segment_type,
    s.content,
    s.retrieval_weight,
    s.opinion_type::text AS opinion_type,
    c.court::text   AS court,
    c.year,
    ts_rank_cd(s.content_tsv, websearch_to_tsquery('english', $1)) AS rank
FROM case_segments s
JOIN cases c ON c.id = s.case_id
WHERE s.content_tsv @@ websearch_to_tsquery('english', $1)
  AND c.status != 'overruled'
  {court_filter}
  {year_filter}
ORDER BY rank DESC
LIMIT $2
"""

_EXACT_SQL = """
SELECT
    c.id::text   AS case_id,
    c.case_name,
    c.citation,
    c.court::text AS court,
    c.year
FROM cases c
WHERE c.status != 'overruled'
  AND (
      c.citation = $1
      OR c.case_name ILIKE $2
  )
ORDER BY CASE WHEN c.citation = $1 THEN 0 ELSE 1 END
LIMIT 5
"""

# ── DenseSearcher ─────────────────────────────────────────────────────────────


class DenseSearcher:
    """Search case segments using pgvector cosine similarity."""

    async def search(
        self,
        conn,
        embedding: list[float],
        limit: int = 60,
        court_codes: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> list[dict]:
        """Return up to `limit` segments ranked by cosine similarity.

        Args:
            conn: asyncpg connection
            embedding: query embedding vector (list of floats)
            limit: max rows
            court_codes: optional list of court codes to filter (e.g. ["NGSC", "NGCA"])
            year_min / year_max: optional year range filter
        """
        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"

        court_filter, year_filter, extra_params = _build_filters(
            court_codes, year_min, year_max, start_param=3
        )
        sql = _DENSE_SQL.format(court_filter=court_filter, year_filter=year_filter)

        try:
            rows = await conn.fetch(sql, emb_str, limit, *extra_params)
        except Exception as exc:
            logger.error("dense_search.failed exc=%s", exc)
            return []

        return [dict(r) for r in rows]


# ── SparseSearcher ────────────────────────────────────────────────────────────


class SparseSearcher:
    """Search case segments using PostgreSQL full-text search."""

    async def search(
        self,
        conn,
        query_text: str,
        limit: int = 60,
        court_codes: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> list[dict]:
        """Return up to `limit` segments ranked by ts_rank_cd.

        Args:
            conn: asyncpg connection
            query_text: raw query string (passed to websearch_to_tsquery)
            limit: max rows
        """
        if not query_text or not query_text.strip():
            return []

        court_filter, year_filter, extra_params = _build_filters(
            court_codes, year_min, year_max, start_param=3
        )
        sql = _SPARSE_SQL.format(court_filter=court_filter, year_filter=year_filter)

        try:
            rows = await conn.fetch(sql, query_text, limit, *extra_params)
        except Exception as exc:
            logger.error("sparse_search.failed query=%.40s exc=%s", query_text, exc)
            return []

        return [dict(r) for r in rows]


# ── ExactSearcher ─────────────────────────────────────────────────────────────


class ExactSearcher:
    """Exact citation / case name match for explicit references in a query."""

    async def search(self, conn, reference: str) -> list[dict]:
        """Find cases matching an exact citation or approximate case name.

        Args:
            conn: asyncpg connection
            reference: citation string or case name fragment
        """
        if not reference or not reference.strip():
            return []

        try:
            rows = await conn.fetch(_EXACT_SQL, reference, f"%{reference}%")
        except Exception as exc:
            logger.error("exact_search.failed reference=%.40s exc=%s", reference, exc)
            return []

        return [dict(r) for r in rows]


# ── Filter builder ────────────────────────────────────────────────────────────


def _build_filters(
    court_codes: list[str] | None,
    year_min: int | None,
    year_max: int | None,
    start_param: int,
) -> tuple[str, str, list]:
    """Return (court_filter_sql, year_filter_sql, extra_params)."""
    extra: list = []
    court_sql = ""
    year_sql = ""
    p = start_param

    if court_codes:
        placeholders = ", ".join(f"${p + i}" for i in range(len(court_codes)))
        court_sql = f"AND c.court::text = ANY(ARRAY[{placeholders}])"
        extra.extend(court_codes)
        p += len(court_codes)

    if year_min is not None:
        year_sql += f" AND c.year >= ${p}"
        extra.append(year_min)
        p += 1

    if year_max is not None:
        year_sql += f" AND c.year <= ${p}"
        extra.append(year_max)
        p += 1

    return court_sql, year_sql, extra
