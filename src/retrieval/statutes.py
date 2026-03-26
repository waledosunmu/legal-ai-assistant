"""Statute retrieval — runs in parallel with case retrieval.

Three sources in order:
  1. Anchor statutes per motion type (always included, looked up by statute title/section)
  2. Explicitly referenced statutes from parsed query (regex-extracted)
  3. Semantic search on statute_segments using query embedding

All results are returned as plain dicts for the API response.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ── Anchor statutes per motion type ───────────────────────────────────────────
# These are always included as foundation statutes for each motion type.

ANCHOR_STATUTES: dict[str, list[dict]] = {
    "motion_to_dismiss": [
        {
            "type": "anchor_statute",
            "title": "Constitution of the Federal Republic of Nigeria 1999",
            "section": "Section 6",
            "relevance": "Vests judicial powers; basis for court jurisdiction",
        },
        {
            "type": "anchor_statute",
            "title": "High Court Law (various States)",
            "section": "Order 22 / Order 23",
            "relevance": "Striking out and dismissal procedure",
        },
    ],
    "interlocutory_injunction": [
        {
            "type": "anchor_statute",
            "title": "High Court Law (various States)",
            "section": "Section 13 / Section 14",
            "relevance": "Jurisdiction to grant injunctions",
        },
        {
            "type": "anchor_statute",
            "title": "Federal High Court Act Cap F12 LFN 2004",
            "section": "Section 7",
            "relevance": "Federal High Court injunction jurisdiction",
        },
    ],
    "stay_of_proceedings": [
        {
            "type": "anchor_statute",
            "title": "Court of Appeal Act Cap C36 LFN 2004",
            "section": "Section 19",
            "relevance": "Stay of proceedings pending appeal",
        },
        {
            "type": "anchor_statute",
            "title": "Constitution of the Federal Republic of Nigeria 1999",
            "section": "Section 241",
            "relevance": "Right of appeal to Court of Appeal",
        },
    ],
    "summary_judgment": [
        {
            "type": "anchor_statute",
            "title": "High Court (Civil Procedure) Rules",
            "section": "Order 10 (Undefended List)",
            "relevance": "Undefended list / summary judgment procedure",
        },
    ],
    "extension_of_time": [
        {
            "type": "anchor_statute",
            "title": "Court of Appeal Rules 2021",
            "section": "Order 7 Rule 10",
            "relevance": "Extension of time to appeal",
        },
        {
            "type": "anchor_statute",
            "title": "Supreme Court Rules 1985",
            "section": "Order 2 Rule 31",
            "relevance": "Extension of time to appeal to Supreme Court",
        },
    ],
}

# ── SQL ───────────────────────────────────────────────────────────────────────

_SEMANTIC_STATUTE_SQL = """
SELECT
    id::text,
    title,
    short_title,
    section,
    content,
    (embedding <-> $1::vector) AS distance
FROM statute_segments
WHERE embedding IS NOT NULL
  AND status = 'in_force'
ORDER BY distance
LIMIT $2
"""


class StatuteRetriever:
    """Retrieve relevant statutes for a search query."""

    async def retrieve(
        self,
        conn,
        motion_type: str | None,
        query_embedding: list[float] | None,
        limit: int = 10,
    ) -> list[dict]:
        """Return list of statute dicts for the API response.

        Args:
            conn: asyncpg connection
            motion_type: detected motion type for anchor selection
            query_embedding: embedding vector for semantic statute search
            limit: max semantic results
        """
        results: list[dict] = []
        seen_ids: set[str] = set()

        # 1. Anchor statutes
        if motion_type and motion_type in ANCHOR_STATUTES:
            for anchor in ANCHOR_STATUTES[motion_type]:
                results.append(anchor)

        # 2. Semantic search (if embedding available + statute_segments populated)
        if query_embedding:
            semantic = await self._semantic_search(conn, query_embedding, limit)
            for row in semantic:
                row_id = row.get("id", "")
                if row_id not in seen_ids:
                    seen_ids.add(row_id)
                    results.append({
                        "type": "semantic_match",
                        "id": row_id,
                        "title": row.get("title", ""),
                        "short_title": row.get("short_title"),
                        "section": row.get("section"),
                        "content": (row.get("content") or "")[:400],
                        "relevance_score": 1.0 - float(row.get("distance") or 1.0),
                    })

        return results

    async def _semantic_search(
        self, conn, embedding: list[float], limit: int
    ) -> list[dict]:
        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
        try:
            rows = await conn.fetch(_SEMANTIC_STATUTE_SQL, emb_str, limit)
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("statute_retriever.semantic_failed exc=%s", exc)
            return []
