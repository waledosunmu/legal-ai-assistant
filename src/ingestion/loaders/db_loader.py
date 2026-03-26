"""Load processed judgments and segments into PostgreSQL via asyncpg."""

from __future__ import annotations

import asyncpg
import structlog

from ingestion.segmentation.models import SegmentType

logger = structlog.get_logger(__name__)


# ── Enum mappings ──────────────────────────────────────────────────────────────

# NigeriaLII crawler Court code → PostgreSQL court_enum value
_COURT_MAP: dict[str, str] = {
    "NGSC": "SUPREME_COURT",
    "NGCA": "COURT_OF_APPEAL",
    "NGFCHC": "FEDERAL_HIGH_COURT",
    "NGLAHC": "LAGOS_HIGH_COURT",
    "NGKNHC": "KANO_HIGH_COURT",
    "NGBAHC": "BAUCHI_HIGH_COURT",
    "NGBEHC": "BENUE_HIGH_COURT",
    "NGEBHC": "EBONYI_HIGH_COURT",
}

# Our SegmentType values → PostgreSQL segment_type enum
# DB enum: FACTS, ISSUE, HOLDING, RATIO, OBITER, ORDER, ANALYSIS, CAPTION, INTRODUCTION
_SEGMENT_TYPE_MAP: dict[str, str] = {
    SegmentType.CAPTION.value: "CAPTION",
    SegmentType.BACKGROUND.value: "ANALYSIS",  # no BACKGROUND in DB enum
    SegmentType.FACTS.value: "FACTS",
    SegmentType.ISSUES.value: "ISSUE",
    SegmentType.SUBMISSION.value: "ANALYSIS",
    SegmentType.ANALYSIS.value: "ANALYSIS",
    SegmentType.HOLDING.value: "HOLDING",
    SegmentType.RATIO.value: "RATIO",
    SegmentType.OBITER.value: "OBITER",
    SegmentType.ORDERS.value: "ORDER",
    SegmentType.DISSENT.value: "ANALYSIS",
    SegmentType.CONCURRENCE.value: "ANALYSIS",
    SegmentType.CITED_CASE.value: "ANALYSIS",
    # Chunk-level segment types from LegalTextChunker
    "ratio": "RATIO",
    "holding": "HOLDING",
    "facts": "FACTS",
    "analysis": "ANALYSIS",
    "orders": "ORDER",
    "obiter": "OBITER",
    "caption": "CAPTION",
    "background": "ANALYSIS",
    "issues": "ISSUE",
}

# Retrieval weight per DB segment type
_RETRIEVAL_WEIGHT: dict[str, float] = {
    "RATIO": 2.0,
    "HOLDING": 1.8,
    "OBITER": 1.3,
    "ORDER": 1.3,
    "ISSUE": 1.2,
    "FACTS": 1.0,
    "ANALYSIS": 1.0,
    "CAPTION": 0.5,
    "INTRODUCTION": 0.7,
}

_UPSERT_CASE_SQL = """
    INSERT INTO cases (
        case_name, citation, lpelr_citation, court,
        year, judges, lead_judge, area_of_law,
        full_text, source_url, source, jurisdiction
    ) VALUES (
        $1, $2, $3, $4::court_enum,
        $5, $6, $7, $8,
        $9, $10, $11, $12
    )
    ON CONFLICT (citation) DO UPDATE SET
        case_name        = EXCLUDED.case_name,
        judges           = EXCLUDED.judges,
        lead_judge       = EXCLUDED.lead_judge,
        area_of_law      = EXCLUDED.area_of_law,
        full_text        = EXCLUDED.full_text,
        source_url       = EXCLUDED.source_url,
        last_verified_at = NOW()
    RETURNING id
"""

_INSERT_CASE_NO_CITATION_SQL = """
    INSERT INTO cases (
        case_name, court, year, judges, lead_judge,
        area_of_law, full_text, source_url, source, jurisdiction
    ) VALUES ($1, $2::court_enum, $3, $4, $5, $6, $7, $8, $9, $10)
    RETURNING id
"""

_UPSERT_SEGMENT_SQL = """
    INSERT INTO case_segments (
        case_id, segment_type, content,
        issue_number, retrieval_weight, metadata
    ) VALUES (
        $1, $2::segment_type, $3,
        $4, $5, $6::jsonb
    )
"""

_INSERT_CHUNK_SQL = """
    INSERT INTO case_segments (
        case_id, segment_type, content, embedding,
        retrieval_weight, opinion_type, metadata
    ) VALUES (
        $1::uuid, $2::segment_type, $3,
        $4::vector,
        $5, 'LEAD'::opinion_type, $6::jsonb
    )
"""

_UPDATE_TSV_SQL = """
    UPDATE cases SET search_vector =
        setweight(to_tsvector('english', coalesce(citation, '')), 'A') ||
        setweight(to_tsvector('english', case_name), 'B') ||
        setweight(to_tsvector('english', substring(full_text, 1, 100000)), 'C')
    WHERE id = $1
"""


class BulkCaseLoader:
    """
    Load processed judgment records into the PostgreSQL ``cases`` and
    ``case_segments`` tables via asyncpg.

    Usage::

        loader = BulkCaseLoader()
        async with get_connection() as conn:
            case_id = await loader.upsert_case(conn, record)
            await loader.upsert_segments(conn, case_id, record["segments"])
            await loader.update_search_vector(conn, case_id)
    """

    @staticmethod
    def map_court(court_code: str) -> str:
        """Map NigeriaLII court code to PostgreSQL court_enum value."""
        mapped = _COURT_MAP.get(court_code.upper())
        if mapped is None:
            raise ValueError(
                f"Unknown court code '{court_code}'. Valid codes: {', '.join(_COURT_MAP)}"
            )
        return mapped

    @staticmethod
    def map_segment_type(seg_type: str) -> str:
        """Map our SegmentType string to PostgreSQL segment_type enum."""
        return _SEGMENT_TYPE_MAP.get(seg_type.lower(), "ANALYSIS")

    async def upsert_case(
        self,
        conn: asyncpg.Connection,
        record: dict,
    ) -> str:
        """
        Upsert a single case record.  Returns the ``cases.id`` UUID string.

        ``record`` dict keys: ``case_name``, ``court``, ``year``,
        ``citation``, ``lpelr_citation``, ``judges``, ``lead_judge``,
        ``area_of_law``, ``full_text``, ``source_url``, ``source``.
        """
        court_enum = self.map_court(record["court"])
        citation = record.get("citation") or None  # coerce empty string to None

        if citation:
            row = await conn.fetchrow(
                _UPSERT_CASE_SQL,
                record["case_name"],
                citation,
                record.get("lpelr_citation"),
                court_enum,
                record.get("year") or 0,
                record.get("judges") or [],
                record.get("lead_judge"),
                record.get("area_of_law") or [],
                record.get("full_text", ""),
                record.get("source_url"),
                record.get("source", "nigerialii"),
                record.get("jurisdiction", "NG"),
            )
        else:
            row = await conn.fetchrow(
                _INSERT_CASE_NO_CITATION_SQL,
                record["case_name"],
                court_enum,
                record.get("year") or 0,
                record.get("judges") or [],
                record.get("lead_judge"),
                record.get("area_of_law") or [],
                record.get("full_text", ""),
                record.get("source_url"),
                record.get("source", "nigerialii"),
                record.get("jurisdiction", "NG"),
            )

        assert row is not None, "fetchrow returned None after upsert"
        case_id = str(row["id"])
        logger.debug("db_loader.upserted_case", case_id=case_id, case_name=record["case_name"])
        return case_id

    async def upsert_segments(
        self,
        conn: asyncpg.Connection,
        case_id: str,
        segments: list[dict],
    ) -> int:
        """
        Insert segments for a case.  Returns the count inserted.

        Deletes existing segments for the case first so re-runs are
        idempotent (cheaper than per-segment upsert on this table).

        ``segments`` list of dicts with keys: ``segment_type``, ``content``,
        ``issue_number`` (optional), ``metadata`` (optional).
        """
        await conn.execute("DELETE FROM case_segments WHERE case_id = $1", case_id)

        count = 0
        import json as _json

        for seg in segments:
            seg_type = self.map_segment_type(seg.get("segment_type", "analysis"))
            weight = _RETRIEVAL_WEIGHT.get(seg_type, 1.0)
            metadata = _json.dumps(seg.get("metadata") or {})
            await conn.execute(
                _UPSERT_SEGMENT_SQL,
                case_id,
                seg_type,
                seg.get("content", ""),
                seg.get("issue_number"),
                weight,
                metadata,
            )
            count += 1

        logger.debug("db_loader.segments_inserted", case_id=case_id, count=count)
        return count

    async def upsert_chunks(
        self,
        conn: asyncpg.Connection,
        case_id: str,
        chunks: list[dict],
    ) -> int:
        """
        Delete existing segments for ``case_id``, then INSERT embedded chunks.

        Each chunk dict must have: ``segment_type``, ``content``, ``embedding``
        (list of floats).  Returns the count inserted.

        asyncpg has no native pgvector codec, so embeddings are serialized as
        ``"[f1,f2,...,f1024]"`` (no spaces) and cast with ``::vector`` in SQL.
        """
        import json as _json

        await conn.execute("DELETE FROM case_segments WHERE case_id = $1::uuid", case_id)

        count = 0
        for chunk in chunks:
            seg_type = self.map_segment_type(chunk.get("segment_type", "analysis"))
            weight = _RETRIEVAL_WEIGHT.get(seg_type, 1.0)
            emb = chunk.get("embedding")
            emb_str = "[" + ",".join(str(v) for v in emb) + "]" if emb else None
            await conn.execute(
                _INSERT_CHUNK_SQL,
                case_id,
                seg_type,
                chunk.get("content", ""),
                emb_str,
                weight,
                _json.dumps({"chunk_id": chunk.get("chunk_id")}),
            )
            count += 1

        logger.debug("db_loader.chunks_inserted", case_id=case_id, count=count)
        return count

    async def update_search_vector(
        self,
        conn: asyncpg.Connection,
        case_id: str,
    ) -> None:
        """Refresh the ``search_vector`` tsvector column for one case."""
        await conn.execute(_UPDATE_TSV_SQL, case_id)

    async def load_batch(
        self,
        conn: asyncpg.Connection,
        records: list[dict],
    ) -> dict:
        """
        Load a batch of processed judgment records.

        Each record must contain the fields expected by ``upsert_case``
        plus an optional ``segments`` list.

        Returns a stats dict: ``{loaded, skipped, errors}``.
        """
        loaded = 0
        errors = 0

        for record in records:
            try:
                async with conn.transaction():
                    case_id = await self.upsert_case(conn, record)
                    if record.get("segments"):
                        await self.upsert_segments(conn, case_id, record["segments"])
                    await self.update_search_vector(conn, case_id)
                loaded += 1
            except Exception as exc:
                logger.error(
                    "db_loader.record_error",
                    case_name=record.get("case_name"),
                    error=str(exc),
                )
                errors += 1

        stats = {"loaded": loaded, "errors": errors, "total": len(records)}
        logger.info("db_loader.batch_done", **stats)
        return stats
