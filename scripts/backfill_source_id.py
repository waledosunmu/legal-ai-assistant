#!/usr/bin/env python3
"""Backfill cases.source_id from the embedded JSONL segment files.

Builds a citation→slug map from data/processed/segments/*_embedded.jsonl,
then updates every row in `cases` whose citation matches. Idempotent — rows
with source_id already set are skipped.

Usage:
    python scripts/backfill_source_id.py [--dry-run]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import asyncpg
import click
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import settings

structlog.configure(
    processors=[structlog.stdlib.add_log_level, structlog.dev.ConsoleRenderer()]
)
logger = structlog.get_logger(__name__)

_SEGMENTS_DIR = Path("data/processed/segments")


def _slug_from_url(source_url: str) -> str | None:
    """Derive AKN slug from a NigeriaLII source URL.

    e.g. "https://nigerialii.org/akn/ng/judgment/ngca/1970/6/eng@1970-12-21"
         → "akn_ng_judgment_ngca_1970_6"
    """
    try:
        from urllib.parse import urlparse
        path = urlparse(source_url).path  # /akn/ng/judgment/ngca/1970/6/eng@1970-12-21
        parts = [p for p in path.strip("/").split("/") if not p.startswith("eng@")]
        return "_".join(parts)
    except Exception:
        return None


def _build_citation_to_slug() -> dict[str, str]:
    """Scan embedded JSONL files and return {citation: slug} mapping."""
    mapping: dict[str, str] = {}
    for path in sorted(_SEGMENTS_DIR.glob("*_embedded.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                cit = chunk.get("citation", "").strip()
                slug = chunk.get("case_id", "").strip()
                if cit and slug and cit not in mapping:
                    mapping[cit] = slug
    return mapping


async def _run(dry_run: bool) -> None:
    citation_to_slug = _build_citation_to_slug()
    logger.info("backfill.slug_map_built", entries=len(citation_to_slug))

    conn = await asyncpg.connect(settings.asyncpg_url)
    try:
        rows = await conn.fetch(
            "SELECT id::text, citation, source_url FROM cases WHERE source_id IS NULL"
        )
        logger.info("backfill.cases_without_source_id", count=len(rows))

        updated = 0
        skipped = 0
        for row in rows:
            # Primary: citation cross-reference from embedded files
            slug = citation_to_slug.get(row["citation"])
            # Fallback: derive slug directly from source_url AKN path
            if slug is None and row["source_url"]:
                slug = _slug_from_url(row["source_url"])
            if slug is None:
                skipped += 1
                logger.debug(
                    "backfill.no_slug_found",
                    case_id=row["id"],
                    citation=row["citation"],
                )
                continue

            if not dry_run:
                await conn.execute(
                    "UPDATE cases SET source_id = $1 WHERE id = $2::uuid",
                    slug,
                    row["id"],
                )
            updated += 1

        logger.info(
            "backfill.complete",
            updated=updated,
            skipped=skipped,
            dry_run=dry_run,
        )
    finally:
        await conn.close()


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Report without writing.")
def cli(dry_run: bool) -> None:
    """Backfill cases.source_id from embedded segment files."""
    asyncio.run(_run(dry_run))


if __name__ == "__main__":
    cli()
