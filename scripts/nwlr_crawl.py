#!/usr/bin/env python3
"""NWLR Online crawler CLI.

Commands::

    # Discover case IDs for a range of parts (saves to data/nwlr_manifest.json)
    python scripts/nwlr_crawl.py discover --parts-from 2034 --parts-to 2034

    # Fetch metadata + HTML for a specific part (reads manifest, hits API)
    python scripts/nwlr_crawl.py fetch --part 2034

    # Parse fetched HTML into NWLR.jsonl
    python scripts/nwlr_crawl.py parse --limit 100

    # Load parsed JSONL into PostgreSQL
    python scripts/nwlr_crawl.py load --limit 100

    # Show counts at each stage
    python scripts/nwlr_crawl.py status
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import settings  # noqa: E402
from ingestion.parsing.nwlr_parser import NWLRParser  # noqa: E402
from ingestion.sources.nwlronline import NWLRCaseId, NWLRCrawler  # noqa: E402

logger = structlog.get_logger(__name__)

_DATA_DIR = Path("data")
_MANIFEST = _DATA_DIR / "nwlr_manifest.json"
_RAW_DIR = _DATA_DIR / "raw" / "nwlr"
_OUT_JSONL = _DATA_DIR / "raw" / "judgments" / "NWLR.jsonl"


# ── CLI group ─────────────────────────────────────────────────────────────────


@click.group()
def cli() -> None:
    """NWLR Online ingestion pipeline."""


# ── discover ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--parts-from", "parts_from", default=1, show_default=True, type=int)
@click.option("--parts-to", "parts_to", default=2034, show_default=True, type=int)
@click.option(
    "--no-skip",
    "no_skip",
    is_flag=True,
    default=False,
    help="Re-discover parts already in the manifest.",
)
@click.option(
    "--max-scan-page",
    "max_scan_page",
    default=700,
    show_default=True,
    type=int,
    help="Scan pages 1–N per part to find the first case entry point.",
)
@click.option(
    "--rate-limit",
    "rate_limit",
    default=3.0,
    show_default=True,
    type=float,
    help="Seconds between API requests. Cloudflare blocks at <2s sustained. Default 3s is safe.",
)
def discover(
    parts_from: int,
    parts_to: int,
    no_skip: bool,
    max_scan_page: int,
    rate_limit: float,
) -> None:
    """Enumerate case IDs for NWLR parts and save to nwlr_manifest.json.

    Uses a linear scan (page 1 → --max-scan-page) per part, caching null
    probes to disk so interrupted runs resume without re-probing.

    Estimated time at --rate-limit 0.5 for a single part:
      worst case = 700 probes × 0.5s = 350s (~6 min)
      typical    = first case found at ~page N, chain-follow adds ~10 probes
    """
    email = settings.nwlr_email
    password = settings.nwlr_password
    if not email or not password:
        raise click.ClickException("NWLR_EMAIL and NWLR_PASSWORD must be set in .env")

    parts = list(range(parts_from, parts_to + 1))
    click.echo(
        f"Discovering {len(parts)} parts ({parts_from}–{parts_to}), "
        f"max_scan_page={max_scan_page}, rate_limit={rate_limit}s …"
    )

    async def _run() -> None:
        async with NWLRCrawler(
            email=email,
            password=password,
            raw_cache_dir=_RAW_DIR,
            rate_limit_seconds=rate_limit,
        ) as crawler:
            result = await crawler.discover_all_parts(
                parts=parts,
                skip_existing=not no_skip,
                manifest_path=_MANIFEST,
                max_scan_page=max_scan_page,
            )
        total_cases = sum(len(v) for v in result.values())
        click.echo(f"Done. {total_cases} case IDs across {len(result)} parts.")
        click.echo(f"Manifest saved to {_MANIFEST}")

    asyncio.run(_run())


# ── fetch ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--part", "part", default=None, type=int, help="Fetch only this part.")
@click.option("--limit", default=0, show_default=True, type=int, help="Max cases to fetch (0=all).")
def fetch(part: int | None, limit: int) -> None:
    """Fetch metadata JSON + HTML for cases listed in the manifest."""
    if not _MANIFEST.exists():
        raise click.ClickException("Manifest not found. Run `discover` first.")

    email = settings.nwlr_email
    password = settings.nwlr_password
    if not email or not password:
        raise click.ClickException("NWLR_EMAIL and NWLR_PASSWORD must be set in .env")

    manifest: dict[str, list[str]] = json.loads(_MANIFEST.read_text())

    # Build work list
    case_ids: list[NWLRCaseId] = []
    for part_key, ids in sorted(manifest.items(), key=lambda x: int(x[0])):
        if part is not None and int(part_key) != part:
            continue
        case_ids.extend(NWLRCaseId.from_str(s) for s in ids)

    if limit:
        case_ids = case_ids[:limit]

    click.echo(f"Fetching {len(case_ids)} cases …")

    async def _run() -> None:
        fetched = skipped = errors = 0
        async with NWLRCrawler(
            email=email,
            password=password,
            raw_cache_dir=_RAW_DIR,
        ) as crawler:
            for cid in case_ids:
                meta_path = _RAW_DIR / "meta" / f"{cid.as_str()}.json"
                html_path = _RAW_DIR / "html" / f"{cid.as_str()}.html"
                if meta_path.exists() and html_path.exists():
                    skipped += 1
                    continue
                try:
                    meta = await crawler.fetch_case_metadata(cid)
                    if meta is None:
                        logger.warning("nwlr_fetch.no_meta", case_id=cid.as_str())
                        errors += 1
                        continue
                    html = await crawler.fetch_case_html(cid)
                    if html is None:
                        logger.warning("nwlr_fetch.no_html", case_id=cid.as_str())
                        errors += 1
                        continue
                    fetched += 1
                except Exception as exc:
                    logger.error("nwlr_fetch.error", case_id=cid.as_str(), error=str(exc))
                    errors += 1

        click.echo(f"Fetched: {fetched}  Skipped (cached): {skipped}  Errors: {errors}")

    asyncio.run(_run())


# ── parse ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--limit", default=0, type=int, show_default=True, help="Max records to parse (0=all)."
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing NWLR.jsonl entries.",
)
def parse(limit: int, overwrite: bool) -> None:
    """Parse cached HTML + metadata into data/raw/judgments/NWLR.jsonl."""
    meta_dir = _RAW_DIR / "meta"
    html_dir = _RAW_DIR / "html"

    if not meta_dir.exists():
        raise click.ClickException("No metadata cache found. Run `fetch` first.")

    meta_files = sorted(meta_dir.glob("*.json"))
    if limit:
        meta_files = meta_files[:limit]

    # Load existing case IDs to allow incremental runs
    existing_ids: set[str] = set()
    if _OUT_JSONL.exists() and not overwrite:
        for line in _OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if "source_url" in rec:
                    existing_ids.add(rec["source_url"])

    _OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    parser = NWLRParser()
    parsed = skipped = errors = 0

    mode = "w" if overwrite else "a"
    with _OUT_JSONL.open(mode, encoding="utf-8") as fh:
        for meta_path in meta_files:
            case_id_str = meta_path.stem
            html_path = html_dir / f"{case_id_str}.html"

            if not html_path.exists():
                logger.debug("nwlr_parse.no_html", case_id=case_id_str)
                continue

            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                html = html_path.read_text(encoding="utf-8")
                raw = parser.parse(html, metadata)

                if raw.source_url in existing_ids:
                    skipped += 1
                    continue

                record = {
                    "case_name": raw.case_name,
                    "citation": raw.media_neutral_citation,
                    "court": raw.court.value,
                    "year": raw.judgment_date.year if raw.judgment_date else 0,
                    "judges": raw.judges,
                    "lead_judge": raw.judges[0] if raw.judges else None,
                    "area_of_law": [],  # will be inferred by MetadataExtractor downstream
                    "full_text": raw.full_text,
                    "full_html": raw.full_html,
                    "source_url": raw.source_url,
                    "source": "nwlronline",
                    "jurisdiction": "NG",
                    "labels": raw.labels,
                }
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                parsed += 1

            except Exception as exc:
                logger.error("nwlr_parse.error", case_id=case_id_str, error=str(exc))
                errors += 1

    click.echo(f"Parsed: {parsed}  Skipped (existing): {skipped}  Errors: {errors}")
    click.echo(f"Output: {_OUT_JSONL}")


# ── embed ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--limit", default=0, type=int, show_default=True, help="Max cases to embed (0=all).")
@click.option("--batch-size", default=128, type=int, show_default=True)
@click.option(
    "--sleep",
    "sleep_between_batches",
    default=0.0,
    type=float,
    show_default=True,
    help="Seconds between Voyage AI batches (use ~21 for free-tier limits).",
)
def embed(limit: int, batch_size: int, sleep_between_batches: float) -> None:
    """Chunk, embed, and upsert NWLR cases into case_segments.

    Reads NWLR.jsonl, looks up DB case IDs by citation, chunks full_text,
    embeds via Voyage AI, and upserts embedded chunks into case_segments.
    Already-embedded cases (segments exist in DB) are skipped.
    """
    if not _OUT_JSONL.exists():
        raise click.ClickException("NWLR.jsonl not found. Run `parse` first.")

    lines = [ln for ln in _OUT_JSONL.read_text(encoding="utf-8").splitlines() if ln.strip()]
    records = [json.loads(ln) for ln in lines]
    if limit:
        records = records[:limit]

    click.echo(f"Embedding {len(records)} NWLR cases …")

    async def _run() -> None:
        from db import close_pool, get_pool  # noqa: PLC0415
        from ingestion.embedding.chunker import LegalTextChunker  # noqa: PLC0415
        from ingestion.embedding.embedder import CorpusEmbedder  # noqa: PLC0415
        from ingestion.loaders.db_loader import BulkCaseLoader  # noqa: PLC0415

        pool = await get_pool()
        chunker = LegalTextChunker()
        embedder = CorpusEmbedder(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            batch_size=batch_size,
            sleep_between_batches=sleep_between_batches,
        )
        loader = BulkCaseLoader()

        embedded = skipped = errors = 0

        try:
            for rec in records:
                citation = rec.get("citation") or ""
                async with pool.acquire() as conn:
                    # Look up DB case ID by citation
                    row = await conn.fetchrow("SELECT id FROM cases WHERE citation = $1", citation)
                    if row is None:
                        logger.warning("nwlr_embed.case_not_found", citation=citation)
                        errors += 1
                        continue

                    case_uuid = str(row["id"])

                    # Skip if already embedded (segments with embedding exist)
                    seg_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM case_segments WHERE case_id = $1::uuid AND embedding IS NOT NULL",
                        case_uuid,
                    )
                    if seg_count and seg_count > 0:
                        skipped += 1
                        continue

                    # Build minimal segmented_judgment dict for the chunker
                    judgment = {
                        "case_id": case_uuid,
                        "court": rec.get("court", ""),
                        "year": rec.get("year", 0),
                        "area_of_law": rec.get("area_of_law", []),
                        "case_name": rec.get("case_name", ""),
                        "citation": citation,
                        "ratio_decidendi": "",
                        "holdings": [],
                        "segments": [
                            {"segment_type": "analysis", "content": rec.get("full_text", "")}
                        ],
                    }

                    chunks = chunker.chunk(judgment)
                    if not chunks:
                        errors += 1
                        continue

                    # Embed the chunks
                    embedded_chunks = await embedder.embed_chunks(chunks)

                    # Upsert into case_segments
                    chunk_dicts = [
                        {
                            "segment_type": c.segment_type,
                            "content": c.content,
                            "embedding": c.embedding,
                            "chunk_id": c.chunk_id,
                        }
                        for c in embedded_chunks
                        if c.embedding is not None
                    ]
                    await loader.upsert_chunks(conn, case_uuid, chunk_dicts)
                    embedded += 1

        finally:
            await close_pool()

        click.echo(f"Embedded: {embedded}  Skipped (already done): {skipped}  Errors: {errors}")

    asyncio.run(_run())


# ── load ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--limit", default=0, type=int, show_default=True, help="Max records to load (0=all)."
)
def load(limit: int) -> None:
    """Load parsed NWLR.jsonl records into PostgreSQL."""
    if not _OUT_JSONL.exists():
        raise click.ClickException("NWLR.jsonl not found. Run `parse` first.")

    lines = _OUT_JSONL.read_text(encoding="utf-8").splitlines()
    records = [json.loads(ln) for ln in lines if ln.strip()]
    if limit:
        records = records[:limit]

    click.echo(f"Loading {len(records)} records into PostgreSQL …")

    async def _run() -> None:
        from db import close_pool, get_pool  # noqa: PLC0415
        from ingestion.loaders.db_loader import BulkCaseLoader  # noqa: PLC0415

        pool = await get_pool()
        loader = BulkCaseLoader()
        try:
            async with pool.acquire() as conn:
                stats = await loader.load_batch(conn, records)
        finally:
            await close_pool()

        click.echo(
            f"Loaded: {stats['loaded']}  Errors: {stats['errors']}  " f"Total: {stats['total']}"
        )

    asyncio.run(_run())


# ── status ────────────────────────────────────────────────────────────────────


@cli.command()
def status() -> None:
    """Show counts at each pipeline stage."""
    # Manifest
    manifest_parts = 0
    manifest_cases = 0
    if _MANIFEST.exists():
        m = json.loads(_MANIFEST.read_text())
        manifest_parts = len(m)
        manifest_cases = sum(len(v) for v in m.values())

    # Cached files
    meta_count = (
        len(list((_RAW_DIR / "meta").glob("*.json"))) if (_RAW_DIR / "meta").exists() else 0
    )
    html_count = (
        len(list((_RAW_DIR / "html").glob("*.html"))) if (_RAW_DIR / "html").exists() else 0
    )

    # Parsed JSONL
    jsonl_count = 0
    if _OUT_JSONL.exists():
        jsonl_count = sum(1 for ln in _OUT_JSONL.read_text().splitlines() if ln.strip())

    click.echo(f"Manifest    : {manifest_cases:>7} cases across {manifest_parts} parts")
    click.echo(f"Meta cache  : {meta_count:>7} files")
    click.echo(f"HTML cache  : {html_count:>7} files")
    click.echo(f"Parsed JSONL: {jsonl_count:>7} records  ({_OUT_JSONL})")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
