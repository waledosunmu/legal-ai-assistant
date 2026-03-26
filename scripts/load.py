#!/usr/bin/env python3
"""CLI entry point for loading processed judgments into PostgreSQL.

Usage examples::

    # Load all courts into the database
    python scripts/load.py run

    # Load only Supreme Court
    python scripts/load.py run --court NGSC

    # Build the citation graph after loading
    python scripts/load.py graph

    # Show DB counts vs file counts per court
    python scripts/load.py status
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import click
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db import close_pool, get_pool  # noqa: E402
from ingestion.citations.extractor import NigerianCitationExtractor  # noqa: E402
from ingestion.citations.graph_builder import CitationGraphBuilder  # noqa: E402
from ingestion.loaders.db_loader import BulkCaseLoader  # noqa: E402

logger = structlog.get_logger(__name__)

_INSERT_EDGE_SQL = """
    INSERT INTO citation_graph (
        citing_case_id, cited_case_id, treatment, context, metadata
    ) VALUES (
        $1::uuid, $2::uuid, $3::citation_treatment, $4, $5::jsonb
    )
    ON CONFLICT (citing_case_id, cited_case_id) DO UPDATE SET
        treatment = EXCLUDED.treatment,
        context   = EXCLUDED.context,
        metadata  = EXCLUDED.metadata
"""


# ── CLI group ──────────────────────────────────────────────────────────────────


@click.group()
@click.option(
    "--data-dir",
    default="data",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Root data directory.",
)
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    )


# ── run ────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--court",
    "court_codes",
    multiple=True,
    help="Court code(s) to load (e.g. NGSC NGCA). Defaults to all courts.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Max cases to load per court (for testing).",
)
@click.pass_context
def run(ctx: click.Context, court_codes: tuple[str, ...], limit: int | None) -> None:
    """Load cases and embedded chunks from disk into PostgreSQL."""
    data_dir: Path = ctx.obj["data_dir"]
    asyncio.run(_run_load(data_dir, court_codes, limit))


async def _run_load(
    data_dir: Path,
    court_codes: tuple[str, ...],
    limit: int | None,
) -> None:
    judgments_dir = data_dir / "processed" / "judgments"
    segments_dir = data_dir / "processed" / "segments"

    if court_codes:
        judgment_files = [judgments_dir / f"{c}.jsonl" for c in court_codes]
    else:
        judgment_files = sorted(judgments_dir.glob("*.jsonl"))

    loader = BulkCaseLoader()
    pool = await get_pool()

    total_cases_loaded = 0
    total_cases_errors = 0
    total_chunks_loaded = 0
    total_chunks_errors = 0

    try:
        for jfile in judgment_files:
            if not jfile.exists():
                click.echo(f"  [skip] {jfile} not found", err=True)
                continue

            court = jfile.stem
            click.echo(f"\nLoading cases from {court}...")

            # Read all judgment records for this court
            records: list[dict] = []
            with jfile.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            if limit:
                records = records[:limit]

            # ── Phase A: upsert cases, build slug→uuid map ────────────────────
            slug_to_uuid: dict[str, str] = {}
            cases_loaded = 0
            cases_errors = 0

            async with pool.acquire() as conn:
                for record in records:
                    try:
                        async with conn.transaction():
                            db_uuid = await loader.upsert_case(conn, record)
                            await loader.update_search_vector(conn, db_uuid)
                        slug_to_uuid[record["case_id"]] = db_uuid
                        cases_loaded += 1
                    except Exception as exc:
                        logger.error(
                            "load.case_error",
                            case_name=record.get("case_name"),
                            error=str(exc),
                        )
                        cases_errors += 1

            click.echo(f"  cases: {cases_loaded} loaded, {cases_errors} errors")
            total_cases_loaded += cases_loaded
            total_cases_errors += cases_errors

            # ── Phase B: load embedded chunks ─────────────────────────────────
            chunks_file = segments_dir / f"{court}_embedded.jsonl"
            if not chunks_file.exists():
                click.echo(f"  [skip] no chunks file for {court}")
                continue

            click.echo(f"Loading chunks from {court}...")

            chunks_by_case: dict[str, list[dict]] = defaultdict(list)
            with chunks_file.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("embedding"):
                        chunks_by_case[chunk["case_id"]].append(chunk)

            chunks_loaded = 0
            chunks_skipped = 0
            chunks_errors = 0

            async with pool.acquire() as conn:
                for text_slug, chunks in chunks_by_case.items():
                    db_uuid = slug_to_uuid.get(text_slug)
                    if db_uuid is None:
                        logger.warning("load.no_uuid", slug=text_slug)
                        chunks_skipped += len(chunks)
                        continue
                    try:
                        async with conn.transaction():
                            n = await loader.upsert_chunks(conn, db_uuid, chunks)
                        chunks_loaded += n
                    except Exception as exc:
                        logger.error(
                            "load.chunks_error",
                            slug=text_slug,
                            error=str(exc),
                        )
                        chunks_errors += len(chunks)

            click.echo(
                f"  chunks: {chunks_loaded} loaded, "
                f"{chunks_skipped} skipped, {chunks_errors} errors"
            )
            total_chunks_loaded += chunks_loaded
            total_chunks_errors += chunks_errors

    finally:
        await close_pool()

    result = {
        "cases_loaded": total_cases_loaded,
        "cases_errors": total_cases_errors,
        "chunks_loaded": total_chunks_loaded,
        "chunks_errors": total_chunks_errors,
    }
    click.echo("\n" + json.dumps(result, indent=2))


# ── graph ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--court",
    "court_codes",
    multiple=True,
    help="Restrict citation extraction to these courts (edges to any court still resolved).",
)
@click.pass_context
def graph(ctx: click.Context, court_codes: tuple[str, ...]) -> None:
    """Extract citations from judgment text and build the citation graph in DB."""
    data_dir: Path = ctx.obj["data_dir"]
    asyncio.run(_run_graph(data_dir, court_codes))


async def _run_graph(
    data_dir: Path,
    court_codes: tuple[str, ...],
) -> None:
    judgments_dir = data_dir / "processed" / "judgments"

    # ── Step 1: load all judgments to build registry ──────────────────────────
    click.echo("Building case registry from all courts...")
    all_files = sorted(judgments_dir.glob("*.jsonl"))

    all_judgments: list[dict] = []
    case_registry: dict[str, str] = {}  # slug → case_name
    citation_to_slug: dict[str, str] = {}  # "[2017] NGSC 23" → slug

    for jfile in all_files:
        with jfile.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                slug = j["case_id"]
                all_judgments.append(j)
                case_registry[slug] = j.get("case_name", "")
                if j.get("citation"):
                    citation_to_slug[j["citation"]] = slug

    click.echo(f"  {len(all_judgments)} judgments, {len(citation_to_slug)} with citations")

    # ── Step 2: build slug→DB UUID mapping via citation lookup ────────────────
    click.echo("Resolving case UUIDs from database...")
    pool = await get_pool()

    slug_to_uuid: dict[str, str] = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT id::text, citation FROM cases")
        for row in rows:
            cit = row["citation"]
            if cit and cit in citation_to_slug:
                slug = citation_to_slug[cit]
                slug_to_uuid[slug] = row["id"]

        click.echo(f"  {len(slug_to_uuid)} cases mapped to DB UUIDs")

        # ── Step 3: extract citations and insert edges ────────────────────────
        extractor = NigerianCitationExtractor()
        builder = CitationGraphBuilder()

        if court_codes:
            source_judgments = [j for j in all_judgments if j.get("court") in court_codes]
        else:
            source_judgments = all_judgments

        click.echo(f"Extracting citations from {len(source_judgments)} judgments...")

        total_edges = 0
        total_skipped = 0
        total_errors = 0

        async with pool.acquire() as conn:
            for judgment in source_judgments:
                citing_slug = judgment["case_id"]
                citing_uuid = slug_to_uuid.get(citing_slug)
                if citing_uuid is None:
                    total_skipped += 1
                    continue

                full_text = judgment.get("full_text", "")
                if not full_text:
                    total_skipped += 1
                    continue

                citations = extractor.extract_all(full_text)
                if not citations:
                    continue

                edges = builder.build_edges(citing_slug, citations, case_registry)
                resolved = [e for e in edges if e.cited_case_id is not None]
                if not resolved:
                    continue

                try:
                    async with conn.transaction():
                        await conn.execute(
                            "DELETE FROM citation_graph WHERE citing_case_id = $1::uuid",
                            citing_uuid,
                        )
                        for edge in resolved:
                            cited_uuid = slug_to_uuid.get(edge.cited_case_id)
                            if cited_uuid is None:
                                continue
                            await conn.execute(
                                _INSERT_EDGE_SQL,
                                citing_uuid,
                                cited_uuid,
                                edge.treatment.upper(),
                                edge.context,
                                json.dumps(
                                    {
                                        "cited_case_name": edge.cited_case_name,
                                        "citation_text": edge.cited_citation,
                                    }
                                ),
                            )
                            total_edges += 1
                except Exception as exc:
                    logger.error(
                        "graph.edge_error",
                        citing=citing_slug,
                        error=str(exc),
                    )
                    total_errors += 1

        # Refresh materialized view outside any transaction
        click.echo("Refreshing case_authority_scores materialized view...")
        async with pool.acquire() as conn:
            await conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY case_authority_scores")

    finally:
        await close_pool()

    result = {
        "edges_inserted": total_edges,
        "cases_skipped": total_skipped,
        "errors": total_errors,
    }
    click.echo(json.dumps(result, indent=2))


# ── status ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show DB counts vs processed file counts per court."""
    data_dir: Path = ctx.obj["data_dir"]
    asyncio.run(_run_status(data_dir))


async def _run_status(data_dir: Path) -> None:
    judgments_dir = data_dir / "processed" / "judgments"
    segments_dir = data_dir / "processed" / "segments"

    pool = await get_pool()
    report: dict = {}

    try:
        for jfile in sorted(judgments_dir.glob("*.jsonl")):
            court = jfile.stem
            try:
                court_enum = BulkCaseLoader.map_court(court)
            except ValueError:
                continue

            # File counts
            file_cases = sum(1 for line in jfile.open() if line.strip())
            chunks_file = segments_dir / f"{court}_embedded.jsonl"
            file_chunks = 0
            if chunks_file.exists():
                with chunks_file.open() as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            if d.get("embedding"):
                                file_chunks += 1

            # DB counts
            async with pool.acquire() as conn:
                db_cases = await conn.fetchval(
                    "SELECT COUNT(*) FROM cases WHERE court = $1::court_enum",
                    court_enum,
                )
                db_segments = await conn.fetchval(
                    """SELECT COUNT(cs.id) FROM case_segments cs
                       JOIN cases c ON cs.case_id = c.id
                       WHERE c.court = $1::court_enum""",
                    court_enum,
                )
                db_with_embedding = await conn.fetchval(
                    """SELECT COUNT(cs.id) FROM case_segments cs
                       JOIN cases c ON cs.case_id = c.id
                       WHERE c.court = $1::court_enum
                         AND cs.embedding IS NOT NULL""",
                    court_enum,
                )
                db_edges = await conn.fetchval(
                    """SELECT COUNT(*) FROM citation_graph cg
                       JOIN cases c ON cg.citing_case_id = c.id
                       WHERE c.court = $1::court_enum""",
                    court_enum,
                )

            report[court] = {
                "file_cases": int(file_cases),
                "db_cases": int(db_cases),
                "file_chunks": int(file_chunks),
                "db_segments": int(db_segments),
                "db_with_embed": int(db_with_embedding),
                "db_edges": int(db_edges),
            }

        # Totals
        async with pool.acquire() as conn:
            total_cases = int(await conn.fetchval("SELECT COUNT(*) FROM cases"))
            total_segments = int(await conn.fetchval("SELECT COUNT(*) FROM case_segments"))
            total_embedded = int(
                await conn.fetchval(
                    "SELECT COUNT(*) FROM case_segments WHERE embedding IS NOT NULL"
                )
            )
            total_edges = int(await conn.fetchval("SELECT COUNT(*) FROM citation_graph"))

        report["_totals"] = {
            "db_cases": total_cases,
            "db_segments": total_segments,
            "db_with_embedding": total_embedded,
            "db_edges": total_edges,
            "phase0_cases_target": 3000,
            "phase0_edges_target": 10000,
            "cases_gap": max(0, 3000 - total_cases),
            "edges_gap": max(0, 10000 - total_edges),
        }

    finally:
        await close_pool()

    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    cli()
