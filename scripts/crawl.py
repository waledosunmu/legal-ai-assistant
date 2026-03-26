#!/usr/bin/env python3
"""CLI entry point for the NigeriaLII crawl pipeline.

Usage examples::

    # Discover all cases for the Supreme Court
    python scripts/crawl.py --court NGSC discover

    # Fetch the first 20 judgments for NGSC
    python scripts/crawl.py --court NGSC fetch --limit 20

    # Run discovery then fetch for all MVP courts
    python scripts/crawl.py all

    # Show manifest summary
    python scripts/crawl.py status

    # Crawl Nigeria Court of Appeal judgments (REST API + PDF extraction)
    python scripts/crawl.py coa
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.orchestrator import IngestionOrchestrator  # noqa: E402
from ingestion.sources.nigerialii import MVP_COURTS, Court  # noqa: E402

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

COURT_CHOICES = [c.value for c in Court]


def _parse_court(court_code: str) -> Court:
    try:
        return Court(court_code.upper())
    except ValueError as exc:
        raise click.BadParameter(
            f"Unknown court '{court_code}'. " f"Valid codes: {', '.join(COURT_CHOICES)}"
        ) from exc


@click.group()
@click.option(
    "--data-dir",
    default="data",
    show_default=True,
    help="Root data directory for raw HTML cache, JSONL output, and manifest.",
)
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    """NigeriaLII ingestion pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)


# ── discover ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--court",
    "court_codes",
    multiple=True,
    help=("Court code(s) to discover (repeatable). " "Defaults to all MVP courts if omitted."),
)
@click.pass_context
def discover(ctx: click.Context, court_codes: tuple[str, ...]) -> None:
    """Crawl court listing pages and collect all case URLs into the manifest."""
    data_dir: Path = ctx.obj["data_dir"]
    courts = [_parse_court(c) for c in court_codes] if court_codes else None

    orchestrator = IngestionOrchestrator(data_dir=data_dir)
    asyncio.run(orchestrator.run_discovery(courts=courts))

    click.echo(json.dumps(orchestrator.summary(), indent=2))


# ── fetch ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--court",
    "court_codes",
    multiple=True,
    help="Court code(s) to fetch. Defaults to all discovered courts.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Max judgments to fetch per court (useful for testing).",
)
@click.pass_context
def fetch(ctx: click.Context, court_codes: tuple[str, ...], limit: int | None) -> None:
    """Fetch individual judgment pages for discovered cases."""
    data_dir: Path = ctx.obj["data_dir"]
    courts = [_parse_court(c) for c in court_codes] if court_codes else None

    orchestrator = IngestionOrchestrator(data_dir=data_dir)
    asyncio.run(orchestrator.run_fetch(courts=courts, limit=limit))

    click.echo(json.dumps(orchestrator.summary(), indent=2))


# ── all ───────────────────────────────────────────────────────────────────────


@cli.command(name="all")
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Max judgments to fetch per court.",
)
@click.pass_context
def run_all(ctx: click.Context, limit: int | None) -> None:
    """Run discovery then fetch for all MVP courts."""
    data_dir: Path = ctx.obj["data_dir"]

    orchestrator = IngestionOrchestrator(data_dir=data_dir)

    async def _run() -> None:
        await orchestrator.run_discovery(courts=MVP_COURTS)
        await orchestrator.run_fetch(courts=MVP_COURTS, limit=limit)

    asyncio.run(_run())
    click.echo(json.dumps(orchestrator.summary(), indent=2))


# ── status ────────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Print manifest summary (discovered / fetched / failed counts)."""
    data_dir: Path = ctx.obj["data_dir"]
    orchestrator = IngestionOrchestrator(data_dir=data_dir)
    click.echo(json.dumps(orchestrator.summary(), indent=2))


# ── coa ───────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--cache-dir",
    default="data/raw/coa",
    show_default=True,
    help="Directory for downloaded PDFs and JSON cache.",
)
@click.pass_context
def coa(ctx: click.Context, cache_dir: str) -> None:
    """Crawl Nigeria Court of Appeal judgments (REST API + PDF extraction).

    Downloads all judgments from the CoA REST API, fetches PDFs from Google
    Drive, extracts text, and writes records to
    data/raw/judgments/NGCA_COA.jsonl in the standard parse.py input format.
    """
    from ingestion.sources.courtofappeal import CourtOfAppealCrawler

    data_dir: Path = ctx.obj["data_dir"]
    raw_dir = data_dir / "raw" / "judgments"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "NGCA_COA.jsonl"

    async def _run() -> int:
        crawler = CourtOfAppealCrawler(cache_dir=Path(cache_dir))
        judgments = await crawler.crawl_all()

        written = 0
        with output_path.open("w", encoding="utf-8") as f:
            for j in judgments:
                if not j.full_text:
                    click.echo(f"  [skip] {j.case_id} — no text extracted", err=True)
                    continue
                # Normalise to the format expected by scripts/parse.py
                record = {
                    "case_id": j.case_id,
                    "case_name": j.case_name,
                    "citation": j.citation,
                    "court": j.court,
                    "judgment_date": j.date_decided,
                    "judges": j.judges,
                    "labels": j.area_of_law,
                    "source_url": j.metadata.get("gdrive_link", ""),
                    "full_text": j.full_text,
                    "metadata": j.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
        return written

    total = asyncio.run(_run())
    click.echo(f"\nCoA crawl complete: {total} judgments written to {output_path}")
    click.echo("Next: python scripts/parse.py run --court NGCA_COA")


if __name__ == "__main__":
    cli()
