#!/usr/bin/env python3
"""CLI entry point for the corpus embedding pipeline.

Usage examples::

    # Embed all unembedded chunks in data/processed/segments/
    python scripts/embed.py run

    # Embed only Supreme Court chunks
    python scripts/embed.py run --court NGSC

    # Show embedding progress across all courts
    python scripts/embed.py status

    # Load embeddings from a file into PostgreSQL
    python scripts/embed.py load --file data/processed/segments/NGSC_embedded.jsonl
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import structlog

from config import settings
from ingestion.embedding.chunker import LegalTextChunker
from ingestion.embedding.embedder import CorpusEmbedder

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger(__name__)


@click.group()
@click.option(
    "--data-dir",
    default="data",
    show_default=True,
    help="Root data directory.",
)
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    """Corpus embedding pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)


# ── run ───────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--court",
    "court_codes",
    multiple=True,
    help="Court code(s) to embed (e.g. NGSC). Defaults to all.",
)
@click.option(
    "--batch-size",
    default=128,
    show_default=True,
    type=int,
    help="Texts per Voyage AI API call.",
)
@click.option(
    "--sleep",
    "sleep_between_batches",
    default=0.0,
    show_default=True,
    type=float,
    help="Seconds to sleep between batches (use ~21 for free-tier rate limits).",
)
@click.pass_context
def run(
    ctx: click.Context, court_codes: tuple[str, ...], batch_size: int, sleep_between_batches: float
) -> None:
    """Chunk and embed all unembedded case segments."""
    data_dir: Path = ctx.obj["data_dir"]
    processed_dir = data_dir / "processed" / "judgments"
    segments_dir = data_dir / "processed" / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(processed_dir.glob("*.jsonl"))
    if court_codes:
        jsonl_files = [f for f in jsonl_files if f.stem in court_codes]

    if not jsonl_files:
        click.echo("No processed judgment files found. Run crawl + parse first.")
        return

    chunker = LegalTextChunker()
    embedder = CorpusEmbedder(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        batch_size=batch_size,
        sleep_between_batches=sleep_between_batches,
    )

    total_embedded = 0

    async def _embed_all() -> None:
        nonlocal total_embedded
        for jsonl_path in jsonl_files:
            court_code = jsonl_path.stem
            click.echo(f"Processing {court_code}...")

            # Chunk all judgments in this file
            all_chunks = []
            with jsonl_path.open(encoding="utf-8") as f:
                for line in f:
                    judgment = json.loads(line)
                    chunks = chunker.chunk(judgment)
                    all_chunks.extend(chunks)

            click.echo(f"  {len(all_chunks)} chunks generated")

            output_path = segments_dir / f"{court_code}_embedded.jsonl"

            # Load any existing embeddings so re-runs skip already-embedded chunks
            existing_embeddings: dict[str, list[float]] = {}
            if output_path.exists():
                with output_path.open(encoding="utf-8") as ef:
                    for line in ef:
                        rec = json.loads(line)
                        if rec.get("embedding") is not None:
                            existing_embeddings[rec["chunk_id"]] = rec["embedding"]

            # Write chunks to disk (with existing embeddings preserved)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as wf:
                for chunk in all_chunks:
                    record = {
                        "chunk_id": chunk.chunk_id,
                        "case_id": chunk.case_id,
                        "segment_type": chunk.segment_type,
                        "content": chunk.content,
                        "embedding": existing_embeddings.get(chunk.chunk_id),
                        "court": chunk.court,
                        "year": chunk.year,
                        "area_of_law": chunk.area_of_law,
                        "case_name": chunk.case_name,
                        "citation": chunk.citation,
                    }
                    wf.write(json.dumps(record) + "\n")

            # embed_file reads the file, skips already-embedded, embeds the rest
            n = await embedder.embed_file(
                chunks_path=output_path,
                output_path=output_path,
            )
            total_embedded += n
            click.echo(f"  {n} chunks newly embedded → {output_path}")

    asyncio.run(_embed_all())
    click.echo(json.dumps({"total_newly_embedded": total_embedded}, indent=2))


# ── status ────────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show embedding progress: total chunks and how many have embeddings."""
    data_dir: Path = ctx.obj["data_dir"]
    segments_dir = data_dir / "processed" / "segments"

    if not segments_dir.exists():
        click.echo("No segments directory found. Run 'embed run' first.")
        return

    report = {}
    for path in sorted(segments_dir.glob("*_embedded.jsonl")):
        court = path.stem.replace("_embedded", "")
        total = 0
        embedded = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                total += 1
                if data.get("embedding") is not None:
                    embedded += 1
        report[court] = {"total_chunks": total, "embedded": embedded}

    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    cli()
