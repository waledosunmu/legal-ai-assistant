#!/usr/bin/env python3
"""CLI entry point for the judgment parsing / segmentation pipeline.

Usage examples::

    # Parse all courts
    python scripts/parse.py run

    # Parse only Supreme Court
    python scripts/parse.py run --court NGSC

    # Show parse progress
    python scripts/parse.py status
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
import structlog

from ingestion.segmentation.structural import StructuralSegmenter
from ingestion.segmentation.nlp_rules import NLPSegmentClassifier
from ingestion.segmentation.models import SegmentType

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

_STRUCTURAL = StructuralSegmenter()
_NLP = NLPSegmentClassifier()


def _make_case_id(record: dict) -> str:
    """Derive a stable case_id from the source URL slug."""
    url = record.get("source_url", "")
    # https://nigerialii.org/akn/ng/judgment/ngsc/2017/5/eng@2017-06-22
    # → akn_ng_judgment_ngsc_2017_5
    path = re.sub(r"^https?://[^/]+", "", url)  # strip domain
    path = re.sub(r"/eng@.*$", "", path)         # strip language@date suffix
    slug = path.strip("/").replace("/", "_")
    return slug or record.get("citation", "unknown").replace(" ", "_").replace("[", "").replace("]", "")


def _extract_year(record: dict) -> int | None:
    date_str = record.get("judgment_date", "")
    m = re.match(r"(\d{4})", date_str)
    return int(m.group(1)) if m else None


def _segment_record(record: dict) -> dict:
    """Run structural + NLP segmentation on a raw crawler record."""
    full_text = record.get("full_text", "")

    # Pass 1: structural segmentation
    raw_segments = _STRUCTURAL.segment(full_text)

    # Pass 2: NLP rescoring
    refined = _NLP.reclassify(raw_segments)

    # Convert segments to serialisable dicts
    segments = [
        {
            "segment_type": seg.segment_type.value,
            "content": seg.content,
            "position": seg.position,
            "confidence": seg.confidence,
        }
        for seg in refined
    ]

    # Extract ratio and holdings from high-confidence segments
    ratio_decidendi: str | None = None
    holdings: list[dict] = []
    for seg in refined:
        if seg.segment_type == SegmentType.RATIO and seg.confidence >= 0.5 and not ratio_decidendi:
            ratio_decidendi = seg.content
        elif seg.segment_type == SegmentType.HOLDING and seg.confidence >= 0.5:
            holdings.append({"issue": "", "determination": seg.content, "reasoning": ""})

    return {
        "case_id":         _make_case_id(record),
        "case_name":       record.get("case_name", ""),
        "citation":        record.get("citation"),
        "court":           record.get("court", ""),
        "year":            _extract_year(record),
        "judges":          record.get("judges", []),
        "judgment_date":   record.get("judgment_date"),
        "source_url":      record.get("source_url"),
        "area_of_law":     record.get("labels", []),
        "ratio_decidendi": ratio_decidendi,
        "holdings":        holdings,
        "segments":        segments,
        "full_text":       full_text,
    }


@click.group()
@click.option(
    "--data-dir",
    default="data",
    show_default=True,
    help="Root data directory.",
)
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    """Judgment parsing / segmentation pipeline CLI."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = Path(data_dir)


@cli.command()
@click.option(
    "--court",
    "court_codes",
    multiple=True,
    help="Court code(s) to parse (e.g. NGSC). Defaults to all.",
)
@click.pass_context
def run(ctx: click.Context, court_codes: tuple[str, ...]) -> None:
    """Parse raw judgment JSONL files into segmented processed records."""
    data_dir: Path = ctx.obj["data_dir"]
    raw_dir = data_dir / "raw" / "judgments"
    processed_dir = data_dir / "processed" / "judgments"
    processed_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(raw_dir.glob("*.jsonl"))
    if court_codes:
        jsonl_files = [f for f in jsonl_files if f.stem in court_codes]

    if not jsonl_files:
        click.echo("No raw judgment files found. Run crawl fetch first.")
        return

    total_parsed = 0
    for jsonl_path in sorted(jsonl_files):
        court_code = jsonl_path.stem
        output_path = processed_dir / jsonl_path.name
        count = 0

        click.echo(f"Parsing {court_code}...")
        with jsonl_path.open(encoding="utf-8") as fin, \
             output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                record = json.loads(line)
                try:
                    processed = _segment_record(record)
                    fout.write(json.dumps(processed) + "\n")
                    count += 1
                except Exception as exc:
                    logger.warning(
                        "parse.skipped",
                        case=record.get("case_name", "?"),
                        error=str(exc),
                    )

        total_parsed += count
        click.echo(f"  {count} records → {output_path}")

    click.echo(json.dumps({"total_parsed": total_parsed}, indent=2))


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show parse status: raw counts vs processed counts per court."""
    data_dir: Path = ctx.obj["data_dir"]
    raw_dir = data_dir / "raw" / "judgments"
    processed_dir = data_dir / "processed" / "judgments"

    report = {}
    for raw_path in sorted(raw_dir.glob("*.jsonl")):
        court = raw_path.stem
        raw_count = sum(1 for _ in raw_path.open(encoding="utf-8"))
        proc_path = processed_dir / raw_path.name
        proc_count = sum(1 for _ in proc_path.open(encoding="utf-8")) if proc_path.exists() else 0
        report[court] = {"raw": raw_count, "processed": proc_count}

    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    cli()
