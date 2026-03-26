"""Ingestion orchestrator — manifest-tracked, resumable pipeline coordinator."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from ingestion.sources.nigerialii import (
    MVP_COURTS,
    CaseListEntry,
    Court,
    NigeriaLIICrawler,
    RawJudgment,
)

logger = structlog.get_logger(__name__)


class IngestionOrchestrator:
    """
    Coordinates the full ingestion pipeline across courts.

    Stage 1 — Discovery: crawl listing pages → collect all case URLs.
    Stage 2 — Fetch:     download each judgment page → save raw JSONL.

    Progress is tracked in ``data/manifest.json`` so the pipeline is
    fully resumable.  Re-running after an interruption skips work that
    is already done.

    Output layout::

        data/
        ├── raw/
        │   ├── nigerialii/        # cached HTML pages (by NigeriaLIICrawler)
        │   └── judgments/
        │       ├── NGSC.jsonl
        │       ├── NGCA.jsonl
        │       └── NGFCHC.jsonl
        └── manifest.json
    """

    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data_dir = data_dir
        self.crawler = NigeriaLIICrawler(raw_cache_dir=data_dir / "raw" / "nigerialii")
        self.manifest_path = data_dir / "manifest.json"
        self.manifest = self._load_manifest()

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        """Load existing manifest or create an empty one."""
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "discovered": {},  # court_code → [{ case_name, case_url, … }]
            "fetched": [],  # [case_url, …] — successfully fetched
            "failed": [],  # [case_url, …] — errored
            "stats": {},  # court_code → int
        }

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Pipeline stages ───────────────────────────────────────────────────────

    async def run_discovery(
        self,
        courts: list[Court] | None = None,
    ) -> None:
        """
        Stage 1: crawl court listing pages to collect all case URLs.

        Courts already in the manifest are skipped (idempotent).
        """
        targets = courts or MVP_COURTS
        async with self.crawler:
            for court in targets:
                if court.value in self.manifest["discovered"]:
                    count = len(self.manifest["discovered"][court.value])
                    logger.info(
                        "orchestrator.discovery_skip",
                        court=court.value,
                        count=count,
                    )
                    continue

                entries = await self.crawler.crawl_court(court)

                self.manifest["discovered"][court.value] = [
                    {
                        "case_name": e.case_name,
                        "case_url": e.case_url,
                        "judgment_date": str(e.judgment_date),
                        "citation": e.citation,
                        "case_number": e.case_number,
                        "labels": e.labels,
                    }
                    for e in entries
                ]
                self.manifest["stats"][court.value] = len(entries)
                self._save_manifest()

                logger.info(
                    "orchestrator.discovery_done",
                    court=court.value,
                    count=len(entries),
                )

    async def run_fetch(
        self,
        courts: list[Court] | None = None,
        limit: int | None = None,
        batch_size: int = 50,
    ) -> None:
        """
        Stage 2: fetch individual judgment pages and save raw JSONL.

        ``limit`` caps the number of judgments fetched per run (useful for
        testing).  ``batch_size`` controls how often the manifest is flushed
        to disk.
        """
        already_fetched = set(self.manifest["fetched"])
        failed = set(self.manifest["failed"])
        targets = courts or MVP_COURTS

        async with self.crawler:
            for court in targets:
                if court.value not in self.manifest["discovered"]:
                    logger.warning(
                        "orchestrator.fetch_skip_no_discovery",
                        court=court.value,
                    )
                    continue

                entries_data: list[dict] = self.manifest["discovered"][court.value]
                pending = [
                    e
                    for e in entries_data
                    if e["case_url"] not in already_fetched and e["case_url"] not in failed
                ]

                if limit is not None:
                    pending = pending[:limit]

                logger.info(
                    "orchestrator.fetch_start",
                    court=court.value,
                    pending=len(pending),
                    already_fetched=len(already_fetched),
                )

                for i, entry_data in enumerate(pending):
                    entry = CaseListEntry(
                        case_name=entry_data["case_name"],
                        case_url=entry_data["case_url"],
                        judgment_date=entry_data.get("judgment_date"),  # type: ignore[arg-type]
                        citation=entry_data.get("citation"),
                        case_number=entry_data.get("case_number"),
                        labels=entry_data.get("labels", []),
                        court=court,
                    )

                    try:
                        judgment = await self.crawler.crawl_judgment(entry)
                        await self._save_raw_judgment(judgment)
                        self.manifest["fetched"].append(entry.case_url)
                        already_fetched.add(entry.case_url)

                        logger.info(
                            "orchestrator.fetched",
                            court=court.value,
                            progress=f"{i + 1}/{len(pending)}",
                            case=entry.case_name[:60],
                        )
                    except Exception as exc:
                        self.manifest["failed"].append(entry.case_url)
                        failed.add(entry.case_url)
                        logger.error(
                            "orchestrator.fetch_error",
                            court=court.value,
                            case=entry.case_name[:60],
                            error=str(exc),
                        )

                    if (i + 1) % batch_size == 0:
                        self._save_manifest()

                self._save_manifest()

    # ── Storage ───────────────────────────────────────────────────────────────

    async def _save_raw_judgment(self, judgment: RawJudgment) -> None:
        """Append a raw judgment as a JSON line to the per-court JSONL file."""
        output_path = self.data_dir / "raw" / "judgments" / f"{judgment.court.value}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "case_name": judgment.case_name,
            "source_url": judgment.source_url,
            "court": judgment.court.value,
            "citation": judgment.media_neutral_citation,
            "case_number": judgment.case_number,
            "judges": judgment.judges,
            "judgment_date": str(judgment.judgment_date),
            "labels": judgment.labels,
            "full_text": judgment.full_text,
            "full_html": judgment.full_html,
            "crawled_at": judgment.crawled_at.isoformat(),
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary of the current manifest state."""
        return {
            "discovered": {
                court: len(entries) for court, entries in self.manifest["discovered"].items()
            },
            "fetched": len(self.manifest["fetched"]),
            "failed": len(self.manifest["failed"]),
        }
