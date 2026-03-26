"""Crawler for the Nigeria Court of Appeal judgment database.

The CoA website (courtofappeal.gov.ng) exposes a REST API at:
  https://coa-admin.courtofappeal.gov.ng/api/judgement/search

Judgment PDFs are hosted on Google Drive.  We:
  1. Enumerate all judgments via the API (paginated).
  2. Download each PDF from Google Drive.
  3. Extract text with PDFTextExtractor (pdfplumber → pymupdf → OCR).
  4. Return RawJudgment objects compatible with the existing pipeline.

Usage::

    crawler = CourtOfAppealCrawler(cache_dir=Path("data/raw/coa"))
    judgments = await crawler.crawl_all()
    for j in judgments:
        # save to data/raw/coa/{case_id}.json
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from ingestion.sources.pdf_extractor import PDFTextExtractor

logger = logging.getLogger(__name__)

_API_BASE = "https://coa-admin.courtofappeal.gov.ng/api"
_GDRIVE_DOWNLOAD = "https://drive.google.com/uc?export=download&id={file_id}"
_GDRIVE_CONFIRM = (
    "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
)


# ── Data models ────────────────────────────────────────────────────────────────


@dataclass
class CoAJudgmentRecord:
    """Raw record as returned by the CoA API."""

    id: int
    file_no: str
    parties: str
    justice: str
    date: str
    subject_matter: str | None
    substantial_issue: str | None
    division: str
    download_link: str


@dataclass
class RawCoAJudgment:
    """Normalized judgment for downstream pipeline compatibility."""

    case_id: str
    source: str = "NGCA_COA"
    court: str = "NGCA"
    citation: str = ""
    case_name: str = ""
    date_decided: str = ""
    judges: list[str] = field(default_factory=list)
    area_of_law: list[str] = field(default_factory=list)
    full_text: str = ""
    metadata: dict = field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_gdrive_file_id(url: str) -> str | None:
    """Extract file ID from a Google Drive share URL."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def _slugify(text: str) -> str:
    """Create a filesystem-safe slug from text."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")[:80]


def _make_case_id(record: CoAJudgmentRecord) -> str:
    """Generate a stable case ID for a CoA judgment record."""
    file_no = re.sub(r"\s+", "_", (record.file_no or "").strip())
    file_no = re.sub(r"[^\w_-]", "", file_no)[:40]
    if file_no:
        return f"ngca_coa_{file_no}".lower()
    # Fall back to a hash of the first party name
    h = hashlib.sha1(record.parties.encode()).hexdigest()[:8]
    return f"ngca_coa_{h}"


def _parse_parties(parties_text: str) -> str:
    """Extract a short case name from raw parties text (Appellant vs Respondent)."""
    lines = [ln.strip() for ln in parties_text.strip().splitlines() if ln.strip()]
    if not lines:
        return parties_text.strip()[:120]

    # Try to extract first appellant and first respondent across "AND"
    appellant = ""
    respondent = ""
    state = "appellant"
    for line in lines:
        if line.upper() in ("AND", "VS", "V.") or re.match(r"^\d+\.\s*AND$", line, re.I):
            state = "respondent"
            continue
        # Skip numbered lines like "1.", "2."
        clean = re.sub(r"^\d+\.\s*", "", line).strip()
        if not clean:
            continue
        if state == "appellant" and not appellant:
            appellant = clean
        elif state == "respondent" and not respondent:
            respondent = clean

    if appellant and respondent:
        return f"{appellant} v. {respondent}"
    return lines[0][:120]


def _parse_judges(justice_text: str) -> list[str]:
    """Split semicolon/comma-separated judge names."""
    judges = []
    for part in re.split(r"[;,]", justice_text):
        name = part.strip()
        if name:
            judges.append(name)
    return judges


# ── Main crawler ───────────────────────────────────────────────────────────────


class CourtOfAppealCrawler:
    """Crawl judgments from the Nigeria Court of Appeal API.

    Args:
        cache_dir: Directory for downloaded PDFs and JSON cache.
        page_size: Records per API request.
        request_delay: Seconds between API requests.
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/raw/coa"),
        page_size: int = 100,
        request_delay: float = 1.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.pdf_dir = cache_dir / "pdfs"
        self.page_size = page_size
        self.request_delay = request_delay
        self.extractor = PDFTextExtractor()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    # ── API enumeration ────────────────────────────────────────────────────────

    async def fetch_all_records(self) -> list[CoAJudgmentRecord]:
        """Fetch all judgment metadata from the CoA API."""
        records: list[CoAJudgmentRecord] = []
        page = 1

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params = {"p": page, "ps": self.page_size}
                resp = await client.get(f"{_API_BASE}/judgement/search", params=params)
                resp.raise_for_status()
                body = resp.json()

                data = body.get("data", [])
                if not data:
                    break

                for item in data:
                    records.append(
                        CoAJudgmentRecord(
                            id=item["id"],
                            file_no=(item.get("file_no") or "").strip(),
                            parties=(item.get("parties") or "").strip(),
                            justice=(item.get("justice") or "").strip(),
                            date=(item.get("date") or "").strip(),
                            subject_matter=item.get("subject_matter"),
                            substantial_issue=item.get("substantial_issue"),
                            division=(item.get("division") or "").strip(),
                            download_link=(item.get("download_link") or "").strip(),
                        )
                    )

                meta = body.get("meta", {}).get("pagination", {})
                total_pages = meta.get("pageCount", 1)
                logger.info(
                    "coa.fetch_page page=%d/%d records_so_far=%d",
                    page,
                    total_pages,
                    len(records),
                )

                if page >= total_pages:
                    break
                page += 1
                await asyncio.sleep(self.request_delay)

        return records

    # ── PDF download ───────────────────────────────────────────────────────────

    async def download_pdf(self, record: CoAJudgmentRecord) -> Path | None:
        """Download the PDF for a record to cache dir. Returns local path or None."""
        file_id = _extract_gdrive_file_id(record.download_link)
        if not file_id:
            logger.warning("coa.no_file_id record_id=%d link=%s", record.id, record.download_link)
            return None

        local_path = self.pdf_dir / f"{file_id}.pdf"
        if local_path.exists() and local_path.stat().st_size > 1024:
            return local_path

        async with httpx.AsyncClient(
            timeout=120,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; legal-ai-research/1.0)"},
        ) as client:
            # Try direct download first
            url = _GDRIVE_DOWNLOAD.format(file_id=file_id)
            resp = await client.get(url)

            # Google may redirect to a confirmation page for large files
            if resp.status_code == 200 and "content-type" in resp.headers:
                ct = resp.headers.get("content-type", "")
                if "pdf" not in ct.lower() and len(resp.content) < 50_000:
                    # Likely a confirmation/HTML page — use confirm URL
                    url = _GDRIVE_CONFIRM.format(file_id=file_id)
                    resp = await client.get(url)

            if resp.status_code != 200:
                logger.warning(
                    "coa.download_failed record_id=%d status=%d", record.id, resp.status_code
                )
                return None

            local_path.write_bytes(resp.content)
            logger.info("coa.pdf_downloaded record_id=%d size=%d", record.id, len(resp.content))

        return local_path

    # ── Full crawl ─────────────────────────────────────────────────────────────

    async def crawl_all(self) -> list[RawCoAJudgment]:
        """Fetch all judgments, download PDFs, extract text, return RawCoAJudgment list."""
        logger.info("coa.crawl_start")
        records = await self.fetch_all_records()
        logger.info("coa.records_fetched count=%d", len(records))

        judgments: list[RawCoAJudgment] = []

        for i, record in enumerate(records):
            case_id = _make_case_id(record)

            # Check JSON cache
            cache_file = self.cache_dir / f"{case_id}.json"
            if cache_file.exists():
                with cache_file.open() as f:
                    judgments.append(RawCoAJudgment(**json.load(f)))
                logger.debug("coa.cache_hit case_id=%s", case_id)
                continue

            # Download PDF
            pdf_path = await self.download_pdf(record)
            full_text = ""
            if pdf_path:
                full_text = self.extractor.extract(pdf_path)
                logger.info("coa.extracted case_id=%s chars=%d", case_id, len(full_text))
            else:
                logger.warning("coa.no_pdf case_id=%s", case_id)

            judgment = RawCoAJudgment(
                case_id=case_id,
                source="NGCA_COA",
                court="NGCA",
                citation=record.file_no,
                case_name=_parse_parties(record.parties),
                date_decided=record.date,
                judges=_parse_judges(record.justice),
                area_of_law=[record.subject_matter] if record.subject_matter else [],
                full_text=full_text,
                metadata={
                    "coa_id": record.id,
                    "division": record.division,
                    "substantial_issue": record.substantial_issue,
                    "gdrive_link": record.download_link,
                },
            )
            judgments.append(judgment)

            # Write JSON cache
            with cache_file.open("w") as f:
                json.dump(asdict(judgment), f, ensure_ascii=False)

            # Rate limiting
            if i < len(records) - 1:
                await asyncio.sleep(self.request_delay)

        logger.info("coa.crawl_done total=%d", len(judgments))
        return judgments
