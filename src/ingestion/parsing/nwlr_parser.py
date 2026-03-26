"""Parser for NWLR Online case HTML + metadata into RawJudgment objects.

NWLR Online serves judgments as HTML fragments from /auth/load-case/{id}.
Metadata (case name, court, part, page range, date) comes from
/auth/case-details/{id} and is merged here.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import structlog
from bs4 import BeautifulSoup

from ingestion.sources.nigerialii import Court, RawJudgment

logger = structlog.get_logger(__name__)

# NWLR court string → our Court enum
_COURT_MAP: dict[str, Court] = {
    "S.C":   Court.SUPREME_COURT,
    "SC":    Court.SUPREME_COURT,
    "C.A":   Court.COURT_OF_APPEAL,
    "CA":    Court.COURT_OF_APPEAL,
    "F.H.C": Court.FEDERAL_HIGH_COURT,
    "FHC":   Court.FEDERAL_HIGH_COURT,
}

_JUDGE_TITLE_RE = re.compile(
    r"([A-Z][A-Z .'-]{2,40?}(?:JSC|JCA|FJCA|J\b|CJN|PCA|JCA))"
)
_HEADNOTE_STOP_RE = re.compile(
    r"(JUDGMENT|RULING|DECISION|LEAD JUDGMENT)",
    re.IGNORECASE,
)


class NWLRParser:
    """
    Parse NWLR Online HTML + case-details metadata into a :class:`RawJudgment`.

    The ``metadata`` dict is the ``data`` field from ``/auth/case-details/{id}``,
    containing keys: ``case_name``, ``court``, ``part``, ``page_start``,
    ``page_end``, ``volume``, ``year``, ``published_date``.
    """

    def parse(self, html: str, metadata: dict[str, Any]) -> RawJudgment:
        """Convert raw HTML + metadata dict into a RawJudgment."""
        soup = BeautifulSoup(html, "lxml")

        full_text = self._extract_text(soup)
        full_html = html

        court = self._map_court(metadata.get("court", ""))
        citation = self._build_citation(metadata)
        judges = self._extract_judges(full_text)
        judgment_date = self._parse_date(metadata.get("published_date", ""))
        headnotes = self._extract_headnotes(soup)
        source_url = (
            f"https://www.nwlronline.com/cases/"
            f"{metadata.get('part', 0)}_1_{metadata.get('page_start', 0)}"
        )

        # Merge headnotes into labels list for downstream use
        labels: list[str] = []
        if headnotes:
            labels.append(f"headnote:{headnotes[:200]}")

        return RawJudgment(
            case_name=metadata.get("case_name", "Unknown"),
            source_url=source_url,
            court=court,
            media_neutral_citation=citation,
            case_number=None,  # NWLR does not expose case numbers via API
            judges=judges,
            judgment_date=judgment_date,
            language="English",
            labels=labels,
            full_text=full_text,
            full_html=full_html,
            source="nwlronline",
        )

    # ── Extraction helpers ─────────────────────────────────────────────────

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean plain text from the HTML."""
        # Remove script/style noise
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=False)

    def _map_court(self, court_str: str) -> Court:
        """Map NWLR court abbreviation to our Court enum."""
        normalized = court_str.strip().upper().replace(".", "")
        for key, court in _COURT_MAP.items():
            if normalized == key.replace(".", "").upper():
                return court
        logger.warning("nwlr_parser.unknown_court", court=court_str)
        return Court.SUPREME_COURT  # default — most NWLR cases are SC

    def _build_citation(self, metadata: dict[str, Any]) -> str:
        """Build an NWLR citation string like '(2026) 4 NWLR (Pt. 2034) 349'."""
        year = metadata.get("year", "")
        volume = metadata.get("volume", "")
        part = metadata.get("part", "")
        page_start = metadata.get("page_start", "")

        if year and part and page_start:
            vol_part = f"{volume} " if volume else ""
            return f"({year}) {vol_part}NWLR (Pt. {part}) {page_start}"
        return ""

    def _extract_judges(self, text: str) -> list[str]:
        """Extract judge names from the first 3,000 characters of judgment text."""
        search_region = text[:3000]
        matches = _JUDGE_TITLE_RE.findall(search_region)
        seen: set[str] = set()
        judges: list[str] = []
        for m in matches:
            clean = m.strip().rstrip(",;")
            if clean not in seen:
                seen.add(clean)
                judges.append(clean)
        return judges[:10]  # cap at 10 to avoid garbage

    def _extract_headnotes(self, soup: BeautifulSoup) -> str:
        """Extract headnote text (appears before the main judgment body)."""
        text = soup.get_text(separator="\n")
        lines = text.splitlines()
        headnote_lines: list[str] = []
        for line in lines:
            if _HEADNOTE_STOP_RE.search(line):
                break
            stripped = line.strip()
            if stripped:
                headnote_lines.append(stripped)
        return "\n".join(headnote_lines[:30])  # first 30 non-empty lines

    def _parse_date(self, date_str: str) -> date | None:
        """Parse ISO date string from API response."""
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
        return None
