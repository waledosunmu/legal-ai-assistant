"""NigeriaLII web crawler — rate-limited, disk-cached, resumable."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import httpx
import structlog
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

BASE_URL = "https://nigerialii.org"

USER_AGENT = "LegalAIAssistant/0.1 (Nigerian legal research tool; contact@legalaiassistant.ng)"


class Court(StrEnum):
    SUPREME_COURT = "NGSC"
    COURT_OF_APPEAL = "NGCA"
    FEDERAL_HIGH_COURT = "NGFCHC"
    LAGOS_HIGH_COURT = "NGLAHC"
    KANO_HIGH_COURT = "NGKNHC"
    BAUCHI_HIGH_COURT = "NGBAHC"
    BENUE_HIGH_COURT = "NGBEHC"
    EBONYI_HIGH_COURT = "NGEBHC"


# Priority order for MVP crawl — highest authority first
MVP_COURTS: list[Court] = [
    Court.SUPREME_COURT,
    Court.COURT_OF_APPEAL,
    Court.FEDERAL_HIGH_COURT,
    Court.LAGOS_HIGH_COURT,
]


@dataclass
class CaseListEntry:
    """Metadata extracted from court listing pages."""

    case_name: str
    case_url: str  # Relative URL to full judgment
    judgment_date: date | None
    citation: str | None  # Media neutral citation e.g. [2017] NGSC 5
    case_number: str | None  # Suit number e.g. SC. 373/2015
    labels: list[str] = field(default_factory=list)
    court: Court | None = None


@dataclass
class RawJudgment:
    """Full judgment data extracted from individual judgment pages."""

    # From listing page
    case_name: str
    source_url: str
    court: Court

    # From judgment page metadata
    media_neutral_citation: str | None = None
    case_number: str | None = None
    judges: list[str] = field(default_factory=list)
    judgment_date: date | None = None
    language: str = "English"
    labels: list[str] = field(default_factory=list)

    # From judgment page body
    full_text: str = ""
    full_html: str = ""

    # Ingestion metadata
    crawled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = "nigerialii"


class NigeriaLIICrawler:
    """
    Respectful crawler for NigeriaLII.

    Design principles:
    1. Rate-limited: 1 request per ``rate_limit_seconds`` (default 2s)
    2. Resumable: raw HTML cached to disk; re-runs skip already-cached pages
    3. Idempotent: re-crawling the same URL returns the cached result
    4. Async context manager: manages the shared httpx client lifecycle

    Usage::

        async with NigeriaLIICrawler() as crawler:
            entries = await crawler.crawl_court(Court.SUPREME_COURT)
            for entry in entries[:10]:
                judgment = await crawler.crawl_judgment(entry)
    """

    def __init__(
        self,
        raw_cache_dir: Path = Path("data/raw/nigerialii"),
        rate_limit_seconds: float = 2.0,
        max_concurrent: int = 1,
    ) -> None:
        self.raw_cache_dir = raw_cache_dir
        self.rate_limit_seconds = rate_limit_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: httpx.AsyncClient | None = None
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> NigeriaLIICrawler:
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def crawl_court(self, court: Court) -> list[CaseListEntry]:
        """
        Crawl all listing pages for a court and return case entries.

        Discovers available years from the court index page, then crawls
        each year's listing (with pagination) to collect all case URLs.
        """
        log = logger.bind(court=court.value)
        log.info("crawler.court_start")

        court_url = f"{BASE_URL}/judgments/{court.value}/"
        html = await self._fetch_cached(court_url, f"{court.value}/index.html")
        soup = BeautifulSoup(html, "lxml")

        year_links = self._extract_year_links(soup, court)
        log.info("crawler.years_found", count=len(year_links))

        all_entries: list[CaseListEntry] = []
        for year in sorted(year_links, reverse=True):
            entries = await self._crawl_year_listing(court, year, year_links[year])
            all_entries.extend(entries)
            log.info("crawler.year_done", year=year, cases=len(entries))

        log.info("crawler.court_done", total=len(all_entries))
        return all_entries

    async def crawl_judgment(self, entry: CaseListEntry) -> RawJudgment:
        """
        Fetch and parse an individual judgment page.

        Returns a :class:`RawJudgment` with all available metadata and the
        full judgment text + HTML.
        """
        assert entry.court is not None, "entry.court must be set"
        url = urljoin(BASE_URL, entry.case_url)

        # Derive a stable filesystem-safe cache key from the AKN URI
        cache_key = "judgments/" + entry.case_url.lstrip("/").replace("/", "_") + ".html"
        html = await self._fetch_cached(url, cache_key)
        soup = BeautifulSoup(html, "lxml")

        metadata = self._extract_metadata(soup)

        content_div = soup.select_one("#document-content, .document-content, article")
        full_text = ""
        full_html = ""
        if content_div:
            full_text = content_div.get_text(separator="\n", strip=False)
            full_html = str(content_div)

        return RawJudgment(
            case_name=entry.case_name,
            source_url=url,
            court=entry.court,
            media_neutral_citation=metadata.get("citation") or entry.citation,
            case_number=metadata.get("case_number") or entry.case_number,
            judges=metadata.get("judges", []),
            judgment_date=entry.judgment_date,
            language=metadata.get("language", "English"),
            labels=entry.labels,
            full_text=full_text,
            full_html=full_html,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_year_links(self, soup: BeautifulSoup, court: Court) -> dict[int, str]:
        """Extract year navigation links from the court index page."""
        years: dict[int, str] = {}
        for link in soup.select("a[href]"):
            href = str(link.get("href", ""))
            if f"/judgments/{court.value}/" not in href:
                continue
            parts = href.rstrip("/").split("/")
            try:
                year = int(parts[-1])
                years[year] = urljoin(BASE_URL, href)
            except (ValueError, IndexError):
                continue
        return years

    async def _crawl_year_listing(self, court: Court, year: int, url: str) -> list[CaseListEntry]:
        """Crawl a single year's listing page (handles pagination)."""
        cache_key = f"{court.value}/{year}/listing.html"
        html = await self._fetch_cached(url, cache_key)
        soup = BeautifulSoup(html, "lxml")

        entries: list[CaseListEntry] = []
        for row in soup.select("table tr"):
            entry = self._parse_listing_row(row, court)
            if entry:
                entries.append(entry)

        # Follow pagination links
        page = 2
        next_anchor = soup.select_one("a.next, a[rel='next']")
        while next_anchor:
            href = str(next_anchor.get("href", ""))
            if not href:
                break
            next_url = urljoin(BASE_URL, href)
            page_cache = f"{court.value}/{year}/listing_p{page}.html"
            html = await self._fetch_cached(next_url, page_cache)
            soup = BeautifulSoup(html, "lxml")
            for row in soup.select("table tr"):
                entry = self._parse_listing_row(row, court)
                if entry:
                    entries.append(entry)
            next_anchor = soup.select_one("a.next, a[rel='next']")
            page += 1

        return entries

    def _parse_listing_row(self, row: BeautifulSoup, court: Court) -> CaseListEntry | None:
        """Parse a single table row from a year listing page."""
        link = row.select_one("a[href*='/akn/']")
        if not link:
            return None

        case_name = link.get_text(strip=True)
        case_url = str(link.get("href", ""))
        if not case_url:
            return None

        # Date is typically in the last <td>
        cells = row.select("td")
        judgment_date: date | None = None
        if cells:
            date_text = cells[-1].get_text(strip=True)
            judgment_date = self._parse_date(date_text)

        # Labels/tags shown as badge elements
        labels = [
            b.get_text(strip=True).replace("CL|", "")
            for b in row.select("span.badge, .label")
            if b.get_text(strip=True)
        ]

        citation = self._extract_citation_from_name(case_name)
        case_number = self._extract_case_number(case_name)

        return CaseListEntry(
            case_name=case_name,
            case_url=case_url,
            judgment_date=judgment_date,
            citation=citation,
            case_number=case_number,
            labels=labels,
            court=court,
        )

    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, object]:
        """
        Extract structured metadata from a judgment page.

        NigeriaLII uses a definition list (<dl>) format::

            Citation: [2017] NGSC 5
            Court: Supreme Court of Nigeria
            Case number: SC. 373/2015
            Judges: Yahaya, JSC
            Judgment date: 22 June 2017
        """
        metadata: dict[str, object] = {}

        for dt in soup.select("dt"):
            key = dt.get_text(strip=True).lower().rstrip(":")
            dd = dt.find_next_sibling("dd")
            if not dd:
                continue
            # Remove copy-to-clipboard buttons before extracting text
            for btn in dd.select("button"):
                btn.decompose()
            value = dd.get_text(strip=True)  # type: ignore[union-attr]

            if "citation" in key:
                metadata["citation"] = value
            elif "case number" in key or "case no" in key:
                metadata["case_number"] = value
            elif "judge" in key:
                judge_links = dd.select("a")  # type: ignore[union-attr]
                if judge_links:
                    metadata["judges"] = [a.get_text(strip=True) for a in judge_links]
                else:
                    metadata["judges"] = [j.strip() for j in value.split(",") if j.strip()]
            elif "language" in key:
                metadata["language"] = value

        return metadata

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_cached(self, url: str, cache_key: str) -> str:
        """
        Fetch ``url`` with disk caching and rate limiting.

        On cache hit: returns cached content immediately (no network call).
        On cache miss: waits ``rate_limit_seconds``, fetches, caches, returns.
        """
        cache_path = self.raw_cache_dir / cache_key
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        async with self._semaphore:
            # Re-check after acquiring semaphore in case another coroutine
            # fetched the same URL concurrently
            if cache_path.exists():
                return cache_path.read_text(encoding="utf-8")

            await asyncio.sleep(self.rate_limit_seconds)

            client = self._client
            if client is None:
                raise RuntimeError("NigeriaLIICrawler must be used as an async context manager")

            logger.debug("crawler.fetch", url=url)
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

        cache_path.write_text(html, encoding="utf-8")
        return html

    # ── Date / citation parsing ───────────────────────────────────────────────

    @staticmethod
    def _parse_date(text: str) -> date | None:
        """Parse date strings like '22 June 2017', '22 Jun 2017', '2017-06-22'."""
        text = text.strip()
        for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%B %d, %Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _extract_citation_from_name(name: str) -> str | None:
        """Extract media neutral citation from case name, e.g. [2017] NGSC 5."""
        match = re.search(r"\[\d{4}\]\s+NG\w+\s+\d+", name)
        return match.group(0) if match else None

    @staticmethod
    def _extract_case_number(name: str) -> str | None:
        """Extract suit number from case name, e.g. SC. 373/2015."""
        match = re.search(r"\(([A-Z]{1,4}[./][^\)]{1,30})\)", name)
        return match.group(1).strip() if match else None
