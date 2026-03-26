"""Laws.Africa Content API client — AKN XML / HTML → statute_segments."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.laws.africa/v3"

# Priority statutes for the MVP (5 motion types)
PRIORITY_LEGISLATION: list[str] = [
    "/akn/ng/act/1999/constitution",   # CFRN 1999
    "/akn/ng/act/2020/3",              # CAMA 2020
    "/akn/ng/act/2011/18",             # Evidence Act 2011
    "/akn/ng/act/2004/sh6",            # Sheriffs & Civil Process Act
    "/akn/ng/act/2023/arb",            # Arbitration & Mediation Act 2023
    "/akn/ng/act/1990/fhc",            # Federal High Court Act
    "/akn/ng/act/2004/ca",             # Court of Appeal Act
    "/akn/ng/act/2004/sc",             # Supreme Court Act
]


@dataclass
class LegislationSection:
    """A single section of a statute, ready for embedding."""

    title: str              # Full statute title
    short_title: str | None
    section: str            # Section heading, e.g. "Section 36"
    content: str            # Full text of the section
    year: int | None
    frbr_uri: str
    jurisdiction: str = "NG"
    source: str = "laws_africa"


@dataclass
class LegislationRecord:
    """A complete piece of legislation from Laws.Africa."""

    frbr_uri: str
    title: str
    short_title: str | None
    year: int | None
    nature: str             # act, regulation, etc.
    publication_date: str | None
    repeal_status: str | None
    sections: list[LegislationSection] = field(default_factory=list)
    full_text: str = ""
    akn_xml: str | None = None


class LawsAfricaClient:
    """
    Client for the Laws.Africa Content API.

    Free for non-commercial use (CC-BY-NC-SA).

    Usage::

        async with LawsAfricaClient(api_token="...") as client:
            works = await client.list_works()
            record = await client.fetch_work(works[0]["frbr_uri"])
    """

    def __init__(self, api_token: str) -> None:
        self._headers = {
            "Authorization": f"Token {api_token}",
            "Accept": "application/json",
        }

    async def list_works(
        self,
        country: str = "ng",
        nature: str = "act",
    ) -> list[dict]:
        """
        List all works (legislation) for a country.

        Paginates automatically through all pages.
        """
        works: list[dict] = []
        url = f"{_BASE_URL}/akn/{country}/.json"
        params: dict = {"nature": nature}

        async with httpx.AsyncClient(headers=self._headers, timeout=30.0) as client:
            while url:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                works.extend(data.get("results", []))
                url = data.get("next")   # next page URL or None
                params = {}              # params only for first request

        logger.info("laws_africa.listed_works", country=country, count=len(works))
        return works

    async def get_work_text(self, frbr_uri: str) -> str:
        """Fetch the full HTML text of a work."""
        url = f"{_BASE_URL}{frbr_uri}.html"
        async with httpx.AsyncClient(headers=self._headers, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    async def get_work_xml(self, frbr_uri: str) -> str:
        """Fetch the Akoma Ntoso XML of a work."""
        url = f"{_BASE_URL}{frbr_uri}.xml"
        async with httpx.AsyncClient(headers=self._headers, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    async def get_work_toc(self, frbr_uri: str) -> list[dict]:
        """Fetch the table of contents for a work."""
        url = f"{_BASE_URL}{frbr_uri}/toc.json"
        async with httpx.AsyncClient(headers=self._headers, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json().get("toc", [])

    async def fetch_work(self, frbr_uri: str) -> LegislationRecord:
        """
        Fetch a complete work and parse it into a ``LegislationRecord``.

        Combines the HTML endpoint for text extraction with the TOC for
        section headings.
        """
        html = await self.get_work_text(frbr_uri)
        sections = self.parse_html_to_sections(html, frbr_uri)

        # Extract basic metadata from the HTML
        soup = BeautifulSoup(html, "lxml")
        title = (
            soup.select_one("h1, .akn-FRBRalias, title")
            or soup.select_one("title")
        )
        title_text = title.get_text(strip=True) if title else frbr_uri

        year = self._extract_year_from_uri(frbr_uri)

        record = LegislationRecord(
            frbr_uri=frbr_uri,
            title=title_text,
            short_title=None,
            year=year,
            nature="act",
            publication_date=None,
            repeal_status=None,
            sections=sections,
            full_text=soup.get_text(separator="\n", strip=True),
        )
        logger.info(
            "laws_africa.fetched_work",
            frbr_uri=frbr_uri,
            sections=len(sections),
        )
        return record

    @staticmethod
    def parse_html_to_sections(
        html: str,
        title: str,
    ) -> list[LegislationSection]:
        """
        Parse Laws.Africa HTML into a flat list of ``LegislationSection`` objects.

        Laws.Africa HTML uses ``<section>`` / ``<div class="akn-section">``
        elements with heading tags for each statutory provision.
        Falls back to splitting on ``<h2>`` / ``<h3>`` headings.
        """
        soup = BeautifulSoup(html, "lxml")

        sections: list[LegislationSection] = []

        # Try AKN-style section elements first
        akn_sections = soup.select(
            ".akn-section, .akn-article, section[class*='akn']"
        )
        if akn_sections:
            for el in akn_sections:
                heading = el.select_one("h1, h2, h3, h4, .akn-num, .akn-heading")
                section_title = heading.get_text(strip=True) if heading else ""
                # Remove heading from content
                if heading:
                    heading.decompose()
                content = el.get_text(separator=" ", strip=True)
                if content:
                    sections.append(
                        LegislationSection(
                            title=title,
                            short_title=None,
                            section=section_title,
                            content=content,
                            year=None,
                            frbr_uri=title,
                        )
                    )
            return sections

        # Fallback: split on h2/h3 headings
        current_heading = "Preamble"
        current_paragraphs: list[str] = []

        for el in soup.find_all(["h2", "h3", "p"]):
            if el.name in ("h2", "h3"):
                if current_paragraphs:
                    sections.append(
                        LegislationSection(
                            title=title,
                            short_title=None,
                            section=current_heading,
                            content=" ".join(current_paragraphs),
                            year=None,
                            frbr_uri=title,
                        )
                    )
                current_heading = el.get_text(strip=True)
                current_paragraphs = []
            else:
                text = el.get_text(strip=True)
                if text:
                    current_paragraphs.append(text)

        if current_paragraphs:
            sections.append(
                LegislationSection(
                    title=title,
                    short_title=None,
                    section=current_heading,
                    content=" ".join(current_paragraphs),
                    year=None,
                    frbr_uri=title,
                )
            )

        return sections

    @staticmethod
    def _extract_year_from_uri(frbr_uri: str) -> int | None:
        """Extract year from an AKN FRBR URI like /akn/ng/act/2020/3."""
        import re
        match = re.search(r"/(\d{4})/", frbr_uri)
        return int(match.group(1)) if match else None
