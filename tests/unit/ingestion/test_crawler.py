"""Unit tests for NigeriaLIICrawler and IngestionOrchestrator."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from ingestion.orchestrator import IngestionOrchestrator
from ingestion.sources.nigerialii import (
    CaseListEntry,
    Court,
    NigeriaLIICrawler,
    RawJudgment,
)

# ── HTML Fixtures ─────────────────────────────────────────────────────────────

COURT_INDEX_HTML = """
<html><body>
  <nav>
    <a href="/judgments/NGSC/2017/">2017</a>
    <a href="/judgments/NGSC/2016/">2016</a>
    <a href="/judgments/NGSC/2015/">2015</a>
    <a href="/judgments/NGCA/">Court of Appeal</a>
    <a href="/about/">About</a>
  </nav>
</body></html>
"""

YEAR_LISTING_HTML = """
<html><body>
  <table>
    <tr>
      <td>
        <a href="/akn/ng/judgment/ngsc/2017/5/eng@2017-06-22">
          Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5 (22 June 2017)
        </a>
      </td>
      <td>22 June 2017</td>
    </tr>
    <tr>
      <td>
        <a href="/akn/ng/judgment/ngsc/2017/4/eng@2017-05-10">
          Abubakar v. Yar'Adua (SC. 100/2007) [2017] NGSC 4 (10 May 2017)
        </a>
        <span class="badge">CL|Constitutional Law</span>
      </td>
      <td>10 May 2017</td>
    </tr>
    <tr><td>Header row with no link</td></tr>
  </table>
</body></html>
"""

YEAR_LISTING_PAGE2_HTML = """
<html><body>
  <table>
    <tr>
      <td>
        <a href="/akn/ng/judgment/ngsc/2017/3/eng@2017-04-01">
          Femi v. State (SC. 50/2015) [2017] NGSC 3 (1 April 2017)
        </a>
      </td>
      <td>1 April 2017</td>
    </tr>
  </table>
</body></html>
"""

YEAR_LISTING_WITH_PAGINATION_HTML = """
<html><body>
  <table>
    <tr>
      <td>
        <a href="/akn/ng/judgment/ngsc/2017/5/eng@2017-06-22">
          Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5 (22 June 2017)
        </a>
      </td>
      <td>22 June 2017</td>
    </tr>
  </table>
  <a class="next" href="/judgments/NGSC/2017/?page=2">Next</a>
</body></html>
"""

JUDGMENT_HTML = """
<html><body>
  <dl>
    <dt>Media Neutral Citation:</dt>
    <dd>[2017] NGSC 5</dd>
    <dt>Court:</dt>
    <dd>Supreme Court of Nigeria</dd>
    <dt>Case Number:</dt>
    <dd>SC. 373/2015</dd>
    <dt>Judges:</dt>
    <dd>
      <a href="/judge/yahaya">Yahaya, JSC</a>,
      <a href="/judge/onnoghen">Onnoghen, JSC</a>
    </dd>
    <dt>Language:</dt>
    <dd>English</dd>
  </dl>
  <article id="document-content">
    IN THE SUPREME COURT OF NIGERIA

    BETWEEN:
    ADELEKE                                          APPELLANT

    AND

    OBI                                              RESPONDENT

    The facts of this case are as follows...

    HELD: The appeal is dismissed.
  </article>
</body></html>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_crawler(tmp_path: Path) -> NigeriaLIICrawler:
    """Return a crawler with a tmp cache directory and no real HTTP client."""
    crawler = NigeriaLIICrawler(
        raw_cache_dir=tmp_path / "nigerialii",
        rate_limit_seconds=0.0,  # No delays in tests
    )
    return crawler


def make_entry(
    case_url: str = "/akn/ng/judgment/ngsc/2017/5/eng@2017-06-22",
    court: Court = Court.SUPREME_COURT,
) -> CaseListEntry:
    return CaseListEntry(
        case_name="Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5 (22 June 2017)",
        case_url=case_url,
        judgment_date=date(2017, 6, 22),
        citation="[2017] NGSC 5",
        case_number="SC. 373/2015",
        labels=[],
        court=court,
    )


# ── _extract_year_links ───────────────────────────────────────────────────────


def test_extract_year_links_finds_all_years(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)
    soup = BeautifulSoup(COURT_INDEX_HTML, "lxml")
    years = crawler._extract_year_links(soup, Court.SUPREME_COURT)

    assert set(years.keys()) == {2015, 2016, 2017}


def test_extract_year_links_ignores_other_courts(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)
    soup = BeautifulSoup(COURT_INDEX_HTML, "lxml")
    years = crawler._extract_year_links(soup, Court.SUPREME_COURT)

    # /judgments/NGCA/ has no year → should not appear
    assert all(isinstance(y, int) for y in years)


def test_extract_year_links_builds_full_urls(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)
    soup = BeautifulSoup(COURT_INDEX_HTML, "lxml")
    years = crawler._extract_year_links(soup, Court.SUPREME_COURT)

    assert years[2017] == "https://nigerialii.org/judgments/NGSC/2017/"


# ── _parse_listing_row ────────────────────────────────────────────────────────


def test_parse_listing_row_extracts_fields(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)
    soup = BeautifulSoup(YEAR_LISTING_HTML, "lxml")
    rows = soup.select("table tr")

    entry = crawler._parse_listing_row(rows[0], Court.SUPREME_COURT)
    assert entry is not None
    assert "Adeleke" in entry.case_name
    assert entry.case_url == "/akn/ng/judgment/ngsc/2017/5/eng@2017-06-22"
    assert entry.judgment_date == date(2017, 6, 22)
    assert entry.citation == "[2017] NGSC 5"
    assert entry.case_number == "SC. 373/2015"
    assert entry.court == Court.SUPREME_COURT


def test_parse_listing_row_extracts_labels(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)
    soup = BeautifulSoup(YEAR_LISTING_HTML, "lxml")
    rows = soup.select("table tr")

    # Second row has a badge
    entry = crawler._parse_listing_row(rows[1], Court.SUPREME_COURT)
    assert entry is not None
    assert "Constitutional Law" in entry.labels


def test_parse_listing_row_returns_none_for_no_akn_link(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)
    soup = BeautifulSoup(YEAR_LISTING_HTML, "lxml")
    rows = soup.select("table tr")

    # Third row has no /akn/ link
    entry = crawler._parse_listing_row(rows[2], Court.SUPREME_COURT)
    assert entry is None


# ── _parse_date ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("22 June 2017", date(2017, 6, 22)),
        ("22 Jun 2017", date(2017, 6, 22)),
        ("2017-06-22", date(2017, 6, 22)),
        ("June 22, 2017", date(2017, 6, 22)),
        ("not a date", None),
        ("", None),
    ],
)
def test_parse_date(text: str, expected: date | None, tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)
    assert crawler._parse_date(text) == expected


# ── _extract_citation_from_name / _extract_case_number ───────────────────────


@pytest.mark.parametrize(
    "name, expected_citation",
    [
        (
            "Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5 (22 June 2017)",
            "[2017] NGSC 5",
        ),
        (
            "Abubakar v. Yar'Adua (CA/L/100/12) [2012] NGCA 3",
            "[2012] NGCA 3",
        ),
        ("No citation here", None),
    ],
)
def test_extract_citation(name: str, expected_citation: str | None) -> None:
    result = NigeriaLIICrawler._extract_citation_from_name(name)
    assert result == expected_citation


@pytest.mark.parametrize(
    "name, expected_number",
    [
        (
            "Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5",
            "SC. 373/2015",
        ),
        (
            "Abubakar v. State (CA/L/692/12) [2012] NGCA 3",
            "CA/L/692/12",
        ),
        ("No case number", None),
    ],
)
def test_extract_case_number(name: str, expected_number: str | None) -> None:
    result = NigeriaLIICrawler._extract_case_number(name)
    assert result == expected_number


# ── _extract_metadata ─────────────────────────────────────────────────────────


def test_extract_metadata_parses_dl(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)
    soup = BeautifulSoup(JUDGMENT_HTML, "lxml")
    metadata = crawler._extract_metadata(soup)

    assert metadata["citation"] == "[2017] NGSC 5"
    assert metadata["case_number"] == "SC. 373/2015"
    assert metadata["judges"] == ["Yahaya, JSC", "Onnoghen, JSC"]
    assert metadata["language"] == "English"


# ── _fetch_cached ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_cached_writes_to_disk(tmp_path: Path) -> None:
    """First fetch writes cache; second fetch reads from disk (no HTTP call)."""
    crawler = make_crawler(tmp_path)
    crawler._client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "<html>hello</html>"
    mock_response.raise_for_status = MagicMock()
    crawler._client.get = AsyncMock(return_value=mock_response)

    result = await crawler._fetch_cached("https://nigerialii.org/test", "test/page.html")
    assert result == "<html>hello</html>"

    # Cache file should now exist
    cache_path = tmp_path / "nigerialii" / "test" / "page.html"
    assert cache_path.exists()
    assert cache_path.read_text() == "<html>hello</html>"

    # Second call must NOT make another HTTP request
    crawler._client.get.reset_mock()
    result2 = await crawler._fetch_cached("https://nigerialii.org/test", "test/page.html")
    assert result2 == "<html>hello</html>"
    crawler._client.get.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_cached_raises_without_client(tmp_path: Path) -> None:
    """Calling _fetch_cached without entering context manager raises RuntimeError."""
    crawler = make_crawler(tmp_path)  # _client is None
    with pytest.raises(RuntimeError, match="async context manager"):
        await crawler._fetch_cached("https://nigerialii.org/x", "x.html")


# ── crawl_judgment ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crawl_judgment_extracts_text(tmp_path: Path) -> None:
    entry = make_entry()
    crawler = make_crawler(tmp_path)

    with patch.object(
        crawler,
        "_fetch_cached",
        new=AsyncMock(return_value=JUDGMENT_HTML),
    ):
        async with crawler:
            judgment = await crawler.crawl_judgment(entry)

    assert isinstance(judgment, RawJudgment)
    assert judgment.media_neutral_citation == "[2017] NGSC 5"
    assert judgment.case_number == "SC. 373/2015"
    assert "Yahaya, JSC" in judgment.judges
    assert "Onnoghen, JSC" in judgment.judges
    assert "SUPREME COURT" in judgment.full_text
    assert judgment.court == Court.SUPREME_COURT


@pytest.mark.asyncio
async def test_crawl_judgment_falls_back_to_entry_citation(
    tmp_path: Path,
) -> None:
    """When judgment page has no citation <dt>, the entry citation is used."""
    no_citation_html = "<html><body><article>Text only.</article></body></html>"
    entry = make_entry()
    crawler = make_crawler(tmp_path)

    with patch.object(
        crawler,
        "_fetch_cached",
        new=AsyncMock(return_value=no_citation_html),
    ):
        async with crawler:
            judgment = await crawler.crawl_judgment(entry)

    assert judgment.media_neutral_citation == entry.citation


# ── crawl_court (integration with mocked HTTP) ───────────────────────────────


@pytest.mark.asyncio
async def test_crawl_court_returns_all_entries(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)

    async def fake_fetch(url: str, cache_key: str) -> str:
        if "index" in cache_key:
            return COURT_INDEX_HTML
        return YEAR_LISTING_HTML  # 2 valid entries per year

    with patch.object(crawler, "_fetch_cached", new=AsyncMock(side_effect=fake_fetch)):
        async with crawler:
            entries = await crawler.crawl_court(Court.SUPREME_COURT)

    # 3 years × 2 valid rows = 6
    assert len(entries) == 6
    assert all(e.court == Court.SUPREME_COURT for e in entries)


@pytest.mark.asyncio
async def test_crawl_court_handles_pagination(tmp_path: Path) -> None:
    crawler = make_crawler(tmp_path)

    pages: dict[str, str] = {}
    for year in (2015, 2016, 2017):
        pages[f"NGSC/{year}/listing.html"] = YEAR_LISTING_WITH_PAGINATION_HTML
        pages[f"NGSC/{year}/listing_p2.html"] = YEAR_LISTING_PAGE2_HTML

    async def fake_fetch(url: str, cache_key: str) -> str:
        if "index" in cache_key:
            return COURT_INDEX_HTML
        return pages.get(cache_key, "<html><body><table></table></body></html>")

    with patch.object(crawler, "_fetch_cached", new=AsyncMock(side_effect=fake_fetch)):
        async with crawler:
            entries = await crawler.crawl_court(Court.SUPREME_COURT)

    # 3 years × (1 entry page1 + 1 entry page2) = 6
    assert len(entries) == 6


# ── IngestionOrchestrator ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_discovery_populates_manifest(
    tmp_path: Path,
) -> None:
    orchestrator = IngestionOrchestrator(data_dir=tmp_path)

    async def fake_crawl_court(court: Court) -> list[CaseListEntry]:
        return [make_entry(court=court), make_entry(court=court)]

    with patch.object(
        orchestrator.crawler,
        "crawl_court",
        new=AsyncMock(side_effect=fake_crawl_court),
    ):
        # Need context manager to be a no-op
        orchestrator.crawler.__aenter__ = AsyncMock(return_value=orchestrator.crawler)
        orchestrator.crawler.__aexit__ = AsyncMock(return_value=None)

        await orchestrator.run_discovery(courts=[Court.SUPREME_COURT])

    assert "NGSC" in orchestrator.manifest["discovered"]
    assert len(orchestrator.manifest["discovered"]["NGSC"]) == 2

    # Manifest file should be written
    assert (tmp_path / "manifest.json").exists()


@pytest.mark.asyncio
async def test_orchestrator_discovery_skips_already_discovered(
    tmp_path: Path,
) -> None:
    """Courts already in the manifest are not re-crawled."""
    orchestrator = IngestionOrchestrator(data_dir=tmp_path)
    orchestrator.manifest["discovered"]["NGSC"] = [{"case_url": "/existing"}]

    crawl_court_mock = AsyncMock()
    with patch.object(orchestrator.crawler, "crawl_court", crawl_court_mock):
        orchestrator.crawler.__aenter__ = AsyncMock(return_value=orchestrator.crawler)
        orchestrator.crawler.__aexit__ = AsyncMock(return_value=None)

        await orchestrator.run_discovery(courts=[Court.SUPREME_COURT])

    crawl_court_mock.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_fetch_saves_jsonl(tmp_path: Path) -> None:
    orchestrator = IngestionOrchestrator(data_dir=tmp_path)

    # Pre-populate manifest with one discovered entry
    entry = make_entry()
    orchestrator.manifest["discovered"]["NGSC"] = [
        {
            "case_name": entry.case_name,
            "case_url": entry.case_url,
            "judgment_date": str(entry.judgment_date),
            "citation": entry.citation,
            "case_number": entry.case_number,
            "labels": entry.labels,
        }
    ]

    dummy_judgment = RawJudgment(
        case_name=entry.case_name,
        source_url="https://nigerialii.org" + entry.case_url,
        court=Court.SUPREME_COURT,
        media_neutral_citation="[2017] NGSC 5",
        full_text="Judgment text here.",
        full_html="<p>Judgment text here.</p>",
    )

    with patch.object(
        orchestrator.crawler,
        "crawl_judgment",
        new=AsyncMock(return_value=dummy_judgment),
    ):
        orchestrator.crawler.__aenter__ = AsyncMock(return_value=orchestrator.crawler)
        orchestrator.crawler.__aexit__ = AsyncMock(return_value=None)

        await orchestrator.run_fetch(courts=[Court.SUPREME_COURT])

    # JSONL file should exist with one line
    jsonl_path = tmp_path / "raw" / "judgments" / "NGSC.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["citation"] == "[2017] NGSC 5"
    assert record["full_text"] == "Judgment text here."


@pytest.mark.asyncio
async def test_orchestrator_fetch_skips_already_fetched(tmp_path: Path) -> None:
    """URLs already in manifest['fetched'] are not fetched again."""
    orchestrator = IngestionOrchestrator(data_dir=tmp_path)

    entry_url = "/akn/ng/judgment/ngsc/2017/5/eng@2017-06-22"
    orchestrator.manifest["discovered"]["NGSC"] = [
        {
            "case_name": "Test case",
            "case_url": entry_url,
            "judgment_date": "2017-06-22",
            "citation": None,
            "case_number": None,
            "labels": [],
        }
    ]
    orchestrator.manifest["fetched"] = [entry_url]  # Already fetched

    crawl_judgment_mock = AsyncMock()
    with patch.object(orchestrator.crawler, "crawl_judgment", crawl_judgment_mock):
        orchestrator.crawler.__aenter__ = AsyncMock(return_value=orchestrator.crawler)
        orchestrator.crawler.__aexit__ = AsyncMock(return_value=None)

        await orchestrator.run_fetch(courts=[Court.SUPREME_COURT])

    crawl_judgment_mock.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_fetch_respects_limit(tmp_path: Path) -> None:
    orchestrator = IngestionOrchestrator(data_dir=tmp_path)

    entries = [
        {
            "case_name": f"Case {i}",
            "case_url": f"/akn/ngsc/{i}",
            "judgment_date": "2017-01-01",
            "citation": None,
            "case_number": None,
            "labels": [],
        }
        for i in range(5)
    ]
    orchestrator.manifest["discovered"]["NGSC"] = entries

    dummy_judgment = RawJudgment(
        case_name="Case",
        source_url="https://nigerialii.org/test",
        court=Court.SUPREME_COURT,
    )

    with patch.object(
        orchestrator.crawler,
        "crawl_judgment",
        new=AsyncMock(return_value=dummy_judgment),
    ):
        orchestrator.crawler.__aenter__ = AsyncMock(return_value=orchestrator.crawler)
        orchestrator.crawler.__aexit__ = AsyncMock(return_value=None)

        await orchestrator.run_fetch(courts=[Court.SUPREME_COURT], limit=2)

    assert len(orchestrator.manifest["fetched"]) == 2


# ── Manifest persistence ──────────────────────────────────────────────────────


def test_manifest_roundtrip(tmp_path: Path) -> None:
    """Manifest survives save → load cycle."""
    orchestrator = IngestionOrchestrator(data_dir=tmp_path)
    orchestrator.manifest["fetched"] = ["/some/url"]
    orchestrator._save_manifest()

    orchestrator2 = IngestionOrchestrator(data_dir=tmp_path)
    assert orchestrator2.manifest["fetched"] == ["/some/url"]


def test_orchestrator_summary(tmp_path: Path) -> None:
    orchestrator = IngestionOrchestrator(data_dir=tmp_path)
    orchestrator.manifest["discovered"]["NGSC"] = [{}, {}]
    orchestrator.manifest["fetched"] = ["a", "b", "c"]
    orchestrator.manifest["failed"] = ["x"]

    summary = orchestrator.summary()
    assert summary["discovered"]["NGSC"] == 2
    assert summary["fetched"] == 3
    assert summary["failed"] == 1
