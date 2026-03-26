"""Nigerian legal citation extraction and treatment classification."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExtractedCitation:
    """A single legal citation extracted from judgment text."""

    raw_text: str  # Matched text as it appears in the judgment
    case_name: str | None  # e.g. "Adesanya v. President of Nigeria"
    year: int | None
    report_series: str | None  # e.g. "NWLR", "LPELR", "SC"
    volume: str | None
    part: str | None  # e.g. "1748" from "Pt. 1748"
    page: str | None
    full_citation: str  # Normalised display string
    position: int  # Character offset in source text
    context: str  # Surrounding sentence (for treatment inference)


class NigerianCitationExtractor:
    """
    Extract legal citations from Nigerian court judgment text.

    Handles all major Nigerian law report formats:

    ======= ===========================================
    Format  Example
    ======= ===========================================
    NWLR    (2020) 15 NWLR (Pt. 1748) 1
    LPELR   (2022) LPELR-57809(SC)
    SC      (1986) 2 SC 87
    NSCC    (1981) NSCC 146
    All NLR [1966] 1 All NLR 186
    AFWLR   (2015) AFWLR (Pt. 789) 15
    FWLR    (2015) FWLR (Pt. 789) 15
    Neutral [2017] NGSC 5
    ======= ===========================================
    """

    # Ordered by specificity (most specific first to reduce false positives)
    _PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        (
            "NWLR",
            re.compile(
                r"\((\d{4})\)\s+(\d+)\s+NWLR\s+\(Pt\.\s*(\d+)\)\s+(\d+)",
                re.IGNORECASE,
            ),
        ),
        (
            "LPELR",
            re.compile(
                r"\((\d{4})\)\s+LPELR[\s\-]*(\d+)\s*\((\w+)\)",
                re.IGNORECASE,
            ),
        ),
        (
            "SC",
            re.compile(
                r"\((\d{4})\)\s+(\d+)(?:\s*-\s*\d+)?\s+SC\s+(\d+)",
                re.IGNORECASE,
            ),
        ),
        (
            "NSCC",
            re.compile(
                r"\((\d{4})\)\s+(?:(\d+)\s+)?NSCC\s+(\d+)",
                re.IGNORECASE,
            ),
        ),
        (
            "All NLR",
            re.compile(
                r"\[(\d{4})\]\s+(\d+)\s+All\s+NLR\s+(\d+)",
                re.IGNORECASE,
            ),
        ),
        (
            "AFWLR",
            re.compile(
                # "AFWLR" = All Federation WLR (5 chars); "FWLR" = Federation WLR (4 chars)
                r"\((\d{4})\)\s+A?FWLR\s+\(Pt\.\s*(\d+)\)\s+(\d+)",
                re.IGNORECASE,
            ),
        ),
        (
            "NeutralNG",
            re.compile(
                r"\[(\d{4})\]\s+(NG\w+)\s+(\d+)",
                re.IGNORECASE,
            ),
        ),
    ]

    # Backwards-look pattern to find "Applicant v. Respondent" before a citation
    _CASE_NAME_PATTERN = re.compile(
        r"([A-Z][A-Za-z\s&.'()\-]+?)\s+v\.?\s+" r"([A-Z][A-Za-z\s&.'()\-]+?)\s*\(",
    )

    def extract_all(self, text: str) -> list[ExtractedCitation]:
        """
        Extract all citations from ``text``, deduplicated and position-sorted.
        """
        seen_positions: set[int] = set()
        citations: list[ExtractedCitation] = []

        for series, pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                pos = match.start()
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                citation = self._build_citation(match, series, text)
                if citation:
                    citations.append(citation)

        return sorted(citations, key=lambda c: c.position)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_citation(
        self,
        match: re.Match[str],
        series: str,
        full_text: str,
    ) -> ExtractedCitation | None:
        raw_text = match.group(0)
        position = match.start()

        year = self._extract_year(match)
        case_name = self._find_case_name(full_text, position)
        context = self._extract_context(full_text, position)
        volume, part, page = self._extract_volume_part_page(match, series)

        display = f"{case_name} {raw_text}" if case_name else raw_text

        return ExtractedCitation(
            raw_text=raw_text,
            case_name=case_name,
            year=year,
            report_series=series,
            volume=volume,
            part=part,
            page=page,
            full_citation=display,
            position=position,
            context=context,
        )

    @staticmethod
    def _extract_year(match: re.Match[str]) -> int | None:
        """Return the first 4-digit group that looks like a year."""
        for group in match.groups():
            if group and group.isdigit() and len(group) == 4:
                return int(group)
        return None

    @staticmethod
    def _extract_volume_part_page(
        match: re.Match[str], series: str
    ) -> tuple[str | None, str | None, str | None]:
        """
        Extract (volume, part, page) using explicit group indices per series.

        Using explicit indices avoids the year-filter approach which accidentally
        drops 4-digit part numbers like "1748" in NWLR citations.
        """
        g = match.groups()
        if series == "NWLR":
            # groups: (year, volume, part, page)
            return g[1], g[2], g[3]
        if series == "LPELR":
            # groups: (year, number, court) — number used as volume
            return g[1], None, None
        if series == "SC":
            # groups: (year, volume, page)
            return g[1], None, g[2]
        if series == "NSCC":
            # groups: (year, optional_volume, page)
            return g[1], None, g[2]
        if series == "All NLR":
            # groups: (year, volume, page)
            return g[1], None, g[2]
        if series == "AFWLR":
            # groups: (year, part, page)
            return None, g[1], g[2]
        if series == "NeutralNG":
            # groups: (year, court_code, number)
            return None, None, g[2]
        return None, None, None

    def _find_case_name(self, text: str, cite_pos: int) -> str | None:
        """
        Look backwards up to 200 characters for a 'Party v. Party' pattern.

        The search region is extended by 1 past cite_pos so the opening
        parenthesis of the citation is included, which the pattern requires.
        """
        search_region = text[max(0, cite_pos - 200) : cite_pos + 1]
        match = self._CASE_NAME_PATTERN.search(search_region)
        if match:
            return f"{match.group(1).strip()} v. {match.group(2).strip()}"
        return None

    @staticmethod
    def _extract_context(text: str, position: int) -> str:
        """
        Extract the sentence containing the citation.

        Sentence boundaries are detected by:
        - ``". "`` after a word whose last 3 chars are lowercase (avoids
          abbreviations like ``v.``, ``Pt.``, ``JSC.``)
        - Any newline (paragraph boundary)
        """
        # Lookbehind width is fixed (3) so Python's re module accepts it.
        # The pattern avoids abbreviation periods: "v." has only 1 lowercase
        # char before ".", "Pt." has only 1, "JSC." has 0 — all fail the
        # 3-char lookbehind. Real sentence endings like "decision." pass.
        boundary = re.compile(r"(?<=[a-z]{3})\.\s+(?=[A-Z])|\n")

        # Sentence start: end of the last boundary before position
        start = 0
        for m in boundary.finditer(text, 0, position):
            start = m.end()

        # Sentence end: start of the next boundary at or after position
        end_match = boundary.search(text, position)
        end = end_match.start() if end_match else len(text)

        return text[start:end].strip()


class CitationTreatmentClassifier:
    """
    Classify how a cited case was treated in a judgment, based on the
    surrounding sentence context.

    Returns one of: ``"followed"``, ``"distinguished"``, ``"overruled"``,
    ``"mentioned"``.
    """

    _TREATMENT_KEYWORDS: dict[str, list[str]] = {
        "followed": [
            "followed",
            "approved",
            "adopted",
            "applied",
            "in line with",
            "consistent with",
            "relying on",
            "as held in",
            "as decided in",
            "in accord with",
            "on all fours",
            "same principle",
            "endorse",
        ],
        "distinguished": [
            "distinguished",
            "distinguishable",
            "different from",
            "unlike the case of",
            "does not apply",
            "inapplicable",
            "not on all fours",
            "no parity",
            "not applicable",
            "distinguishing",
        ],
        "overruled": [
            "overruled",
            "overturned",
            "reversed",
            "no longer good law",
            "departed from",
            "declined to follow",
            "per incuriam",
            "bad law",
        ],
    }

    def classify(self, context: str) -> str:
        """
        Return the treatment label for ``context``.

        Checks from strongest signal (overruled) to weakest (followed),
        returning ``"mentioned"`` as the default when no signal is found.
        """
        context_lower = context.lower()
        for treatment in ("overruled", "distinguished", "followed"):
            if any(kw in context_lower for kw in self._TREATMENT_KEYWORDS[treatment]):
                return treatment
        return "mentioned"
