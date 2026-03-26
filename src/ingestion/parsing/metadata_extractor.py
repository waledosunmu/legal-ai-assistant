"""Metadata extraction from raw NigeriaLII judgments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ingestion.sources.nigerialii import RawJudgment


@dataclass
class JudgmentMetadata:
    """Structured metadata extracted from a raw judgment."""

    case_name: str
    case_name_short: str         # e.g. "Malami v. Ohikhuare"
    applicant: str
    respondent: str
    court: str
    case_number: str | None
    media_neutral_citation: str | None
    judgment_date: date | None
    judges: list[str] = field(default_factory=list)
    lead_judge: str | None = None
    area_of_law: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


class MetadataExtractor:
    """
    Extract and normalise metadata from a :class:`~ingestion.sources.nigerialii.RawJudgment`.

    Combines metadata already present on the NigeriaLII page (case name,
    judges, citation) with metadata inferred from the judgment text
    (lead judge, area of law).
    """

    # Matches "Applicant v. Respondent" with optional trailing parentheticals
    _PARTY_PATTERN = re.compile(
        r"^(.+?)\s+v\.?\s+(.+?)(?:\s*\([^)]*\))*$",
        re.IGNORECASE | re.DOTALL,
    )

    # Titles and postnominals to skip when deriving short names
    _SKIP_WORDS: frozenset[str] = frozenset(
        {
            "ALH.", "ALHAJI", "CHIEF", "DR.", "DR", "ENGR.", "HON.", "MR.",
            "MRS.", "MS.", "AMBASSADOR", "SIR", "OFR", "CON", "GCON", "SAN",
            "KFR", "CFR", "OON", "MFR", "LTCOL", "GEN.", "COL.", "PROF.",
        }
    )

    # Keyword sets for each area-of-law label
    _AREA_KEYWORDS: dict[str, list[str]] = {
        "contract": [
            "contract", "breach of contract", "agreement", "offer",
            "acceptance", "consideration", "quantum meruit",
        ],
        "land_law": [
            "land", "property", "trespass", "certificate of occupancy",
            "right of occupancy", "revocation", "leasehold", "freehold",
            "possessory", "root of title",
        ],
        "criminal": [
            "murder", "robbery", "theft", "criminal", "conviction",
            "sentence", "the state v.", "federal republic of nigeria",
            "charge", "culpable homicide", "armed robbery",
        ],
        "company_law": [
            "company", "winding up", "shareholder", "director",
            "CAMA", "Companies and Allied Matters", "incorporation",
            "debenture", "memorandum of association",
        ],
        "constitutional": [
            "constitution", "fundamental rights", "human rights",
            "enforcement of fundamental rights", "section 36", "section 46",
            "constitutional", "CFRN",
        ],
        "election_petition": [
            "election", "petition", "electoral", "INEC",
            "tribunal", "governorship", "senatorial", "governorship election",
        ],
        "tort": [
            "negligence", "nuisance", "defamation", "libel",
            "personal injury", "damages", "nervous shock", "occupier",
        ],
        "family_law": [
            "marriage", "divorce", "custody", "matrimonial",
            "child", "inheritance", "succession", "intestate",
        ],
        "employment": [
            "employment", "labour", "worker", "termination",
            "industrial court", "wrongful dismissal", "reinstatement",
        ],
        "admiralty": [
            "admiralty", "shipping", "vessel", "maritime",
            "carriage of goods", "bill of lading",
        ],
        "banking_finance": [
            "bank", "loan", "mortgage", "debenture", "finance",
            "CBN", "central bank", "credit facility",
        ],
        "taxation": [
            "tax", "taxation", "FIRS", "customs", "duty", "VAT",
            "income tax", "capital gains", "stamp duty",
        ],
    }

    # Patterns for identifying the judge who delivered the lead judgment
    _LEAD_JUDGE_PATTERNS: list[str] = [
        r"[Dd]elivered\s+by\s+([A-Z][A-Z\s,.\-]+?(?:JSC|JCA|J\b))",
        r"\(Per\s+([A-Z][A-Z\s,.\-]+?(?:JSC|JCA|J\b))\)",
        r"[Rr]ead\s+by\s+([A-Z][A-Z\s,.\-]+?(?:JSC|JCA|J\b))",
        r"LEAD\s+JUDGMENT[:\s]+([A-Z][A-Z\s,.\-]+?(?:JSC|JCA|J\b))",
        r"[Ll]ead\s+[Jj]udgment\s+(?:was\s+)?delivered\s+by\s+"
        r"([A-Z][A-Z\s,.\-]+?(?:JSC|JCA|J\b))",
    ]

    def extract(self, raw: RawJudgment) -> JudgmentMetadata:
        """Extract structured metadata from a :class:`RawJudgment`."""
        parties = self._extract_parties(raw.case_name)
        short_name = self._make_short_name(parties)
        areas = self._infer_area_of_law(raw.case_name, raw.full_text)
        lead_judge = self._extract_lead_judge(raw.full_text)

        # Normalise judgment_date — might arrive as a string from JSONL load
        judgment_date = raw.judgment_date
        if isinstance(judgment_date, str):
            judgment_date = self._parse_date_str(judgment_date)

        return JudgmentMetadata(
            case_name=raw.case_name,
            case_name_short=short_name,
            applicant=parties[0] if parties else "",
            respondent=parties[1] if len(parties) > 1 else "",
            court=raw.court.value,
            case_number=raw.case_number,
            media_neutral_citation=raw.media_neutral_citation,
            judgment_date=judgment_date,
            judges=raw.judges,
            lead_judge=lead_judge,
            area_of_law=areas,
            labels=raw.labels,
        )

    # ── Party extraction ──────────────────────────────────────────────────────

    def _extract_parties(self, case_name: str) -> list[str]:
        """
        Extract [applicant, respondent] from a case name string.

        Strips trailing parentheticals (suit numbers, citations, dates)
        before applying the "X v. Y" pattern.
        """
        # Remove citation/date portion: "(SC. 373/2015) [2017] NGSC 5 ..."
        name_portion = re.sub(
            r"\s*\([A-Z/.\s\d]+\)\s*\[\d{4}\].*$", "", case_name
        ).strip()
        # Also remove trailing bare parentheticals with a date
        name_portion = re.sub(r"\s*\(\d{1,2}\s+\w+\s+\d{4}\)\s*$", "", name_portion).strip()

        match = self._PARTY_PATTERN.match(name_portion)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]
        return [name_portion]

    def _make_short_name(self, parties: list[str]) -> str:
        """Create a short citation name, e.g. 'Malami v. Ohikhuare'."""
        if len(parties) < 2:
            return parties[0][:50] if parties else ""
        return f"{self._last_name(parties[0])} v. {self._last_name(parties[1])}"

    def _last_name(self, full: str) -> str:
        """Extract the last significant word from a party name."""
        words = full.replace("&", "").split()
        significant = [
            w for w in words
            if w.upper().rstrip(".") not in self._SKIP_WORDS and len(w) > 2
        ]
        source = significant if significant else words
        return source[-1].title() if source else full[:20]

    # ── Area of law ───────────────────────────────────────────────────────────

    def _infer_area_of_law(self, case_name: str, full_text: str) -> list[str]:
        """Infer area(s) of law from case name and the first 5,000 chars of text."""
        combined = (case_name + " " + full_text[:5000]).lower()
        areas = [
            area
            for area, keywords in self._AREA_KEYWORDS.items()
            if any(kw.lower() in combined for kw in keywords)
        ]
        return areas or ["general"]

    # ── Lead judge ────────────────────────────────────────────────────────────

    def _extract_lead_judge(self, text: str) -> str | None:
        """
        Identify the judge who delivered the lead judgment.

        Searches the first 3,000 characters where the attribution typically
        appears.
        """
        search_region = text[:3000]
        for pattern in self._LEAD_JUDGE_PATTERNS:
            match = re.search(pattern, search_region)
            if match:
                return match.group(1).strip().rstrip(",")
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_date_str(text: str) -> date | None:
        """Parse a date string (e.g. '2017-06-22' or 'None') to a date."""
        if not text or text in ("None", "null", ""):
            return None
        for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
            try:
                from datetime import datetime
                return datetime.strptime(text.strip(), fmt).date()
            except ValueError:
                continue
        return None
