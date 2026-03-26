"""Unit tests for NigerianCitationExtractor and CitationTreatmentClassifier."""

from __future__ import annotations

import pytest

from ingestion.citations.extractor import (
    CitationTreatmentClassifier,
    ExtractedCitation,
    NigerianCitationExtractor,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

EXTRACTOR = NigerianCitationExtractor()
CLASSIFIER = CitationTreatmentClassifier()


# ── Individual citation format tests ─────────────────────────────────────────


class TestNWLR:
    def test_basic_nwlr(self) -> None:
        text = "as decided in (2020) 15 NWLR (Pt. 1748) 1"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        c = citations[0]
        assert c.report_series == "NWLR"
        assert c.year == 2020
        assert c.volume == "15"
        assert c.part == "1748"
        assert c.page == "1"

    def test_nwlr_case_insensitive(self) -> None:
        text = "(2018) 7 nwlr (pt. 1618) 20"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        assert citations[0].report_series == "NWLR"

    def test_nwlr_extracts_case_name(self) -> None:
        text = (
            "The decision in Adeleke v. Obi "
            "(2020) 15 NWLR (Pt. 1748) 1 was followed."
        )
        citations = EXTRACTOR.extract_all(text)
        assert citations[0].case_name is not None
        assert "Adeleke" in citations[0].case_name
        assert "Obi" in citations[0].case_name


class TestLPELR:
    def test_basic_lpelr(self) -> None:
        text = "see (2022) LPELR-57809(SC)"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        c = citations[0]
        assert c.report_series == "LPELR"
        assert c.year == 2022
        assert c.volume == "57809"

    def test_lpelr_with_space(self) -> None:
        text = "(2019) LPELR 43701(CA)"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        assert citations[0].report_series == "LPELR"

    def test_lpelr_with_hyphen(self) -> None:
        text = "(2015) LPELR-24681(SC)"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1


class TestSC:
    def test_basic_sc(self) -> None:
        text = "as held in (1986) 2 SC 87"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        c = citations[0]
        assert c.report_series == "SC"
        assert c.year == 1986
        assert c.volume == "2"
        assert c.page == "87"

    def test_sc_with_range_volume(self) -> None:
        text = "(1975) 1-2 SC 37"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        assert citations[0].report_series == "SC"


class TestNSCC:
    def test_basic_nscc(self) -> None:
        text = "decided in (1981) NSCC 146"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        c = citations[0]
        assert c.report_series == "NSCC"
        assert c.year == 1981

    def test_nscc_with_volume(self) -> None:
        text = "(1985) 16 NSCC 199"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1


class TestAllNLR:
    def test_basic_all_nlr(self) -> None:
        text = "[1966] 1 All NLR 186"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        c = citations[0]
        assert c.report_series == "All NLR"
        assert c.year == 1966
        assert c.volume == "1"
        assert c.page == "186"


class TestAFWLR:
    def test_afwlr(self) -> None:
        text = "(2015) AFWLR (Pt. 789) 15"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        c = citations[0]
        assert c.report_series == "AFWLR"
        assert c.year == 2015

    def test_fwlr(self) -> None:
        text = "(2012) FWLR (Pt. 650) 42"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        assert citations[0].report_series == "AFWLR"


class TestNeutralCitation:
    def test_ngsc_neutral(self) -> None:
        text = "[2017] NGSC 5"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        c = citations[0]
        assert c.report_series == "NeutralNG"
        assert c.year == 2017

    def test_ngca_neutral(self) -> None:
        text = "[2019] NGCA 12"
        citations = EXTRACTOR.extract_all(text)
        assert len(citations) == 1
        assert citations[0].year == 2019


# ── extract_all edge cases ────────────────────────────────────────────────────


def test_extract_all_multiple_citations() -> None:
    text = (
        "The court followed (2020) 15 NWLR (Pt. 1748) 1 and also "
        "referred to (1986) 2 SC 87."
    )
    citations = EXTRACTOR.extract_all(text)
    assert len(citations) == 2


def test_extract_all_sorted_by_position() -> None:
    text = "(1986) 2 SC 87 … (2020) 15 NWLR (Pt. 1748) 1"
    citations = EXTRACTOR.extract_all(text)
    assert citations[0].position < citations[1].position


def test_extract_all_deduplicates_same_position() -> None:
    # A single citation string that could match multiple patterns should
    # only produce one result at the same character offset.
    text = "(2020) 15 NWLR (Pt. 1748) 1"
    citations = EXTRACTOR.extract_all(text)
    assert len(citations) == 1


def test_extract_all_empty_text() -> None:
    assert EXTRACTOR.extract_all("") == []


def test_extract_all_no_citations() -> None:
    text = "The court considered the submissions of learned counsel."
    assert EXTRACTOR.extract_all(text) == []


def test_context_extraction() -> None:
    text = (
        "The appellant relied on the following authority. "
        "In (2020) 15 NWLR (Pt. 1748) 1 the court held that. "
        "This was distinguished on the facts."
    )
    citations = EXTRACTOR.extract_all(text)
    assert len(citations) == 1
    assert len(citations[0].context) > 0


def test_full_citation_includes_case_name_when_found() -> None:
    text = "Adeleke v. Obi (2020) 15 NWLR (Pt. 1748) 1"
    citations = EXTRACTOR.extract_all(text)
    assert "Adeleke" in citations[0].full_citation


def test_full_citation_fallback_to_raw_text_when_no_case_name() -> None:
    text = "As previously held, see (1986) 2 SC 87."
    citations = EXTRACTOR.extract_all(text)
    assert "(1986) 2 SC 87" in citations[0].full_citation


# ── CitationTreatmentClassifier ───────────────────────────────────────────────


class TestCitationTreatmentClassifier:
    @pytest.mark.parametrize(
        "context, expected",
        [
            # followed signals
            (
                "This court followed the decision in Adeleke v. Obi",
                "followed",
            ),
            (
                "The principle in that case was adopted and applied.",
                "followed",
            ),
            (
                "relying on the authority cited by counsel",
                "followed",
            ),
            (
                "as held in the earlier decision, the same principle applies",
                "followed",
            ),
            # distinguished signals
            (
                "The case is distinguishable on its facts.",
                "distinguished",
            ),
            (
                "Unlike the case of Adeleke, the facts here are different.",
                "distinguished",
            ),
            (
                "The case does not apply here as it is inapplicable.",
                "distinguished",
            ),
            # overruled signals
            (
                "That decision was overruled by the Supreme Court.",
                "overruled",
            ),
            (
                "The court declined to follow the earlier authority.",
                "overruled",
            ),
            (
                "The earlier case is no longer good law.",
                "overruled",
            ),
            # default — mentioned
            (
                "The court referred to the case of Adeleke v. Obi.",
                "mentioned",
            ),
            (
                "Counsel cited (2020) 15 NWLR (Pt. 1748) 1 in his brief.",
                "mentioned",
            ),
        ],
    )
    def test_classify(self, context: str, expected: str) -> None:
        result = CLASSIFIER.classify(context)
        assert result == expected, (
            f"Expected '{expected}' for context: {repr(context)!r}, got '{result}'"
        )

    def test_overruled_takes_priority_over_followed(self) -> None:
        # A sentence containing both "followed" and "overruled" signals
        # should resolve to "overruled" (strongest signal wins).
        context = "Although originally followed, it was later overruled."
        assert CLASSIFIER.classify(context) == "overruled"

    def test_distinguished_takes_priority_over_followed(self) -> None:
        context = "The case was applied but distinguished on the facts."
        assert CLASSIFIER.classify(context) == "distinguished"

    def test_empty_context_returns_mentioned(self) -> None:
        assert CLASSIFIER.classify("") == "mentioned"

    def test_case_insensitive(self) -> None:
        assert CLASSIFIER.classify("The court FOLLOWED the decision.") == "followed"
        assert CLASSIFIER.classify("The case was OVERRULED.") == "overruled"


# ── Integration: extract then classify ───────────────────────────────────────


def test_extract_and_classify_integration() -> None:
    """Citations extracted from real-style text should classify correctly."""
    text = (
        "Learned counsel for the appellant relied on Adeleke v. Obi "
        "(2020) 15 NWLR (Pt. 1748) 1 which this court followed in reaching "
        "its decision.\n"
        "The case of Bello v. State (1986) 2 SC 87 was distinguished "
        "on the grounds that the facts are not on all fours."
    )
    citations = EXTRACTOR.extract_all(text)
    assert len(citations) == 2

    treatments = [CLASSIFIER.classify(c.context) for c in citations]
    assert "followed" in treatments
    assert "distinguished" in treatments
