"""Unit tests for JudgmentTextCleaner and MetadataExtractor."""

from __future__ import annotations

from datetime import date

import pytest

from ingestion.parsing.metadata_extractor import JudgmentMetadata, MetadataExtractor
from ingestion.parsing.text_cleaner import JudgmentTextCleaner
from ingestion.sources.nigerialii import Court, RawJudgment

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_raw(
    case_name: str = "Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5 (22 June 2017)",
    court: Court = Court.SUPREME_COURT,
    full_text: str = "",
    judges: list[str] | None = None,
    citation: str = "[2017] NGSC 5",
    case_number: str = "SC. 373/2015",
    judgment_date: date | None = date(2017, 6, 22),
) -> RawJudgment:
    return RawJudgment(
        case_name=case_name,
        source_url="https://nigerialii.org/test",
        court=court,
        media_neutral_citation=citation,
        case_number=case_number,
        judges=judges or ["Yahaya, JSC", "Onnoghen, JSC"],
        judgment_date=judgment_date,
        full_text=full_text,
    )


# ── JudgmentTextCleaner ───────────────────────────────────────────────────────


class TestJudgmentTextCleaner:
    def setup_method(self) -> None:
        self.cleaner = JudgmentTextCleaner()

    def test_fix_encoding_removes_mojibake(self) -> None:
        text = "Appellant\u2019s counsel argued â€œthe agreement was validâ€\x9d."
        result = self.cleaner._fix_encoding(text)
        assert "â€œ" not in result
        assert "â€\x9d" not in result

    def test_fix_encoding_removes_nbsp(self) -> None:
        text = "In\u00a0the\u00a0Supreme\u00a0Court"
        result = self.cleaner._fix_encoding(text)
        assert "\u00a0" not in result
        assert result == "In the Supreme Court"

    def test_fix_ocr_artifacts_replaces_ligatures(self) -> None:
        text = "The af?davit was satis?ed and justi?ed."
        result = self.cleaner._fix_ocr_artifacts(text)
        assert "affidavit" in result
        assert "satisfied" in result
        assert "justified" in result
        assert "?" not in result

    def test_fix_ocr_certificate(self) -> None:
        result = self.cleaner._fix_ocr_artifacts("certi?cate of occupancy")
        assert result == "certificate of occupancy"

    def test_remove_page_noise_strips_pagination(self) -> None:
        text = "First paragraph.\nPage 3 of 27\nSecond paragraph."
        result = self.cleaner._remove_page_noise(text)
        assert "Page 3 of 27" not in result

    def test_remove_page_noise_strips_page_separator(self) -> None:
        text = "End of page.\n - 12 - \nStart of next page."
        result = self.cleaner._remove_page_noise(text)
        assert "- 12 -" not in result

    def test_normalise_whitespace_collapses_blank_lines(self) -> None:
        text = "Para one.\n\n\n\n\nPara two."
        result = self.cleaner._normalise_whitespace(text)
        assert "\n\n\n" not in result
        assert "Para one." in result
        assert "Para two." in result

    def test_normalise_whitespace_removes_trailing_spaces(self) -> None:
        text = "Line one.   \nLine two.  "
        result = self.cleaner._normalise_whitespace(text)
        for line in result.splitlines():
            assert not line.endswith(" "), f"Trailing space in: {repr(line)}"

    def test_normalise_legal_formatting_standardises_versus(self) -> None:
        assert "v." in self.cleaner._normalise_legal_formatting("Adeleke vs Obi")
        assert "v." in self.cleaner._normalise_legal_formatting("Adeleke vs. Obi")

    def test_clean_full_pipeline(self) -> None:
        dirty = (
            "IN THE SUPREME COURT OF NIGERIA\n\n"
            "Adeleke vs Obi\nPage 1 of 10\n\n\n\n"
            "The af?davit was ?led in court.\n"
        )
        result = self.cleaner.clean(dirty)
        assert "Page 1 of 10" not in result
        assert "affidavit" in result
        assert "filed" in result
        assert "v." in result
        assert "\n\n\n" not in result

    def test_clean_returns_stripped_text(self) -> None:
        result = self.cleaner.clean("   \n  Hello world.  \n  ")
        assert result == result.strip()

    def test_clean_preserves_paragraph_structure(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        result = self.cleaner.clean(text)
        assert "\n\n" in result


# ── MetadataExtractor — party extraction ──────────────────────────────────────


class TestMetadataExtractorParties:
    def setup_method(self) -> None:
        self.extractor = MetadataExtractor()

    @pytest.mark.parametrize(
        "case_name, expected_applicant, expected_respondent",
        [
            (
                "Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5 (22 June 2017)",
                "Adeleke",
                "Obi",
            ),
            (
                "MALAMI v. OHIKHUARE",
                "MALAMI",
                "OHIKHUARE",
            ),
            (
                "ALHAJI ABUBAKAR v. FEDERAL REPUBLIC OF NIGERIA",
                "ALHAJI ABUBAKAR",
                "FEDERAL REPUBLIC OF NIGERIA",
            ),
            (
                "CHIEF BISI AKANDE v. THE STATE",
                "CHIEF BISI AKANDE",
                "THE STATE",
            ),
        ],
    )
    def test_extract_parties(
        self,
        case_name: str,
        expected_applicant: str,
        expected_respondent: str,
    ) -> None:
        raw = make_raw(case_name=case_name)
        metadata = self.extractor.extract(raw)
        assert expected_applicant in metadata.applicant
        assert expected_respondent in metadata.respondent

    def test_extract_parties_fallback_no_v(self) -> None:
        raw = make_raw(case_name="In Re: Estate of Bello")
        metadata = self.extractor.extract(raw)
        assert metadata.applicant  # should not be empty
        assert metadata.respondent == ""


# ── MetadataExtractor — short name ────────────────────────────────────────────


class TestMetadataExtractorShortName:
    def setup_method(self) -> None:
        self.extractor = MetadataExtractor()

    @pytest.mark.parametrize(
        "case_name, expected_contains",
        [
            ("Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5", "Adeleke v. Obi"),
            ("MALAMI v. OHIKHUARE", "Malami v. Ohikhuare"),
            (
                "ALHAJI ABUBAKAR v. THE STATE",
                "v.",  # ALHAJI stripped, Abubakar remains
            ),
        ],
    )
    def test_make_short_name(self, case_name: str, expected_contains: str) -> None:
        raw = make_raw(case_name=case_name)
        metadata = self.extractor.extract(raw)
        assert expected_contains in metadata.case_name_short


# ── MetadataExtractor — area of law ───────────────────────────────────────────


class TestMetadataExtractorAreaOfLaw:
    def setup_method(self) -> None:
        self.extractor = MetadataExtractor()

    def test_infers_criminal_law(self) -> None:
        raw = make_raw(
            case_name="Federal Republic of Nigeria v. Bello",
            full_text="The accused was convicted of armed robbery and sentenced.",
        )
        metadata = self.extractor.extract(raw)
        assert "criminal" in metadata.area_of_law

    def test_infers_land_law(self) -> None:
        raw = make_raw(
            full_text="The dispute concerns a certificate of occupancy over the land.",
        )
        metadata = self.extractor.extract(raw)
        assert "land_law" in metadata.area_of_law

    def test_infers_contract_law(self) -> None:
        raw = make_raw(
            full_text="The plaintiff seeks damages for breach of contract.",
        )
        metadata = self.extractor.extract(raw)
        assert "contract" in metadata.area_of_law

    def test_infers_constitutional(self) -> None:
        raw = make_raw(
            full_text=(
                "This is an application for enforcement of fundamental rights "
                "under the Constitution of the Federal Republic."
            ),
        )
        metadata = self.extractor.extract(raw)
        assert "constitutional" in metadata.area_of_law

    def test_defaults_to_general(self) -> None:
        raw = make_raw(full_text="The court considered the matter before it.")
        metadata = self.extractor.extract(raw)
        assert metadata.area_of_law == ["general"]

    def test_multiple_areas_detected(self) -> None:
        raw = make_raw(
            full_text=(
                "A company director was convicted of fraud following a breach of "
                "contract. The land in dispute was also subject to a certificate "
                "of occupancy."
            ),
        )
        metadata = self.extractor.extract(raw)
        assert len(metadata.area_of_law) >= 2


# ── MetadataExtractor — lead judge ────────────────────────────────────────────


class TestMetadataExtractorLeadJudge:
    def setup_method(self) -> None:
        self.extractor = MetadataExtractor()

    @pytest.mark.parametrize(
        "text_fragment, expected_name_contains",
        [
            ("This judgment was Delivered by EJEMBI EKO, JSC", "EKO"),
            ("(Per RHODES-VIVOUR, JSC): The appeal succeeds.", "RHODES-VIVOUR"),
            ("LEAD JUDGMENT: YAHAYA, JSC", "YAHAYA"),
            ("Read by OKO, JCA on behalf of the panel.", "OKO"),
        ],
    )
    def test_extract_lead_judge(self, text_fragment: str, expected_name_contains: str) -> None:
        raw = make_raw(full_text=text_fragment)
        metadata = self.extractor.extract(raw)
        assert metadata.lead_judge is not None
        assert expected_name_contains in metadata.lead_judge

    def test_lead_judge_none_when_absent(self) -> None:
        raw = make_raw(full_text="The court considered the application.")
        metadata = self.extractor.extract(raw)
        assert metadata.lead_judge is None


# ── MetadataExtractor — full extract ─────────────────────────────────────────


def test_extract_returns_judgment_metadata() -> None:
    raw = make_raw(
        case_name="Adeleke v. Obi (SC. 373/2015) [2017] NGSC 5 (22 June 2017)",
        full_text=(
            "Delivered by EJEMBI EKO, JSC.\n\n"
            "The parties entered into a contract for the sale of land."
        ),
        judgment_date=date(2017, 6, 22),
    )
    extractor = MetadataExtractor()
    meta = extractor.extract(raw)

    assert isinstance(meta, JudgmentMetadata)
    assert meta.case_number == "SC. 373/2015"
    assert meta.media_neutral_citation == "[2017] NGSC 5"
    assert meta.judgment_date == date(2017, 6, 22)
    assert "Yahaya, JSC" in meta.judges
    assert meta.lead_judge is not None
    assert "EKO" in meta.lead_judge
    assert "contract" in meta.area_of_law or "land_law" in meta.area_of_law


def test_extract_handles_string_judgment_date() -> None:
    """Judgment dates may arrive as strings after JSONL roundtrip."""
    raw = make_raw(judgment_date=None)
    raw.judgment_date = "2017-06-22"  # type: ignore[assignment]
    extractor = MetadataExtractor()
    meta = extractor.extract(raw)
    assert meta.judgment_date == date(2017, 6, 22)


def test_extract_handles_none_judgment_date() -> None:
    raw = make_raw(judgment_date=None)
    extractor = MetadataExtractor()
    meta = extractor.extract(raw)
    assert meta.judgment_date is None
