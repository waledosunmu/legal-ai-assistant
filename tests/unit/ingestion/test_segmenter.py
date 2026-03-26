"""Tests for Phase 0 Week 3: segmentation and citation graph builder."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion.citations.extractor import ExtractedCitation
from ingestion.citations.graph_builder import CitationEdge, CitationGraphBuilder
from ingestion.segmentation.llm_segmenter import LLMSegmenter
from ingestion.segmentation.models import JudgmentSegment, SegmentType
from ingestion.segmentation.nlp_rules import NLPSegmentClassifier
from ingestion.segmentation.structural import StructuralSegmenter


# ── StructuralSegmenter ────────────────────────────────────────────────────────

class TestStructuralSegmenterParagraphSplit:
    def test_single_paragraph(self):
        segmenter = StructuralSegmenter()
        result = segmenter._split_paragraphs("This is a single paragraph with enough text.")
        assert len(result) == 1

    def test_two_paragraphs_separated_by_blank_line(self):
        segmenter = StructuralSegmenter()
        text = "First paragraph with enough content.\n\nSecond paragraph with enough content."
        result = segmenter._split_paragraphs(text)
        assert len(result) == 2

    def test_short_paragraphs_filtered(self):
        segmenter = StructuralSegmenter()
        text = "ok\n\nThis is a paragraph that is long enough to pass the filter."
        result = segmenter._split_paragraphs(text)
        assert len(result) == 1

    def test_empty_text_returns_empty(self):
        segmenter = StructuralSegmenter()
        assert segmenter.segment("") == []

    def test_only_short_content_returns_empty(self):
        segmenter = StructuralSegmenter()
        assert segmenter.segment("Hi\n\nOk\n\nBye") == []


class TestStructuralSegmenterClassify:
    def test_issues_pattern_match(self):
        segmenter = StructuralSegmenter()
        para = "ISSUES FOR DETERMINATION\n1. Whether the plaintiff had standing."
        seg_type, confidence = segmenter._classify_paragraph(para, 5, 20)
        assert seg_type == SegmentType.ISSUES
        assert confidence == 0.8

    def test_held_pattern_gives_holding(self):
        segmenter = StructuralSegmenter()
        para = "HELD: The appeal fails. The decision of the lower court is affirmed."
        seg_type, confidence = segmenter._classify_paragraph(para, 15, 20)
        assert seg_type == SegmentType.HOLDING
        assert confidence == 0.8

    def test_dissent_pattern(self):
        segmenter = StructuralSegmenter()
        para = "DISSENTING JUDGMENT by Okafor, JSC"
        seg_type, confidence = segmenter._classify_paragraph(para, 18, 20)
        assert seg_type == SegmentType.DISSENT
        assert confidence == 0.8

    def test_orders_pattern(self):
        segmenter = StructuralSegmenter()
        para = "Accordingly, this court orders that the appeal is dismissed with costs."
        seg_type, confidence = segmenter._classify_paragraph(para, 19, 20)
        assert seg_type == SegmentType.ORDERS
        assert confidence == 0.8

    def test_positional_early_gives_caption(self):
        segmenter = StructuralSegmenter()
        # position=0 out of 20 → rel=0/19 < 0.05
        para = "Some opening text without any pattern match but content enough."
        seg_type, confidence = segmenter._classify_paragraph(para, 0, 20)
        assert seg_type == SegmentType.CAPTION
        assert confidence == 0.5

    def test_positional_near_start_gives_background(self):
        segmenter = StructuralSegmenter()
        # position=2 out of 20 → rel=2/19 ≈ 0.105, between 0.05 and 0.15
        para = "Some text without any pattern match and is content enough to keep."
        seg_type, confidence = segmenter._classify_paragraph(para, 2, 20)
        assert seg_type == SegmentType.BACKGROUND
        assert confidence == 0.4

    def test_positional_very_late_gives_orders(self):
        segmenter = StructuralSegmenter()
        # position=19 out of 20 → rel=19/19=1.0 > 0.90
        para = "Some text at the end without any pattern match but enough chars."
        seg_type, confidence = segmenter._classify_paragraph(para, 19, 20)
        assert seg_type == SegmentType.ORDERS
        assert confidence == 0.4

    def test_default_mid_document_gives_analysis(self):
        segmenter = StructuralSegmenter()
        # position=10 out of 20 → rel=10/19 ≈ 0.53 — middle, no pattern match
        para = "Some unrecognised paragraph in the middle of the judgment text here."
        seg_type, confidence = segmenter._classify_paragraph(para, 10, 20)
        assert seg_type == SegmentType.ANALYSIS
        assert confidence == 0.3


class TestStructuralSegmenterMerge:
    def test_consecutive_same_type_merged(self):
        segmenter = StructuralSegmenter()
        segments = [
            JudgmentSegment(SegmentType.ANALYSIS, "Para A.", 0, 0.3),
            JudgmentSegment(SegmentType.ANALYSIS, "Para B.", 1, 0.4),
        ]
        merged = segmenter._merge_consecutive(segments)
        assert len(merged) == 1
        assert "Para A." in merged[0].content
        assert "Para B." in merged[0].content
        assert merged[0].confidence == 0.3  # min of the two

    def test_different_types_not_merged(self):
        segmenter = StructuralSegmenter()
        segments = [
            JudgmentSegment(SegmentType.ISSUES, "Issues.", 0, 0.8),
            JudgmentSegment(SegmentType.ANALYSIS, "Analysis.", 1, 0.5),
        ]
        merged = segmenter._merge_consecutive(segments)
        assert len(merged) == 2

    def test_empty_segments(self):
        segmenter = StructuralSegmenter()
        assert segmenter._merge_consecutive([]) == []


class TestStructuralSegmenterIntegration:
    def test_segment_full_text(self):
        text = """IN THE SUPREME COURT OF NIGERIA
Suit No. SC.100/2020

This is the background of the case. The plaintiff appealed the lower court decision.

ISSUES FOR DETERMINATION

The following issues were formulated for determination:
1. Whether the lower court erred in law.

I have carefully considered the submissions of learned counsel on both sides.
The law is well settled on this point.

HELD: The appeal fails. For the foregoing reasons, this court holds that the
decision of the lower court was correct.

Accordingly, the appeal is dismissed with costs assessed at N100,000."""

        segmenter = StructuralSegmenter()
        result = segmenter.segment(text)

        types = [s.segment_type for s in result]
        assert SegmentType.ISSUES in types
        assert SegmentType.HOLDING in types
        assert SegmentType.ORDERS in types

    def test_segment_returns_sorted_by_position(self):
        text = "Para one long enough.\n\n" * 5
        segmenter = StructuralSegmenter()
        result = segmenter.segment(text)
        positions = [s.position for s in result]
        assert positions == sorted(positions)


# ── NLPSegmentClassifier ───────────────────────────────────────────────────────

class TestNLPSegmentClassifierHeading:
    def test_all_caps_short_is_heading(self):
        assert NLPSegmentClassifier._is_heading("BACKGROUND") is True

    def test_all_caps_with_spaces_is_heading(self):
        assert NLPSegmentClassifier._is_heading("ISSUES FOR DETERMINATION") is True

    def test_mixed_case_not_heading(self):
        assert NLPSegmentClassifier._is_heading("This is a normal paragraph.") is False

    def test_long_all_caps_not_heading(self):
        assert NLPSegmentClassifier._is_heading(
            "THIS IS A VERY LONG HEADING THAT HAS MORE THAN TEN WORDS IN IT"
        ) is False

    def test_empty_string_not_heading(self):
        assert NLPSegmentClassifier._is_heading("") is False


class TestNLPSegmentClassifierReclassify:
    def _make_seg(self, content: str, seg_type=SegmentType.ANALYSIS, confidence=0.3):
        return JudgmentSegment(
            segment_type=seg_type,
            content=content,
            position=0,
            confidence=confidence,
        )

    def test_high_confidence_unchanged(self):
        clf = NLPSegmentClassifier()
        seg = self._make_seg("Some content.", confidence=0.75)
        result = clf.reclassify([seg])
        assert result[0].confidence == 0.75
        assert result[0].segment_type == SegmentType.ANALYSIS

    def test_all_caps_heading_reclassified_as_caption(self):
        clf = NLPSegmentClassifier()
        seg = self._make_seg("BACKGROUND", confidence=0.3)
        result = clf.reclassify([seg])
        assert result[0].segment_type == SegmentType.CAPTION
        assert result[0].confidence == 0.65

    def test_held_keyword_boosts_holding(self):
        clf = NLPSegmentClassifier()
        content = "For the above reasons, we hold that the appeal is dismissed."
        seg = self._make_seg(content, seg_type=SegmentType.ANALYSIS, confidence=0.3)
        result = clf.reclassify([seg])
        assert result[0].segment_type == SegmentType.HOLDING
        assert result[0].confidence > 0.3

    def test_issues_keyword_boosts_issues(self):
        clf = NLPSegmentClassifier()
        content = "The following issues for determination have been formulated."
        seg = self._make_seg(content, confidence=0.3)
        result = clf.reclassify([seg])
        assert result[0].segment_type == SegmentType.ISSUES

    def test_no_keyword_signal_unchanged(self):
        clf = NLPSegmentClassifier()
        content = "Lorem ipsum dolor sit amet consectetur adipiscing elit."
        seg = self._make_seg(content, confidence=0.3)
        result = clf.reclassify([seg])
        assert result[0].confidence == 0.3
        assert result[0].segment_type == SegmentType.ANALYSIS

    def test_confidence_capped_at_0_9(self):
        clf = NLPSegmentClassifier()
        content = (
            "HELD In the result this appeal is dismissed. "
            "We hold that the decision is upheld."
        )
        seg = self._make_seg(content, confidence=0.3)
        result = clf.reclassify([seg])
        assert result[0].confidence <= 0.90


# ── LLMSegmenter ──────────────────────────────────────────────────────────────

def _make_mock_client(response_text: str) -> MagicMock:
    """Return a mock Anthropic client that returns response_text."""
    mock_content = MagicMock()
    mock_content.text = response_text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


_VALID_RESPONSE = """{
  "issues": ["Whether the appellant had locus standi."],
  "holdings": [{"issue": "locus standi", "determination": "yes", "reasoning": "The court applied established principles."}],
  "ratio_decidendi": "A party must demonstrate sufficient interest to sue.",
  "obiter_dicta": ["Courts should avoid technicality."],
  "orders": ["Appeal dismissed."],
  "cases_cited": [{"name": "Adesanya v. President", "citation": "(1981) 2 NCLR 358", "treatment": "followed", "context": "The court followed this authority."}]
}"""


class TestLLMSegmenter:
    def test_happy_path_parses_json(self):
        client = _make_mock_client(_VALID_RESPONSE)
        segmenter = LLMSegmenter(client=client)
        result = segmenter.segment("Some judgment text of sufficient length.")
        assert result["issues"] == ["Whether the appellant had locus standi."]
        assert result["ratio_decidendi"] is not None
        assert len(result["orders"]) == 1

    def test_markdown_fence_stripped(self):
        fenced = "```json\n" + _VALID_RESPONSE + "\n```"
        client = _make_mock_client(fenced)
        segmenter = LLMSegmenter(client=client)
        result = segmenter.segment("Judgment text here.")
        assert "issues" in result
        assert "_parse_error" not in result

    def test_malformed_json_returns_fallback(self):
        client = _make_mock_client("This is not JSON at all.")
        segmenter = LLMSegmenter(client=client)
        result = segmenter.segment("Judgment text here.")
        assert result["_parse_error"] is True
        assert result["issues"] == []

    def test_long_text_truncated(self):
        """Text longer than max_input_chars should be truncated before the API call."""
        client = _make_mock_client(_VALID_RESPONSE)
        segmenter = LLMSegmenter(client=client)
        long_text = "A" * 200_000
        segmenter.segment(long_text, max_input_chars=1_000)
        call_args = client.messages.create.call_args
        sent_content = call_args.kwargs["messages"][0]["content"]
        assert "omitted for processing" in sent_content

    def test_short_text_not_truncated(self):
        client = _make_mock_client(_VALID_RESPONSE)
        segmenter = LLMSegmenter(client=client)
        short_text = "Short judgment text."
        segmenter.segment(short_text)
        call_args = client.messages.create.call_args
        sent_content = call_args.kwargs["messages"][0]["content"]
        assert "omitted for processing" not in sent_content

    def test_estimate_cost_returns_dict(self):
        segmenter = LLMSegmenter(client=MagicMock())
        result = segmenter.estimate_cost(num_cases=1000, avg_tokens=15_000)
        assert result["num_cases"] == 1000
        assert result["estimated_input_tokens"] == 15_000_000
        assert result["estimated_cost_usd"] > 0


# ── CitationGraphBuilder ───────────────────────────────────────────────────────

def _make_citation(
    raw_text: str,
    case_name: str | None,
    context: str,
    position: int = 0,
) -> ExtractedCitation:
    return ExtractedCitation(
        raw_text=raw_text,
        case_name=case_name,
        year=2020,
        report_series="NWLR",
        volume="15",
        part="1000",
        page="1",
        full_citation=f"{case_name} {raw_text}" if case_name else raw_text,
        position=position,
        context=context,
    )


class TestCitationGraphBuilder:
    def test_build_edges_creates_one_edge_per_unique_citation(self):
        builder = CitationGraphBuilder()
        citations = [
            _make_citation("(2020) 1 NWLR (Pt.100) 1", "Malami v. Ohikhuare", "The court followed this authority.", 0),
            _make_citation("(2019) 2 SC 50", "Eze v. Okafor", "The case was distinguished.", 100),
        ]
        edges = builder.build_edges("case_001", citations, {})
        assert len(edges) == 2
        assert all(isinstance(e, CitationEdge) for e in edges)

    def test_duplicate_citations_deduplicated(self):
        builder = CitationGraphBuilder()
        cite = _make_citation("(2020) 1 NWLR (Pt.100) 1", "A v. B", "mentioned.", 0)
        edges = builder.build_edges("case_001", [cite, cite], {})
        assert len(edges) == 1

    def test_treatment_followed(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2020) 5 SC 10", "X v. Y", "The court followed this decision.", 0)
        edges = builder.build_edges("case_001", [citation], {})
        assert edges[0].treatment == "followed"

    def test_treatment_distinguished(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2018) 3 SC 5", "A v. B", "The case was distinguished from the present facts.", 0)
        edges = builder.build_edges("case_001", [citation], {})
        assert edges[0].treatment == "distinguished"

    def test_treatment_overruled(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2005) 1 SC 1", "C v. D", "The earlier decision was overruled.", 0)
        edges = builder.build_edges("case_001", [citation], {})
        assert edges[0].treatment == "overruled"

    def test_treatment_default_mentioned(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2010) 2 SC 20", "E v. F", "The court noted this case in passing.", 0)
        edges = builder.build_edges("case_001", [citation], {})
        assert edges[0].treatment == "mentioned"

    def test_citing_case_id_preserved(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2020) 1 SC 1", "G v. H", "mentioned.", 0)
        edges = builder.build_edges("my_case_id", [citation], {})
        assert edges[0].citing_case_id == "my_case_id"


class TestCitationGraphBuilderResolution:
    def test_resolve_no_registry_returns_none(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2020) 1 SC 1", "Malami v. Ohikhuare", "ctx", 0)
        result = builder._resolve_to_case_id(citation, {})
        assert result is None

    def test_resolve_exact_fuzzy_match(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2020) 1 SC 1", "Malami v. Ohikhuare", "ctx", 0)
        registry = {"case_42": "Malami v. Ohikhuare"}
        result = builder._resolve_to_case_id(citation, registry)
        assert result == "case_42"

    def test_resolve_close_fuzzy_match(self):
        builder = CitationGraphBuilder()
        # Slight variation in spelling
        citation = _make_citation("(2020) 1 SC 1", "Malami v Ohikhuare", "ctx", 0)
        registry = {"case_42": "Malami v. Ohikhuare"}
        result = builder._resolve_to_case_id(citation, registry)
        assert result == "case_42"

    def test_resolve_no_match_below_threshold(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2020) 1 SC 1", "Completely Different Name", "ctx", 0)
        registry = {"case_42": "Malami v. Ohikhuare"}
        result = builder._resolve_to_case_id(citation, registry)
        assert result is None

    def test_resolve_no_case_name_returns_none(self):
        builder = CitationGraphBuilder()
        citation = _make_citation("(2020) 1 SC 1", None, "ctx", 0)
        registry = {"case_42": "Malami v. Ohikhuare"}
        result = builder._resolve_to_case_id(citation, registry)
        assert result is None


class TestCitationGraphBuilderAuthorityScores:
    def _make_edge(self, citing: str, cited: str | None) -> CitationEdge:
        return CitationEdge(
            citing_case_id=citing,
            cited_case_id=cited,
            cited_case_name="X v. Y",
            cited_citation="(2020) 1 SC 1",
            treatment="mentioned",
            context="",
        )

    def test_most_cited_has_score_1(self):
        builder = CitationGraphBuilder()
        edges = [
            self._make_edge("a", "target"),
            self._make_edge("b", "target"),
            self._make_edge("c", "target"),
            self._make_edge("d", "other"),
        ]
        scores = builder.compute_authority_scores(edges)
        assert scores["target"] == 1.0
        assert scores["other"] < 1.0

    def test_empty_edges_returns_empty(self):
        builder = CitationGraphBuilder()
        assert builder.compute_authority_scores([]) == {}

    def test_unresolved_edges_excluded(self):
        builder = CitationGraphBuilder()
        edges = [self._make_edge("a", None)]
        scores = builder.compute_authority_scores(edges)
        assert scores == {}

    def test_scores_between_0_and_1(self):
        builder = CitationGraphBuilder()
        edges = [
            self._make_edge("a", "X"),
            self._make_edge("b", "X"),
            self._make_edge("c", "Y"),
        ]
        scores = builder.compute_authority_scores(edges)
        for s in scores.values():
            assert 0.0 <= s <= 1.0
