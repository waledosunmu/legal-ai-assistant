"""Unit tests for src/retrieval/fusion.py — pure unit tests, no DB needed."""

from __future__ import annotations

import pytest

from retrieval.fusion import RRFFusion, _authority_boost, _recency_boost


def _row(
    segment_id: str,
    case_id: str,
    seg_type: str = "RATIO",
    court: str = "NGSC",
    year: int = 2020,
    opinion: str = "LEAD",
    weight: float = 1.0,
) -> dict:
    return {
        "segment_id": segment_id,
        "case_id": case_id,
        "segment_type": seg_type,
        "content": f"content of {segment_id}",
        "court": court,
        "year": year,
        "opinion_type": opinion,
        "retrieval_weight": weight,
    }


class TestRRFFusion:
    fusion = RRFFusion()

    def test_returns_empty_for_empty_input(self) -> None:
        result = self.fusion.fuse([], [])
        assert result == []

    def test_single_result_list(self) -> None:
        rows = [_row("s1", "c1"), _row("s2", "c2")]
        result = self.fusion.fuse([rows], [])
        assert len(result) == 2

    def test_rrf_score_higher_for_top_ranked(self) -> None:
        """Segment ranked #1 in both lists should outscore segment ranked last."""
        top = _row("s-top", "case-top")
        bottom = _row("s-bot", "case-bot")
        dense = [top, bottom]
        sparse = [top, bottom]
        result = self.fusion.fuse([dense], [sparse])
        scores = {c.segment_id: c.fusion_score for c in result}
        assert scores["s-top"] > scores["s-bot"]

    def test_deduplicates_by_case_id(self) -> None:
        """Multiple segments from the same case → only highest-scoring survives."""
        s1 = _row("seg-1", "same-case")
        s2 = _row("seg-2", "same-case")
        result = self.fusion.fuse([[s1, s2]], [])
        case_ids = [c.case_id for c in result]
        assert case_ids.count("same-case") == 1

    def test_court_boost_applies(self) -> None:
        """Supreme Court segment should outscore Federal High Court (same rank)."""
        sc = _row("s-sc", "case-sc", court="NGSC")
        fhc = _row("s-fhc", "case-fhc", court="NGFCHC")
        result = self.fusion.fuse([[sc, fhc]], [[sc, fhc]])
        sc_result = next(c for c in result if c.case_id == "case-sc")
        fhc_result = next(c for c in result if c.case_id == "case-fhc")
        assert sc_result.boosted_score > fhc_result.boosted_score

    def test_segment_type_boost_ratio_over_facts(self) -> None:
        """RATIO segment should outscore FACTS segment from same case rank."""
        ratio = _row("s-ratio", "case-r", seg_type="RATIO")
        facts = _row("s-facts", "case-f", seg_type="FACTS")
        result = self.fusion.fuse([[ratio, facts]], [[ratio, facts]])
        ratio_result = next(c for c in result if c.case_id == "case-r")
        facts_result = next(c for c in result if c.case_id == "case-f")
        assert ratio_result.boosted_score > facts_result.boosted_score

    def test_authority_boost_applies(self) -> None:
        """Highly cited case should outscore uncited case at same rank."""
        cited = _row("s-cited", "case-cited")
        uncited = _row("s-uncited", "case-uncited")
        authority_scores = {"case-cited": 500, "case-uncited": 0}
        result = self.fusion.fuse(
            [[cited, uncited]],
            [[cited, uncited]],
            authority_scores=authority_scores,
        )
        cited_result = next(c for c in result if c.case_id == "case-cited")
        uncited_result = next(c for c in result if c.case_id == "case-uncited")
        assert cited_result.boosted_score > uncited_result.boosted_score

    def test_recency_boost_newer_over_older(self) -> None:
        """Newer case (same rank) should outscore older case."""
        new = _row("s-new", "case-new", year=2022)
        old = _row("s-old", "case-old", year=1980)
        result = self.fusion.fuse([[new, old]], [], current_year=2024)
        new_result = next(c for c in result if c.case_id == "case-new")
        old_result = next(c for c in result if c.case_id == "case-old")
        assert new_result.boosted_score > old_result.boosted_score

    def test_top_n_limit(self) -> None:
        rows = [_row(f"s-{i}", f"c-{i}") for i in range(20)]
        result = self.fusion.fuse([rows], [], top_n=5)
        assert len(result) <= 5

    def test_sorted_by_boosted_score_descending(self) -> None:
        rows = [_row(f"s-{i}", f"c-{i}") for i in range(10)]
        result = self.fusion.fuse([rows], [])
        scores = [c.boosted_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_dense_and_sparse_ranks_recorded(self) -> None:
        row = _row("s1", "c1")
        result = self.fusion.fuse([[row]], [[row]])
        assert result[0].dense_rank == 1
        assert result[0].sparse_rank == 1

    def test_none_rank_when_not_in_list(self) -> None:
        row = _row("s1", "c1")
        result = self.fusion.fuse([[row]], [])  # no sparse
        assert result[0].dense_rank == 1
        assert result[0].sparse_rank is None

    def test_dissenting_opinion_downranked(self) -> None:
        lead = _row("s-lead", "c-lead", opinion="LEAD")
        dissent = _row("s-dissent", "c-dissent", opinion="DISSENTING")
        result = self.fusion.fuse([[lead, dissent]], [])
        lead_result = next(c for c in result if c.case_id == "c-lead")
        dissent_result = next(c for c in result if c.case_id == "c-dissent")
        assert lead_result.boosted_score > dissent_result.boosted_score


class TestBoostHelpers:
    def test_recency_boost_current_year_is_one(self) -> None:
        assert _recency_boost(2024, 2024, half_life=15) == pytest.approx(1.0)

    def test_recency_boost_decreases_with_age(self) -> None:
        boost_10 = _recency_boost(2014, 2024, half_life=15)
        boost_20 = _recency_boost(2004, 2024, half_life=15)
        assert boost_10 > boost_20
        assert 0 < boost_20 < boost_10 < 1.0

    def test_authority_boost_zero_for_no_citations(self) -> None:
        assert _authority_boost(0, 100, max_boost=0.10) == pytest.approx(0.0)

    def test_authority_boost_max_for_max_citations(self) -> None:
        assert _authority_boost(500, 500, max_boost=0.10) == pytest.approx(0.10)

    def test_authority_boost_intermediate(self) -> None:
        boost = _authority_boost(50, 500, max_boost=0.10)
        assert 0 < boost < 0.10

    def test_authority_boost_handles_zero_max(self) -> None:
        assert _authority_boost(0, 0, max_boost=0.10) == pytest.approx(0.0)
