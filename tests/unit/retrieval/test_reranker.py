"""Unit tests for src/retrieval/reranker.py"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from retrieval.models import CandidateResult, RetrievalConfig
from retrieval.reranker import LLMReranker, _short_name


def _candidate(
    segment_id: str,
    case_id: str,
    fusion_score: float = 0.5,
    boosted_score: float = 0.5,
    seg_type: str = "RATIO",
    content: str = "test content",
    court: str = "NGSC",
    year: int = 2020,
) -> CandidateResult:
    return CandidateResult(
        case_id=case_id,
        segment_id=segment_id,
        segment_type=seg_type,
        content=content,
        court=court,
        year=year,
        fusion_score=fusion_score,
        boosted_score=boosted_score,
    )


def _meta(case_id: str, name: str = "Case Name", citation: str | None = None) -> dict:
    return {
        case_id: {
            "case_name": name,
            "citation": citation or "(2020) 1 NWLR 1",
            "court": "NGSC",
            "year": 2020,
            "times_cited": 10,
            "authority_score": 10,
        }
    }


def _make_anthropic(scores: list[dict]) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(scores))]
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=msg)
    return client


class TestLLMReranker:
    @pytest.mark.asyncio
    async def test_returns_empty_for_no_candidates(self) -> None:
        reranker = LLMReranker()
        result = await reranker.rerank("test query", [], {})
        assert result == []

    @pytest.mark.asyncio
    async def test_fallback_fusion_when_no_anthropic(self) -> None:
        reranker = LLMReranker(anthropic_client=None)
        c1 = _candidate("s1", "c1", boosted_score=0.8)
        c2 = _candidate("s2", "c2", boosted_score=0.3)
        meta = {**_meta("c1", "Alpha Case"), **_meta("c2", "Beta Case")}
        result = await reranker.rerank("test", [c1, c2], meta)
        assert len(result) == 2
        # Should be sorted by normalised fusion score
        assert result[0].case_id == "c1"
        assert result[1].case_id == "c2"

    @pytest.mark.asyncio
    async def test_llm_scores_applied(self) -> None:
        """LLM gives high score to c2, which should then outrank c1."""
        llm_data = [
            {"id": "s1", "factual_similarity": 0.2, "principle_applicability": 0.2,
             "authority_strength": 0.2, "explanation": "weakly relevant"},
            {"id": "s2", "factual_similarity": 0.9, "principle_applicability": 0.9,
             "authority_strength": 0.9, "explanation": "highly relevant"},
        ]
        anthropic = _make_anthropic(llm_data)
        reranker = LLMReranker(anthropic_client=anthropic)
        c1 = _candidate("s1", "c1", boosted_score=0.8)   # fusion rank 1
        c2 = _candidate("s2", "c2", boosted_score=0.3)   # fusion rank 2
        meta = {**_meta("c1", "Alpha Case"), **_meta("c2", "Beta Case")}
        result = await reranker.rerank("test query", [c1, c2], meta)
        # LLM score reversal should flip the ranking
        assert result[0].case_id == "c2"

    @pytest.mark.asyncio
    async def test_relevance_score_blended_correctly(self) -> None:
        """Final score = 0.70 × llm_score + 0.30 × norm_fusion."""
        llm_data = [
            {"id": "s1", "factual_similarity": 1.0, "principle_applicability": 1.0,
             "authority_strength": 1.0, "explanation": "perfect match"}
        ]
        anthropic = _make_anthropic(llm_data)
        reranker = LLMReranker(anthropic_client=anthropic)
        c1 = _candidate("s1", "c1", boosted_score=1.0)
        meta = _meta("c1", "Test Case")
        result = await reranker.rerank("test", [c1], meta)
        # llm=1.0, norm_fusion=1.0 → 0.70 × 1.0 + 0.30 × 1.0 = 1.0
        assert result[0].relevance_score == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_explanation_populated_from_llm(self) -> None:
        llm_data = [
            {"id": "s1", "factual_similarity": 0.8, "principle_applicability": 0.8,
             "authority_strength": 0.8, "explanation": "Leading case on jurisdiction"}
        ]
        anthropic = _make_anthropic(llm_data)
        reranker = LLMReranker(anthropic_client=anthropic)
        c1 = _candidate("s1", "c1", boosted_score=0.9)
        meta = _meta("c1", "Jurisdictional Case")
        result = await reranker.rerank("test", [c1], meta)
        assert "Leading case on jurisdiction" in result[0].relevance_explanation

    @pytest.mark.asyncio
    async def test_haiku_failure_graceful_fallback(self) -> None:
        anthropic = MagicMock()
        anthropic.messages = MagicMock()
        anthropic.messages.create = AsyncMock(side_effect=Exception("API error"))
        reranker = LLMReranker(anthropic_client=anthropic)
        c1 = _candidate("s1", "c1", boosted_score=0.9)
        meta = _meta("c1", "Test Case")
        result = await reranker.rerank("test", [c1], meta)
        # Should not raise; should return result with fusion-based score
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_top_n_respected(self) -> None:
        candidates = [_candidate(f"s{i}", f"c{i}", boosted_score=1.0 / (i + 1)) for i in range(10)]
        meta = {f"c{i}": {"case_name": f"Case {i}", "citation": None, "court": "NGSC",
                           "year": 2020, "times_cited": 0, "authority_score": 0}
                for i in range(10)}
        reranker = LLMReranker(anthropic_client=None)
        result = await reranker.rerank("test", candidates, meta, top_n=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_matched_segment_content_truncated(self) -> None:
        long_content = "x" * 1000
        c1 = _candidate("s1", "c1", boosted_score=0.9, content=long_content)
        reranker = LLMReranker(anthropic_client=None)
        meta = _meta("c1", "Test")
        result = await reranker.rerank("q", [c1], meta)
        assert len(result[0].matched_segment["content"]) <= 500

    @pytest.mark.asyncio
    async def test_metadata_populated_in_result(self) -> None:
        reranker = LLMReranker(anthropic_client=None)
        c1 = _candidate("s1", "c1", boosted_score=0.9, court="NGCA", year=2019)
        meta = {"c1": {"case_name": "Obi v. INEC", "citation": "(2023) CA 123",
                       "court": "NGCA", "year": 2023, "times_cited": 50, "authority_score": 50}}
        result = await reranker.rerank("q", [c1], meta)
        assert result[0].case_name == "Obi v. INEC"
        assert result[0].citation == "(2023) CA 123"
        assert result[0].times_cited == 50


class TestShortName:
    def test_short_name_unchanged(self) -> None:
        assert _short_name("Short v. Name") == "Short v. Name"

    def test_long_name_truncated_at_v(self) -> None:
        name = "Very Long Appellant Company Limited v. Very Long Respondent Corporation LLC"
        result = _short_name(name)
        assert "v." in result
        assert len(result) < len(name)

    def test_long_name_no_v_gets_ellipsis(self) -> None:
        name = "a" * 80
        result = _short_name(name, max_length=60)
        assert result.endswith("...")
