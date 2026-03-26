"""Unit tests for src/retrieval/engine.py"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval.engine import (
    RetrievalEngine,
    _collect_case_ids,
    _empty_response,
    _result_to_dict,
)
from retrieval.models import (
    CandidateResult,
    ExpandedQueries,
    ParsedQuery,
    SearchResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _parsed(
    motion_type: str | None = "motion_to_dismiss",
    concepts: list[str] | None = None,
    case_refs: list[str] | None = None,
    confidence: float = 0.7,
) -> ParsedQuery:
    return ParsedQuery(
        original="test query",
        motion_type=motion_type,
        detected_concepts=concepts or ["jurisdiction"],
        case_references=case_refs or [],
        confidence=confidence,
    )


def _expanded(
    n_dense: int = 1,
    n_sparse: int = 1,
) -> ExpandedQueries:
    return ExpandedQueries(
        dense_texts=["query"] * n_dense,
        dense_embeddings=[[0.1] * 1024] * n_dense,
        sparse_texts=["keyword string"] * n_sparse,
    )


def _candidate(case_id: str = "c1", segment_id: str = "s1") -> CandidateResult:
    return CandidateResult(
        case_id=case_id,
        segment_id=segment_id,
        segment_type="RATIO",
        content="leading case on jurisdiction",
        court="NGSC",
        year=2020,
        fusion_score=0.5,
        boosted_score=0.6,
    )


def _search_result(case_id: str = "c1") -> SearchResult:
    return SearchResult(
        case_id=case_id,
        case_name="Test v. State",
        case_name_short="Test v. State",
        citation="(2020) 1 NWLR 1",
        court="NGSC",
        year=2020,
        relevance_score=0.85,
        relevance_explanation="Leading case on jurisdiction",
        authority_score=42,
        times_cited=42,
        matched_segment={"type": "RATIO", "content": "ratio text"},
        verification_status="verified",
    )


def _make_engine(
    parsed: ParsedQuery | None = None,
    expanded: ExpandedQueries | None = None,
    candidates: list[CandidateResult] | None = None,
    ranked: list[SearchResult] | None = None,
    statutes: list[dict] | None = None,
    authority_rows: list[dict] | None = None,
    metadata_rows: list[dict] | None = None,
    cache: MagicMock | None = None,
) -> tuple[RetrievalEngine, MagicMock]:
    """Build a RetrievalEngine with all async dependencies mocked."""
    parsed = parsed or _parsed()
    expanded = expanded or _expanded()
    candidates = candidates if candidates is not None else [_candidate()]
    ranked = ranked if ranked is not None else [_search_result()]
    statutes = statutes if statutes is not None else [{"title": "CFRN 1999", "section": "1"}]

    parser = MagicMock()
    parser.parse = AsyncMock(return_value=parsed)

    expander = MagicMock()
    expander.expand = AsyncMock(return_value=expanded)

    dense = MagicMock()
    dense.search = AsyncMock(
        return_value=[
            {
                "case_id": "c1",
                "segment_id": "s1",
                "segment_type": "RATIO",
                "content": "text",
                "court": "NGSC",
                "year": 2020,
            }
        ]
    )

    sparse = MagicMock()
    sparse.search = AsyncMock(
        return_value=[
            {
                "case_id": "c1",
                "segment_id": "s1",
                "segment_type": "RATIO",
                "content": "text",
                "court": "NGSC",
                "year": 2020,
            }
        ]
    )

    exact = MagicMock()
    exact.search = AsyncMock(return_value=[])

    fusion = MagicMock()
    fusion.fuse = MagicMock(return_value=candidates)

    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=ranked)

    statute_retriever = MagicMock()
    statute_retriever.retrieve = AsyncMock(return_value=statutes)

    # Mock asyncpg connection
    conn = MagicMock()
    authority_rows = authority_rows or [{"case_id": "c1", "times_cited": 42, "authority_score": 42}]
    metadata_rows = metadata_rows or [
        {
            "case_id": "c1",
            "case_name": "Test v. State",
            "citation": "(2020) 1 NWLR 1",
            "court": "NGSC",
            "year": 2020,
            "status": "good_law",
        }
    ]
    conn.fetch = AsyncMock(side_effect=[authority_rows, metadata_rows])

    engine = RetrievalEngine(
        parser=parser,
        expander=expander,
        dense=dense,
        sparse=sparse,
        exact=exact,
        fusion=fusion,
        reranker=reranker,
        statutes=statute_retriever,
        cache=cache,
    )
    return engine, conn


def _mock_get_connection(conn: MagicMock):
    """Return a callable that creates a fresh async context manager on each call.

    Each parallel search task calls get_connection() independently, so each
    call must produce a new (non-exhausted) context manager instance.
    """

    @asynccontextmanager
    async def _ctx():
        yield conn

    return MagicMock(side_effect=lambda: _ctx())


# ── Happy-path tests ──────────────────────────────────────────────────────────


class TestRetrievalEngineSearch:
    @pytest.mark.asyncio
    async def test_response_shape(self) -> None:
        """Full pipeline returns required top-level keys."""
        engine, conn = _make_engine()
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        assert "cases" in result
        assert "statutes" in result
        assert "query_analysis" in result
        assert "search_metadata" in result

    @pytest.mark.asyncio
    async def test_cases_serialised_correctly(self) -> None:
        """Case dicts contain expected fields."""
        engine, conn = _make_engine()
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        assert len(result["cases"]) == 1
        case = result["cases"][0]
        for key in (
            "case_id",
            "case_name",
            "citation",
            "court",
            "year",
            "relevance_score",
            "relevance_explanation",
            "authority_score",
            "times_cited",
            "matched_segment",
            "verification_status",
        ):
            assert key in case, f"missing key: {key}"

    @pytest.mark.asyncio
    async def test_query_analysis_populated(self) -> None:
        engine, conn = _make_engine()
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        qa = result["query_analysis"]
        assert qa["detected_motion_type"] == "motion_to_dismiss"
        assert "jurisdiction" in qa["detected_concepts"]
        assert isinstance(qa["case_references_found"], list)

    @pytest.mark.asyncio
    async def test_search_metadata_has_timing(self) -> None:
        engine, conn = _make_engine()
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        meta = result["search_metadata"]
        assert meta["results_returned"] == 1
        assert isinstance(meta["total_time_ms"], int)
        assert meta["total_time_ms"] >= 0
        assert "stage_timings_ms" in meta
        assert "stage_counts" in meta
        assert "cache" in meta

    @pytest.mark.asyncio
    async def test_statutes_included(self) -> None:
        engine, conn = _make_engine(statutes=[{"title": "CFRN", "section": "36"}])
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query", include_statutes=True)

        assert len(result["statutes"]) == 1
        assert result["statutes"][0]["title"] == "CFRN"

    @pytest.mark.asyncio
    async def test_statutes_skipped_when_disabled(self) -> None:
        engine, conn = _make_engine()
        engine.statutes.retrieve.assert_not_called()  # sanity
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query", include_statutes=False)

        engine.statutes.retrieve.assert_not_called()
        assert result["statutes"] == []

    @pytest.mark.asyncio
    async def test_motion_type_override(self) -> None:
        """Caller-supplied motion_type overrides parser-detected value."""
        engine, conn = _make_engine(parsed=_parsed(motion_type="interlocutory_injunction"))
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query", motion_type="motion_to_dismiss")

        assert result["query_analysis"]["detected_motion_type"] == "motion_to_dismiss"

    @pytest.mark.asyncio
    async def test_max_results_respected(self) -> None:
        ranked = [_search_result(f"c{i}") for i in range(10)]
        engine, conn = _make_engine(ranked=ranked)
        conn.fetch = AsyncMock(
            side_effect=[
                [{"case_id": f"c{i}", "times_cited": 1, "authority_score": 1} for i in range(10)],
                [
                    {
                        "case_id": f"c{i}",
                        "case_name": f"Case {i}",
                        "citation": None,
                        "court": "NGSC",
                        "year": 2020,
                        "status": "good_law",
                    }
                    for i in range(10)
                ],
            ]
        )
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query", max_results=3)

        assert len(result["cases"]) == 3

    @pytest.mark.asyncio
    async def test_dense_search_called_per_embedding(self) -> None:
        expanded = _expanded(n_dense=2, n_sparse=1)
        engine, conn = _make_engine(expanded=expanded)
        with patch("db.get_connection", _mock_get_connection(conn)):
            await engine.search("test query")

        assert engine.dense.search.call_count == 2

    @pytest.mark.asyncio
    async def test_sparse_search_called_per_text(self) -> None:
        expanded = _expanded(n_dense=1, n_sparse=2)
        engine, conn = _make_engine(expanded=expanded)
        with patch("db.get_connection", _mock_get_connection(conn)):
            await engine.search("test query")

        assert engine.sparse.search.call_count == 2

    @pytest.mark.asyncio
    async def test_court_filter_passed_to_searchers(self) -> None:
        engine, conn = _make_engine()
        with patch("db.get_connection", _mock_get_connection(conn)):
            await engine.search("test query", court_filter=["NGSC", "NGCA"])

        call_kwargs = engine.dense.search.call_args[1]
        assert call_kwargs["court_codes"] == ["NGSC", "NGCA"]

    @pytest.mark.asyncio
    async def test_year_filter_passed_to_searchers(self) -> None:
        engine, conn = _make_engine()
        with patch("db.get_connection", _mock_get_connection(conn)):
            await engine.search("test query", year_min=2010, year_max=2020)

        call_kwargs = engine.dense.search.call_args[1]
        assert call_kwargs["year_min"] == 2010
        assert call_kwargs["year_max"] == 2020


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEmptyAndFallbackPaths:
    @pytest.mark.asyncio
    async def test_empty_response_when_no_candidates(self) -> None:
        """If fusion returns no candidates, engine returns an empty response."""
        engine, conn = _make_engine(candidates=[], ranked=[])
        conn.fetch = AsyncMock(return_value=[])  # authority scores fetch
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        assert result["cases"] == []
        assert result["statutes"] == []
        assert result["search_metadata"]["results_returned"] == 0

    @pytest.mark.asyncio
    async def test_statute_failure_returns_empty_list(self) -> None:
        engine, conn = _make_engine()
        engine.statutes.retrieve = AsyncMock(side_effect=Exception("statute DB error"))
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query", include_statutes=True)

        assert result["statutes"] == []

    @pytest.mark.asyncio
    async def test_authority_score_failure_returns_empty(self) -> None:
        """If authority score fetch fails, pipeline continues with empty scores."""
        engine, conn = _make_engine()
        # First fetch (authority) raises, second fetch (metadata) succeeds
        conn.fetch = AsyncMock(
            side_effect=[
                Exception("DB down"),
                [
                    {
                        "case_id": "c1",
                        "case_name": "Test",
                        "citation": None,
                        "court": "NGSC",
                        "year": 2020,
                        "status": "good_law",
                    }
                ],
            ]
        )
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        # Should complete without raising
        assert "cases" in result

    @pytest.mark.asyncio
    async def test_case_metadata_failure_graceful(self) -> None:
        engine, conn = _make_engine()
        conn.fetch = AsyncMock(
            side_effect=[
                [{"case_id": "c1", "times_cited": 5, "authority_score": 5}],
                Exception("metadata fail"),
            ]
        )
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        # Should complete without raising
        assert "cases" in result

    @pytest.mark.asyncio
    async def test_search_task_exception_logged_not_raised(self) -> None:
        """If one searcher raises, gather catches it and returns partial results."""
        engine, conn = _make_engine()
        engine.dense.search = AsyncMock(side_effect=Exception("vector index error"))
        with patch("db.get_connection", _mock_get_connection(conn)):
            # Should not raise
            result = await engine.search("test query")

        assert "cases" in result

    @pytest.mark.asyncio
    async def test_no_case_references_skips_exact_search(self) -> None:
        """If no citations in query, ExactSearcher is not called."""
        engine, conn = _make_engine(parsed=_parsed(case_refs=[]))
        with patch("db.get_connection", _mock_get_connection(conn)):
            await engine.search("test query")

        engine.exact.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_case_reference_triggers_exact_search(self) -> None:
        engine, conn = _make_engine(parsed=_parsed(case_refs=["Bakare v. State (2000) 3 NWLR 1"]))
        with patch("db.get_connection", _mock_get_connection(conn)):
            await engine.search("test query")

        engine.exact.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_parsed_query_cache_hit_skips_parser(self) -> None:
        cache = MagicMock()
        cache.get_parsed = AsyncMock(
            return_value={
                "original": "test query",
                "motion_type": "motion_to_dismiss",
                "detected_concepts": ["jurisdiction"],
                "case_references": [],
                "area_of_law": None,
                "confidence": 0.7,
                "step_back_query": None,
            }
        )
        cache.set_parsed = AsyncMock()
        cache.get_candidates = AsyncMock(return_value=None)
        cache.set_candidates = AsyncMock()
        cache.close = AsyncMock()

        engine, conn = _make_engine(cache=cache)
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        engine.parser.parse.assert_not_called()
        assert result["search_metadata"]["cache"]["parsed_query"] is True

    @pytest.mark.asyncio
    async def test_candidate_cache_hit_skips_searchers_and_fusion(self) -> None:
        cache = MagicMock()
        cache.get_parsed = AsyncMock(return_value=None)
        cache.set_parsed = AsyncMock()
        cache.get_candidates = AsyncMock(
            return_value=[
                {
                    "case_id": "c1",
                    "segment_id": "s1",
                    "segment_type": "RATIO",
                    "content": "leading case on jurisdiction",
                    "court": "NGSC",
                    "year": 2020,
                    "opinion_type": "LEAD",
                    "dense_rank": 1,
                    "sparse_rank": 1,
                    "fusion_score": 0.5,
                    "boosted_score": 0.6,
                }
            ]
        )
        cache.set_candidates = AsyncMock()
        cache.close = AsyncMock()

        engine, conn = _make_engine(cache=cache)
        with patch("db.get_connection", _mock_get_connection(conn)):
            result = await engine.search("test query")

        engine.dense.search.assert_not_called()
        engine.sparse.search.assert_not_called()
        engine.fusion.fuse.assert_not_called()
        assert result["search_metadata"]["cache"]["candidate_results"] is True


# ── Helper function tests ─────────────────────────────────────────────────────


class TestCollectCaseIds:
    def test_deduplicates_across_lists(self) -> None:
        lists = [
            [{"case_id": "c1"}, {"case_id": "c2"}],
            [{"case_id": "c2"}, {"case_id": "c3"}],
        ]
        result = _collect_case_ids(lists)
        assert result == ["c1", "c2", "c3"]

    def test_preserves_order_of_first_appearance(self) -> None:
        lists = [
            [{"case_id": "c3"}, {"case_id": "c1"}],
            [{"case_id": "c2"}],
        ]
        result = _collect_case_ids(lists)
        assert result == ["c3", "c1", "c2"]

    def test_empty_case_id_skipped(self) -> None:
        lists = [[{"case_id": ""}, {"case_id": "c1"}]]
        result = _collect_case_ids(lists)
        assert result == ["c1"]

    def test_missing_case_id_key_skipped(self) -> None:
        lists = [[{"segment_id": "s1"}, {"case_id": "c1"}]]
        result = _collect_case_ids(lists)
        assert result == ["c1"]

    def test_empty_input(self) -> None:
        assert _collect_case_ids([]) == []


class TestEmptyResponse:
    def test_structure(self) -> None:
        import time

        parsed = _parsed()
        result = _empty_response("test query", parsed, time.perf_counter())
        assert result["cases"] == []
        assert result["statutes"] == []
        assert result["search_metadata"]["results_returned"] == 0
        assert result["query_analysis"]["detected_motion_type"] == "motion_to_dismiss"


class TestResultToDict:
    def test_all_fields_present(self) -> None:
        sr = _search_result()
        d = _result_to_dict(sr)
        for key in (
            "case_id",
            "case_name",
            "case_name_short",
            "citation",
            "court",
            "year",
            "relevance_score",
            "relevance_explanation",
            "verification_status",
            "authority_score",
            "times_cited",
            "matched_segment",
        ):
            assert key in d, f"missing: {key}"

    def test_relevance_score_rounded(self) -> None:
        sr = _search_result()
        sr.relevance_score = 0.123456789
        d = _result_to_dict(sr)
        # Should be rounded to 4 decimal places
        assert d["relevance_score"] == pytest.approx(0.1235, abs=0.0001)
