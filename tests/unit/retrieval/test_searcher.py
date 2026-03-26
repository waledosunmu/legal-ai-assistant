"""Unit tests for src/retrieval/searcher.py"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from retrieval.searcher import DenseSearcher, ExactSearcher, SparseSearcher


def _fake_row(**kwargs) -> dict:
    return dict(**kwargs)


def _mock_conn(rows: list[dict]) -> MagicMock:
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    return conn


_SAMPLE_SEGMENT = {
    "segment_id": "seg-uuid-1",
    "case_id": "case-uuid-1",
    "segment_type": "RATIO",
    "content": "The court held that jurisdiction requires...",
    "retrieval_weight": 1.2,
    "opinion_type": "LEAD",
    "court": "NGSC",
    "year": 2018,
    "distance": 0.15,
}

_SAMPLE_SEGMENT_SPARSE = {**_SAMPLE_SEGMENT, "rank": 0.42}
del _SAMPLE_SEGMENT_SPARSE["distance"]  # type: ignore[misc]

_SAMPLE_CASE = {
    "case_id": "case-uuid-1",
    "case_name": "Madukolu v. Nkemdilim",
    "citation": "(1962) 2 SCNLR 341",
    "court": "NGSC",
    "year": 1962,
}


class TestDenseSearcher:
    searcher = DenseSearcher()

    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self) -> None:
        conn = _mock_conn([_SAMPLE_SEGMENT])
        result = await self.searcher.search(conn, [0.1] * 1024)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["segment_id"] == "seg-uuid-1"

    @pytest.mark.asyncio
    async def test_embedding_serialized_as_string(self) -> None:
        conn = _mock_conn([])
        await self.searcher.search(conn, [0.1, 0.2, 0.3])
        args = conn.fetch.call_args[0]
        # args[0] is SQL, args[1] is embedding string
        emb_arg = args[1]
        assert emb_arg.startswith("[")
        assert emb_arg.endswith("]")
        assert "0.1" in emb_arg

    @pytest.mark.asyncio
    async def test_limit_passed_as_second_arg(self) -> None:
        conn = _mock_conn([])
        await self.searcher.search(conn, [0.1] * 5, limit=30)
        args = conn.fetch.call_args[0]
        assert 30 in args

    @pytest.mark.asyncio
    async def test_returns_empty_on_db_error(self) -> None:
        conn = MagicMock()
        conn.fetch = AsyncMock(side_effect=Exception("DB down"))
        result = await self.searcher.search(conn, [0.1] * 5)
        assert result == []

    @pytest.mark.asyncio
    async def test_court_filter_applied(self) -> None:
        conn = _mock_conn([])
        await self.searcher.search(conn, [0.1] * 5, court_codes=["NGSC"])
        sql_called = conn.fetch.call_args[0][0]
        assert "ANY" in sql_called or "ARRAY" in sql_called

    @pytest.mark.asyncio
    async def test_year_filter_applied(self) -> None:
        conn = _mock_conn([])
        await self.searcher.search(conn, [0.1] * 5, year_min=2010, year_max=2020)
        sql_called = conn.fetch.call_args[0][0]
        assert "year >=" in sql_called
        assert "year <=" in sql_called


class TestSparseSearcher:
    searcher = SparseSearcher()

    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self) -> None:
        conn = _mock_conn([_SAMPLE_SEGMENT_SPARSE])
        result = await self.searcher.search(conn, "jurisdiction locus standi")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["segment_type"] == "RATIO"

    @pytest.mark.asyncio
    async def test_query_passed_as_first_arg(self) -> None:
        conn = _mock_conn([])
        await self.searcher.search(conn, "test legal query")
        args = conn.fetch.call_args[0]
        assert "test legal query" in args

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        conn = _mock_conn([])
        result = await self.searcher.search(conn, "  ")
        assert result == []
        conn.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_on_db_error(self) -> None:
        conn = MagicMock()
        conn.fetch = AsyncMock(side_effect=Exception("DB error"))
        result = await self.searcher.search(conn, "test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_limit_passed_correctly(self) -> None:
        conn = _mock_conn([])
        await self.searcher.search(conn, "test", limit=45)
        args = conn.fetch.call_args[0]
        assert 45 in args


class TestExactSearcher:
    searcher = ExactSearcher()

    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self) -> None:
        conn = _mock_conn([_SAMPLE_CASE])
        result = await self.searcher.search(conn, "(1962) 2 SCNLR 341")
        assert isinstance(result, list)
        assert result[0]["case_name"] == "Madukolu v. Nkemdilim"

    @pytest.mark.asyncio
    async def test_empty_reference_returns_empty(self) -> None:
        conn = _mock_conn([])
        result = await self.searcher.search(conn, "")
        assert result == []
        conn.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_reference_passed_as_args(self) -> None:
        conn = _mock_conn([])
        await self.searcher.search(conn, "(2000) 3 NWLR 1")
        args = conn.fetch.call_args[0]
        assert "(2000) 3 NWLR 1" in args

    @pytest.mark.asyncio
    async def test_returns_empty_on_db_error(self) -> None:
        conn = MagicMock()
        conn.fetch = AsyncMock(side_effect=Exception("DB error"))
        result = await self.searcher.search(conn, "Madukolu")
        assert result == []
