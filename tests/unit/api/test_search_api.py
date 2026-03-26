"""Unit tests for the FastAPI search endpoint (api/routers/search.py + schemas.py).

Uses httpx.AsyncClient + ASGITransport to exercise the full HTTP layer without
a real DB or AI service — the RetrievalEngine is monkey-patched before the app
is imported.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ── Shared fixture data ────────────────────────────────────────────────────────


def _engine_response(n: int = 1) -> dict:
    """Minimal valid engine.search() return dict."""
    return {
        "cases": [
            {
                "case_id": f"c{i}",
                "case_name": f"Case {i} v. State",
                "case_name_short": f"Case {i} v. State",
                "citation": f"(2020) {i} NWLR 1",
                "court": "NGSC",
                "year": 2020,
                "relevance_score": 0.9,
                "relevance_explanation": "Leading case",
                "authority_score": 10,
                "times_cited": 10,
                "matched_segment": {"type": "RATIO", "content": "ratio text"},
                "verification_status": "verified",
            }
            for i in range(n)
        ],
        "statutes": [{"title": "CFRN 1999", "section": "36"}],
        "query_analysis": {
            "detected_motion_type": "motion_to_dismiss",
            "detected_concepts": ["jurisdiction"],
            "case_references_found": [],
        },
        "search_metadata": {"total_time_ms": 250, "results_returned": n},
    }


def _make_mock_engine(response: dict | None = None, raises: Exception | None = None):
    engine = MagicMock()
    if raises:
        engine.search = AsyncMock(side_effect=raises)
    else:
        engine.search = AsyncMock(return_value=response or _engine_response())
    return engine


@pytest.fixture
def mock_engine():
    engine = _make_mock_engine()
    with patch("api.app._engine", engine):
        yield engine


@pytest.fixture
async def client(mock_engine):
    """AsyncClient wired to the FastAPI app with engine mocked."""
    from api.app import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── Happy-path tests ──────────────────────────────────────────────────────────


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_200_on_valid_request(self, client, mock_engine) -> None:
        resp = await client.post(
            "/api/v1/search",
            json={"query": "grounds for dismissal for want of jurisdiction in Nigeria"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_required_top_level_keys(self, client, mock_engine) -> None:
        resp = await client.post(
            "/api/v1/search",
            json={"query": "grounds for dismissal for want of jurisdiction in Nigeria"},
        )
        body = resp.json()
        for key in ("cases", "statutes", "query_analysis", "search_metadata"):
            assert key in body

    @pytest.mark.asyncio
    async def test_case_fields_present(self, client, mock_engine) -> None:
        resp = await client.post(
            "/api/v1/search",
            json={"query": "grounds for dismissal for want of jurisdiction in Nigeria"},
        )
        case = resp.json()["cases"][0]
        for field in (
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
            assert field in case

    @pytest.mark.asyncio
    async def test_engine_called_with_query(self, client, mock_engine) -> None:
        await client.post(
            "/api/v1/search",
            json={"query": "interlocutory injunction balance of convenience Nigeria"},
        )
        call_kwargs = mock_engine.search.call_args[1]
        assert "interlocutory injunction" in call_kwargs["query"]

    @pytest.mark.asyncio
    async def test_motion_type_forwarded_to_engine(self, client, mock_engine) -> None:
        await client.post(
            "/api/v1/search",
            json={
                "query": "grounds for dismissal for want of jurisdiction in Nigeria",
                "motion_type": "motion_to_dismiss",
            },
        )
        assert mock_engine.search.call_args[1]["motion_type"] == "motion_to_dismiss"

    @pytest.mark.asyncio
    async def test_court_filter_forwarded(self, client, mock_engine) -> None:
        await client.post(
            "/api/v1/search",
            json={
                "query": "grounds for dismissal for want of jurisdiction in Nigeria",
                "court_filter": ["NGSC"],
            },
        )
        assert mock_engine.search.call_args[1]["court_filter"] == ["NGSC"]

    @pytest.mark.asyncio
    async def test_max_results_forwarded(self, client, mock_engine) -> None:
        await client.post(
            "/api/v1/search",
            json={
                "query": "grounds for dismissal for want of jurisdiction in Nigeria",
                "max_results": 5,
            },
        )
        assert mock_engine.search.call_args[1]["max_results"] == 5

    @pytest.mark.asyncio
    async def test_include_statutes_false(self, client, mock_engine) -> None:
        await client.post(
            "/api/v1/search",
            json={
                "query": "grounds for dismissal for want of jurisdiction in Nigeria",
                "include_statutes": False,
            },
        )
        assert mock_engine.search.call_args[1]["include_statutes"] is False

    @pytest.mark.asyncio
    async def test_year_range_forwarded(self, client, mock_engine) -> None:
        await client.post(
            "/api/v1/search",
            json={
                "query": "grounds for dismissal for want of jurisdiction in Nigeria",
                "year_min": 2000,
                "year_max": 2020,
            },
        )
        kwargs = mock_engine.search.call_args[1]
        assert kwargs["year_min"] == 2000
        assert kwargs["year_max"] == 2020


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    @pytest.mark.asyncio
    async def test_422_when_query_too_short(self, client, mock_engine) -> None:
        resp = await client.post("/api/v1/search", json={"query": "short"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_422_when_max_results_out_of_range(self, client, mock_engine) -> None:
        resp = await client.post(
            "/api/v1/search",
            json={
                "query": "grounds for dismissal for want of jurisdiction in Nigeria",
                "max_results": 100,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_422_when_invalid_motion_type(self, client, mock_engine) -> None:
        resp = await client.post(
            "/api/v1/search",
            json={
                "query": "grounds for dismissal for want of jurisdiction in Nigeria",
                "motion_type": "invalid_motion",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_422_on_missing_query(self, client, mock_engine) -> None:
        resp = await client.post("/api/v1/search", json={})
        assert resp.status_code == 422


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_500_when_engine_raises(self, mock_engine) -> None:
        engine = _make_mock_engine(raises=RuntimeError("DB connection lost"))
        with patch("api.app._engine", engine):
            from api.app import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/v1/search",
                    json={"query": "grounds for dismissal for want of jurisdiction in Nigeria"},
                )
        assert resp.status_code == 500
        assert "Search failed" in resp.json()["detail"]


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client, mock_engine) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
