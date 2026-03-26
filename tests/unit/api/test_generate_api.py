"""Unit tests for POST /api/v1/generate endpoint.

Mirrors the pattern in test_search_api.py — uses httpx.AsyncClient + ASGITransport
with the generation pipeline monkey-patched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from generation.models import (
    ArgumentSection,
    GenerationResult,
    MotionPaper,
    SupportingAffidavit,
    WrittenAddress,
)


# ── Fixture data ───────────────────────────────────────────────────────────────


def _valid_payload() -> dict:
    """Minimal valid GenerateRequest payload."""
    return {
        "case_facts": (
            "The defendant ABC Ltd breached a supply contract "
            "with XYZ Ltd by failing to deliver goods worth N50M. "
            "This is a detailed description of the facts of the case."
        ),
        "motion_type": "motion_to_dismiss",
        "court_name": "Federal High Court",
        "division": "Lagos",
        "location": "Lagos",
        "suit_number": "FHC/L/CS/1/2026",
        "applicant_name": "ABC Ltd",
        "respondent_name": "XYZ Ltd",
        "position": "applicant",
        "relief_sought": "An order dismissing this suit for want of jurisdiction",
        "selected_cases": [
            {
                "case_id": "c1",
                "case_name": "Madukolu v. Nkemdilim",
                "citation": "(1962) 2 SCNLR 341",
                "court": "NGSC",
                "year": 1962,
            }
        ],
        "statutes": [
            {"section": "Section 251(1)(d)", "act": "CFRN 1999"},
        ],
        "counsel_name": "A. Barrister Esq.",
    }


def _pipeline_result() -> GenerationResult:
    """Minimal valid GenerationResult for mocking."""
    arg = ArgumentSection(
        issue_number=1,
        issue_text="Whether the court has jurisdiction?",
        argument_text="Jurisdiction is fundamental. Madukolu v. Nkemdilim (1962) 2 SCNLR 341.",
        cases_cited=[
            {"name": "Madukolu v. Nkemdilim", "citation": "(1962) 2 SCNLR 341", "verified": True, "status": "fully_verified"}
        ],
    )
    return GenerationResult(
        motion_paper=MotionPaper(
            court_name="IN THE FEDERAL HIGH COURT",
            division="Lagos",
            location="Lagos",
            suit_number="FHC/L/CS/1/2026",
            applicant_name="ABC Ltd",
            applicant_description="Applicant",
            respondent_name="XYZ Ltd",
            respondent_description="Respondent",
            motion_type="Motion to Dismiss",
            brought_pursuant_to=["Section 251(1)(d) CFRN 1999"],
            prayers=["AN ORDER dismissing the suit"],
            grounds=["The court lacks jurisdiction"],
            date="19th March, 2026",
            counsel_name="A. Barrister Esq.",
            counsel_firm="Law Chambers",
            counsel_address="1 Law Lane, Lagos",
        ),
        affidavit=SupportingAffidavit(
            court_header="IN THE FEDERAL HIGH COURT",
            suit_number="FHC/L/CS/1/2026",
            parties="ABC Ltd v. XYZ Ltd",
            deponent_name="ABC Ltd",
            deponent_description="Applicant",
            deponent_capacity="Director",
            paragraphs=["I am the Director of the Applicant."],
        ),
        written_address=WrittenAddress(
            court_header="IN THE FEDERAL HIGH COURT",
            suit_number="FHC/L/CS/1/2026",
            parties="ABC Ltd v. XYZ Ltd",
            title="Written Address in Support of Motion To Dismiss",
            introduction="This Written Address is filed in support.",
            issues_for_determination=["Whether the court has jurisdiction?"],
            arguments=[arg],
            conclusion="We urge the Court to grant the application.",
            counsel_signature="A. Barrister Esq.\nLaw Chambers\n1 Law Lane, Lagos",
        ),
        citation_report=[
            {
                "name": "Madukolu v. Nkemdilim",
                "citation": "(1962) 2 SCNLR 341",
                "case_id": "c1",
                "status": "fully_verified",
                "verified": True,
                "checks": {},
            }
        ],
        strength_report=[
            {
                "issue_number": 1,
                "issue_text": "Whether the court has jurisdiction?",
                "legal_soundness": 8,
                "factual_applicability": 7,
                "authority_strength": 9,
                "vulnerability": 6,
                "overall_score": 7.5,
                "weaknesses": [],
            }
        ],
        counter_arguments=[
            {
                "issue_number": 1,
                "counter_argument": "CA1",
                "potential_authority": "",
                "suggested_rebuttal": "R1",
            }
        ],
        readiness_report={"ready": True, "warnings": [], "cases_count": 1, "statutes_count": 1},
        audit_log=[],
    )


def _make_mock_pipeline(
    result: GenerationResult | None = None,
    raises: Exception | None = None,
):
    pipeline = MagicMock()
    if raises:
        pipeline.generate = AsyncMock(side_effect=raises)
    else:
        pipeline.generate = AsyncMock(return_value=result or _pipeline_result())
    return pipeline


@pytest.fixture
def mock_pipeline():
    pipeline = _make_mock_pipeline()
    with patch("api.app._pipeline", pipeline):
        yield pipeline


@pytest.fixture
async def client(mock_pipeline):
    from api.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── Happy-path tests ──────────────────────────────────────────────────────────


class TestGenerateEndpoint:
    @pytest.mark.asyncio
    async def test_200_on_valid_request(self, client, mock_pipeline) -> None:
        resp = await client.post("/api/v1/generate", json=_valid_payload())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_top_level_keys(self, client, mock_pipeline) -> None:
        resp = await client.post("/api/v1/generate", json=_valid_payload())
        body = resp.json()
        for key in (
            "motion_paper", "affidavit", "written_address",
            "citation_report", "citation_summary",
            "strength_report", "counter_arguments", "readiness",
        ):
            assert key in body, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_motion_paper_fields(self, client, mock_pipeline) -> None:
        resp = await client.post("/api/v1/generate", json=_valid_payload())
        mp = resp.json()["motion_paper"]
        assert mp["court_name"] == "IN THE FEDERAL HIGH COURT"
        assert mp["applicant_name"] == "ABC Ltd"
        assert "rendered_text" in mp
        assert len(mp["prayers"]) >= 1

    @pytest.mark.asyncio
    async def test_affidavit_fields(self, client, mock_pipeline) -> None:
        resp = await client.post("/api/v1/generate", json=_valid_payload())
        aff = resp.json()["affidavit"]
        assert aff["deponent_name"] == "ABC Ltd"
        assert len(aff["paragraphs"]) >= 1
        assert "rendered_text" in aff

    @pytest.mark.asyncio
    async def test_written_address_fields(self, client, mock_pipeline) -> None:
        resp = await client.post("/api/v1/generate", json=_valid_payload())
        wa = resp.json()["written_address"]
        assert wa["title"] == "Written Address in Support of Motion To Dismiss"
        assert len(wa["arguments"]) >= 1
        assert "rendered_text" in wa

    @pytest.mark.asyncio
    async def test_citation_report(self, client, mock_pipeline) -> None:
        resp = await client.post("/api/v1/generate", json=_valid_payload())
        cr = resp.json()["citation_report"]
        assert len(cr) >= 1
        assert cr[0]["verified"] is True

    @pytest.mark.asyncio
    async def test_readiness_report(self, client, mock_pipeline) -> None:
        resp = await client.post("/api/v1/generate", json=_valid_payload())
        r = resp.json()["readiness"]
        assert r["ready"] is True

    @pytest.mark.asyncio
    async def test_pipeline_called_with_correct_data(self, client, mock_pipeline) -> None:
        await client.post("/api/v1/generate", json=_valid_payload())
        call_args = mock_pipeline.generate.call_args[0]
        gen_req = call_args[0]
        assert gen_req.motion_type == "motion_to_dismiss"
        assert gen_req.applicant_name == "ABC Ltd"
        assert len(gen_req.selected_cases) == 1


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    @pytest.mark.asyncio
    async def test_422_when_case_facts_too_short(self, client, mock_pipeline) -> None:
        payload = _valid_payload()
        payload["case_facts"] = "Too short"
        resp = await client.post("/api/v1/generate", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_422_when_missing_required_field(self, client, mock_pipeline) -> None:
        payload = _valid_payload()
        del payload["suit_number"]
        resp = await client.post("/api/v1/generate", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_422_when_invalid_motion_type(self, client, mock_pipeline) -> None:
        payload = _valid_payload()
        payload["motion_type"] = "invalid_motion"
        resp = await client.post("/api/v1/generate", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_422_on_empty_body(self, client, mock_pipeline) -> None:
        resp = await client.post("/api/v1/generate", json={})
        assert resp.status_code == 422


# ── Error handling ─────────────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_500_when_pipeline_raises(self, mock_pipeline) -> None:
        failing = _make_mock_pipeline(raises=RuntimeError("LLM unavailable"))
        with patch("api.app._pipeline", failing):
            from api.app import app
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post("/api/v1/generate", json=_valid_payload())
        assert resp.status_code == 500
        assert "Generation failed" in resp.json()["detail"]
