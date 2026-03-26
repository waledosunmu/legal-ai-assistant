"""Unit tests for generation.verification — citation verification engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from generation.models import VerificationStatus
from generation.verification import CitationVerifier

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_pool(fetchrow_returns=None, fetch_returns=None):
    """Create a mock asyncpg pool with configurable return values."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_returns)
    conn.fetch = AsyncMock(return_value=fetch_returns or [])

    pool = MagicMock()
    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


def _case_row(
    case_id="11111111-1111-1111-1111-111111111111",
    case_name="Madukolu v. Nkemdilim",
    citation="(1962) 2 SCNLR 341",
    status="active",
):
    return {"id": case_id, "case_name": case_name, "citation": citation, "status": status}


# ── No database ────────────────────────────────────────────────────────────────


class TestVerifierNoDb:
    @pytest.mark.asyncio
    async def test_returns_warning_without_db(self):
        verifier = CitationVerifier(db_pool=None)
        result = await verifier.verify_single({"name": "Test v. Case"})
        assert result["status"] == VerificationStatus.NOT_IN_CORPUS.value
        assert "not available" in result.get("warning", "")

    @pytest.mark.asyncio
    async def test_batch_without_db(self):
        verifier = CitationVerifier(db_pool=None)
        results = await verifier.verify_batch(
            [
                {"name": "A v. B"},
                {"name": "C v. D"},
            ]
        )
        assert len(results) == 2
        assert all(r["status"] == VerificationStatus.NOT_IN_CORPUS.value for r in results)


# ── Existence check ────────────────────────────────────────────────────────────


class TestExistenceCheck:
    @pytest.mark.asyncio
    async def test_case_found_by_id(self):
        pool, conn = _mock_pool()
        case = _case_row()
        # First fetchrow (by case_id) returns the case
        conn.fetchrow = AsyncMock(return_value=case)
        conn.fetch = AsyncMock(return_value=[])

        verifier = CitationVerifier(db_pool=pool)
        result = await verifier.verify_single(
            {
                "case_id": "11111111-1111-1111-1111-111111111111",
                "name": "Madukolu v. Nkemdilim",
            }
        )
        assert result["checks"]["existence"] is True
        assert result["case_id"] == case["id"]

    @pytest.mark.asyncio
    async def test_case_not_found(self):
        pool, conn = _mock_pool()
        conn.fetchrow = AsyncMock(return_value=None)

        verifier = CitationVerifier(db_pool=pool)
        result = await verifier.verify_single({"name": "Nonexistent v. Case"})
        assert result["checks"]["existence"] is False
        assert result["status"] == VerificationStatus.NOT_IN_CORPUS.value
        assert "NOT found" in result.get("warning", "")


# ── Overruled detection ────────────────────────────────────────────────────────


class TestOverruledDetection:
    @pytest.mark.asyncio
    async def test_overruled_case_flagged(self):
        pool, conn = _mock_pool()
        overruled_case = _case_row(status="overruled")
        conn.fetchrow = AsyncMock(return_value=overruled_case)
        conn.fetch = AsyncMock(return_value=[])

        verifier = CitationVerifier(db_pool=pool)
        result = await verifier.verify_single(
            {
                "case_id": "11111111-1111-1111-1111-111111111111",
                "name": "Old v. Case",
            }
        )
        assert result["status"] == VerificationStatus.OVERRULED.value
        assert "OVERRULED" in result.get("warning", "")


# ── Format check ───────────────────────────────────────────────────────────────


class TestFormatCheck:
    def test_matching_format(self):
        assert CitationVerifier._check_format("(1962) 2 SCNLR 341", "(1962) 2 SCNLR 341")

    def test_format_subset_match(self):
        assert CitationVerifier._check_format("2 SCNLR 341", "(1962) 2 SCNLR 341")

    def test_mismatched_format(self):
        assert not CitationVerifier._check_format("(2020) 15 NWLR 1", "(1962) 2 SCNLR 341")

    def test_empty_db_format(self):
        assert not CitationVerifier._check_format("(2020) 1 NWLR 1", "")


# ── Holding attribution ───────────────────────────────────────────────────────


class TestHoldingAttribution:
    @pytest.mark.asyncio
    async def test_high_overlap_verified(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "content": "The court must have jurisdiction over the subject matter and the parties before it can validly adjudicate"
                },
            ]
        )

        verifier = CitationVerifier(db_pool=pool)
        result = await verifier._verify_holding_attribution(
            conn,
            "11111111-1111-1111-1111-111111111111",
            "The court must have jurisdiction over the subject matter",
        )
        assert result["verified"] is True
        assert result["similarity"] > 0.4

    @pytest.mark.asyncio
    async def test_low_overlap_not_verified(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(
            return_value=[
                {"content": "Damages for breach of contract shall be assessed by evidence"},
            ]
        )

        verifier = CitationVerifier(db_pool=pool)
        result = await verifier._verify_holding_attribution(
            conn,
            "11111111-1111-1111-1111-111111111111",
            "The court must have jurisdiction over the subject matter",
        )
        assert result["verified"] is False

    @pytest.mark.asyncio
    async def test_no_holdings_in_corpus(self):
        pool, conn = _mock_pool()
        conn.fetch = AsyncMock(return_value=[])

        verifier = CitationVerifier(db_pool=pool)
        result = await verifier._verify_holding_attribution(
            conn,
            "11111111-1111-1111-1111-111111111111",
            "Some principle",
        )
        assert result["verified"] is False
        assert "No holdings" in result.get("reason", "")


# ── Full verification flow ─────────────────────────────────────────────────────


class TestFullVerificationFlow:
    @pytest.mark.asyncio
    async def test_fully_verified_case(self):
        """Case exists, active, principle matches → FULLY_VERIFIED."""
        pool, conn = _mock_pool()
        active_case = _case_row(status="active")
        conn.fetchrow = AsyncMock(return_value=active_case)
        # Holdings that match the principle
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "content": "jurisdiction is fundamental to the competence of a court to entertain any matter"
                },
            ]
        )

        verifier = CitationVerifier(db_pool=pool)
        result = await verifier.verify_single(
            {
                "case_id": "11111111-1111-1111-1111-111111111111",
                "name": "Madukolu v. Nkemdilim",
                "citation": "(1962) 2 SCNLR 341",
                "principle_cited": "jurisdiction is fundamental to the competence of a court",
            }
        )
        assert result["verified"] is True
        assert result["status"] == VerificationStatus.FULLY_VERIFIED.value

    @pytest.mark.asyncio
    async def test_case_verified_principle_unconfirmed(self):
        """Case exists, active, but principle doesn't match → CASE_VERIFIED."""
        pool, conn = _mock_pool()
        active_case = _case_row(status="active")
        conn.fetchrow = AsyncMock(return_value=active_case)
        # Holdings that DON'T match the principle
        conn.fetch = AsyncMock(
            return_value=[
                {"content": "The limitation period for contract claims is six years"},
            ]
        )

        verifier = CitationVerifier(db_pool=pool)
        result = await verifier.verify_single(
            {
                "case_id": "11111111-1111-1111-1111-111111111111",
                "name": "Madukolu v. Nkemdilim",
                "citation": "(1962) 2 SCNLR 341",
                "principle_cited": "jurisdiction is a threshold issue in every court proceeding",
            }
        )
        assert result["verified"] is False
        assert result["status"] == VerificationStatus.CASE_VERIFIED.value
        assert "may not match" in result.get("warning", "")

    @pytest.mark.asyncio
    async def test_batch_mixed_results(self):
        pool, conn = _mock_pool()

        call_count = 0

        async def _mock_fetchrow(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:  # First citation found (1 fetchrow for case_id lookup)
                return _case_row()
            return None  # Second citation not found

        conn.fetchrow = _mock_fetchrow
        conn.fetch = AsyncMock(return_value=[])

        verifier = CitationVerifier(db_pool=pool)
        results = await verifier.verify_batch(
            [
                {"case_id": "11111111-1111-1111-1111-111111111111", "name": "Found v. Case"},
                {"name": "Missing v. Case"},
            ]
        )
        assert len(results) == 2
        found_statuses = {r["status"] for r in results}
        assert VerificationStatus.NOT_IN_CORPUS.value in found_statuses
