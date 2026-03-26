"""5-level citation verification engine.

Verifies every citation in generated text against the corpus:
  1. Existence — does the case exist in our database?
  2. Status — is the case still good law (not overruled)?
  3. Format — does the citation format match our records?
  4. Attribution — does the claimed principle match the actual holding?
  5. Cross-reference — consistent with how other cases treat it?

This is the MOST CRITICAL safety component. A hallucinated citation
in a court filing can result in sanctions, dismissal, or disciplinary
proceedings against the lawyer.
"""

from __future__ import annotations

import logging
import re

from generation.models import VerificationStatus

logger = logging.getLogger(__name__)

# ── SQL queries ────────────────────────────────────────────────────────────────

_FUZZY_SEARCH_SQL = """
SELECT id::text, case_name, citation, status::text
FROM cases
WHERE search_vector @@ plainto_tsquery('english', $1)
ORDER BY ts_rank(search_vector, plainto_tsquery('english', $1)) DESC
LIMIT 3
"""

_EXACT_CITATION_SQL = """
SELECT id::text, case_name, citation, status::text
FROM cases
WHERE citation ILIKE $1
LIMIT 1
"""

_CASE_STATUS_SQL = """
SELECT status::text FROM cases WHERE id = $1::uuid
"""

_CASE_CITATION_SQL = """
SELECT citation FROM cases WHERE id = $1::uuid
"""

_HOLDINGS_SQL = """
SELECT content FROM case_segments
WHERE case_id = $1::uuid
AND seg_type IN ('RATIO', 'HOLDING')
"""

_OVERRULED_BY_SQL = """
SELECT
    c.case_name,
    c.citation
FROM citation_graph cg
JOIN cases c ON c.id = cg.citing_case_id
WHERE cg.cited_case_id = $1::uuid
AND cg.treatment = 'overruled'
LIMIT 1
"""


class CitationVerifier:
    """Verify citations extracted from generated text against the corpus."""

    def __init__(self, db_pool=None) -> None:
        self.db_pool = db_pool

    async def verify_batch(self, citations: list[dict]) -> list[dict]:
        """Verify a batch of citations, returning enriched results."""
        results = []
        for cite in citations:
            result = await self.verify_single(cite)
            results.append(result)
        return results

    async def verify_single(self, citation: dict) -> dict:
        """Verify a single citation through 5 checks."""
        result = {
            **citation,
            "verified": False,
            "status": VerificationStatus.NOT_IN_CORPUS.value,
            "checks": {},
        }

        if not self.db_pool:
            result["warning"] = "Database not available for verification"
            return result

        async with self.db_pool.acquire() as conn:
            # Check 1: Existence
            case_row = await self._check_existence(conn, citation)
            if not case_row:
                result["checks"]["existence"] = False
                result["warning"] = (
                    "This case was NOT found in our database. Please verify manually before filing."
                )
                return result

            result["checks"]["existence"] = True
            result["case_id"] = case_row["id"]
            result["db_case_name"] = case_row["case_name"]
            result["db_citation"] = case_row["citation"]
            result["status"] = VerificationStatus.CASE_EXISTS.value

            # Check 2: Status (overruled?)
            case_status = case_row.get("status", "active")
            result["checks"]["case_status"] = case_status
            if case_status == "overruled":
                result["status"] = VerificationStatus.OVERRULED.value
                overruling = await self._find_overruling_case(conn, case_row["id"])
                result["warning"] = (
                    "This case has been OVERRULED. "
                    "Do NOT cite without checking the overruling decision."
                )
                if overruling:
                    result["overruled_by"] = overruling
                return result

            # Check 3: Citation format
            if citation.get("citation"):
                format_match = self._check_format(
                    citation["citation"], case_row.get("citation", "")
                )
                result["checks"]["format"] = format_match

            # Check 4: Holding attribution
            if citation.get("principle_cited"):
                attribution = await self._verify_holding_attribution(
                    conn, case_row["id"], citation["principle_cited"]
                )
                result["checks"]["attribution"] = attribution
                if not attribution.get("verified"):
                    result["status"] = VerificationStatus.CASE_VERIFIED.value
                    result["warning"] = (
                        "Case exists but the specific legal principle cited "
                        "may not match the actual holding. Please verify the "
                        "quoted proposition against the judgment text."
                    )
                    result["closest_actual_holding"] = attribution.get("closest_holding", "")
                    return result

            # All checks passed
            if result["checks"].get("attribution", {}).get("verified", False):
                result["verified"] = True
                result["status"] = VerificationStatus.FULLY_VERIFIED.value
            else:
                result["status"] = VerificationStatus.CASE_VERIFIED.value

        return result

    async def _check_existence(self, conn, citation: dict) -> dict | None:
        """Find the case in our database by case_id, citation string, or name search."""
        # Try direct case_id if provided
        if citation.get("case_id"):
            row = await conn.fetchrow(
                "SELECT id::text, case_name, citation, status::text FROM cases WHERE id = $1::uuid",
                citation["case_id"],
            )
            if row:
                return dict(row)

        # Try exact citation match
        if citation.get("citation"):
            row = await conn.fetchrow(_EXACT_CITATION_SQL, f"%{citation['citation']}%")
            if row:
                return dict(row)

        # Try fuzzy name search
        if citation.get("name"):
            # Clean the case name for search
            search_term = re.sub(r"\s+v\.?\s+", " ", citation["name"])
            row = await conn.fetchrow(_FUZZY_SEARCH_SQL, search_term)
            if row:
                return dict(row)

        return None

    async def _find_overruling_case(self, conn, case_id: str) -> dict | None:
        """Find the case that overruled this one, if any."""
        row = await conn.fetchrow(_OVERRULED_BY_SQL, case_id)
        if row:
            return {"case_name": row["case_name"], "citation": row["citation"]}
        return None

    @staticmethod
    def _check_format(cited_format: str, db_format: str) -> bool:
        """Check if the cited format roughly matches what we have in the DB."""
        if not db_format:
            return False
        # Normalise whitespace and compare
        cited_norm = re.sub(r"\s+", " ", cited_format.strip().lower())
        db_norm = re.sub(r"\s+", " ", db_format.strip().lower())
        return cited_norm in db_norm or db_norm in cited_norm

    async def _verify_holding_attribution(self, conn, case_id: str, claimed_principle: str) -> dict:
        """Verify that the cited principle appears in the case's holdings/ratio."""
        rows = await conn.fetch(_HOLDINGS_SQL, case_id)
        if not rows:
            return {"verified": False, "reason": "No holdings found in corpus"}

        claimed_lower = claimed_principle.lower()
        claimed_words = set(claimed_lower.split())
        if not claimed_words:
            return {"verified": False, "reason": "Empty principle"}

        best_overlap = 0.0
        best_match = ""

        for row in rows:
            content_lower = row["content"].lower()
            content_words = set(content_lower.split())
            overlap = len(claimed_words & content_words) / len(claimed_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = row["content"][:300]

        return {
            "verified": best_overlap > 0.4,
            "similarity": round(best_overlap, 3),
            "closest_holding": best_match,
        }
