"""Data models for the Phase 2 generation engine.

All generation components share these types — document structures,
citation references, verification statuses, audit events, and the
request/result containers for the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

# ── Verification ───────────────────────────────────────────────────────────────


class VerificationStatus(StrEnum):
    """Citation verification level (§5.4 of the PRD)."""

    FULLY_VERIFIED = "fully_verified"  # ✅ Case exists, active, principle confirmed
    CASE_VERIFIED = "case_verified"  # 🟡 Case exists, active, principle unconfirmed
    CASE_EXISTS = "case_exists"  # ⚠️ Case in corpus, citation format unconfirmed
    NOT_IN_CORPUS = "not_in_corpus"  # ❌ Case not found
    OVERRULED = "overruled"  # 🔄 Case exists but overruled


# ── Document structures ───────────────────────────────────────────────────────


@dataclass
class ArgumentSection:
    """A single argument section within the Written Address."""

    issue_number: int
    issue_text: str
    argument_text: str
    cases_cited: list[dict] = field(default_factory=list)
    statutes_cited: list[dict] = field(default_factory=list)


@dataclass
class MotionPaper:
    """The formal motion paper filed with the court."""

    court_name: str
    division: str
    location: str
    suit_number: str
    applicant_name: str
    applicant_description: str
    respondent_name: str
    respondent_description: str
    motion_type: str
    brought_pursuant_to: list[str]
    prayers: list[str]
    grounds: list[str]
    date: str
    counsel_name: str
    counsel_firm: str
    counsel_address: str


@dataclass
class SupportingAffidavit:
    """The affidavit in support of the motion."""

    court_header: str
    suit_number: str
    parties: str
    deponent_name: str
    deponent_description: str
    deponent_capacity: str
    paragraphs: list[str]
    exhibits: list[dict] = field(default_factory=list)
    jurat: str = ""


@dataclass
class WrittenAddress:
    """The legal argument — the core document we generate."""

    court_header: str
    suit_number: str
    parties: str
    title: str
    introduction: str
    issues_for_determination: list[str]
    arguments: list[ArgumentSection]
    conclusion: str
    counsel_signature: str


# ── Pipeline request / result ──────────────────────────────────────────────────


@dataclass
class GenerationRequest:
    """Input to MotionGenerationPipeline.generate()."""

    case_facts: str
    motion_type: str
    court_name: str
    division: str
    location: str
    suit_number: str
    applicant_name: str
    applicant_description: str
    respondent_name: str
    respondent_description: str
    position: str  # "applicant" or "respondent"
    relief_sought: str
    selected_cases: list[dict]  # cases from retrieval engine
    statutes: list[dict] = field(default_factory=list)
    counsel_name: str = ""
    counsel_firm: str = ""
    counsel_address: str = ""
    date: str = ""
    deponent_name: str = ""
    deponent_description: str = ""
    deponent_capacity: str = ""
    draft_id: str = ""


@dataclass
class GenerationResult:
    """Complete generated motion documents with verification."""

    motion_paper: MotionPaper
    affidavit: SupportingAffidavit
    written_address: WrittenAddress
    citation_report: list[dict]
    strength_report: list[dict]
    counter_arguments: list[dict]
    readiness_report: dict
    audit_log: list[dict]

    @property
    def unverified_citations(self) -> list[dict]:
        return [
            c
            for c in self.citation_report
            if c.get("status") == VerificationStatus.NOT_IN_CORPUS.value
        ]

    @property
    def overruled_citations(self) -> list[dict]:
        return [
            c for c in self.citation_report if c.get("status") == VerificationStatus.OVERRULED.value
        ]

    @property
    def misgrounded_citations(self) -> list[dict]:
        """Citations where the case exists but the principle may not match."""
        return [
            c
            for c in self.citation_report
            if c.get("status") == VerificationStatus.CASE_VERIFIED.value
            and c.get("checks", {}).get("attribution", {}).get("verified") is False
        ]

    @property
    def weak_arguments(self) -> list[dict]:
        """Arguments with overall strength score below 5."""
        return [
            a for a in self.strength_report if a.get("overall_score") and a["overall_score"] < 5.0
        ]

    @property
    def citation_summary(self) -> dict:
        """Summary of citation verification for the UI."""
        total = len(self.citation_report)
        fully_verified = sum(
            1
            for c in self.citation_report
            if c.get("status") == VerificationStatus.FULLY_VERIFIED.value
        )
        case_verified = sum(
            1
            for c in self.citation_report
            if c.get("status") == VerificationStatus.CASE_VERIFIED.value
        )
        return {
            "total_citations": total,
            "fully_verified": fully_verified,
            "case_verified_principle_unconfirmed": case_verified,
            "unverified": len(self.unverified_citations),
            "overruled": len(self.overruled_citations),
            "misgrounded": len(self.misgrounded_citations),
            "verification_rate": (fully_verified + case_verified) / max(total, 1),
        }


# ── Audit log ──────────────────────────────────────────────────────────────────


class AuditLog:
    """Append-only event log for a single draft."""

    def __init__(self, draft_id: str) -> None:
        self.draft_id = draft_id
        self.entries: list[dict] = []

    def record(self, event_type: str, data: dict) -> None:
        self.entries.append(
            {
                "event_type": event_type,
                "timestamp": time.time(),
                "draft_id": self.draft_id,
                "data": data,
            }
        )
