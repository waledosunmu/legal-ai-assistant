"""Unit tests for generation.models — data structures and audit log."""

from __future__ import annotations

import pytest

from generation.models import (
    ArgumentSection,
    AuditLog,
    GenerationRequest,
    GenerationResult,
    MotionPaper,
    SupportingAffidavit,
    VerificationStatus,
    WrittenAddress,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _motion_paper() -> MotionPaper:
    return MotionPaper(
        court_name="IN THE FEDERAL HIGH COURT OF NIGERIA",
        division="IN THE LAGOS JUDICIAL DIVISION",
        location="HOLDEN AT LAGOS",
        suit_number="FHC/L/CS/123/2026",
        applicant_name="ABC Manufacturing Ltd",
        applicant_description="Applicant",
        respondent_name="XYZ Supplies Ltd",
        respondent_description="Respondent",
        motion_type="MOTION ON NOTICE",
        brought_pursuant_to=["Order 26 Rule 1 of the FHC Rules"],
        prayers=["An order dismissing the suit"],
        grounds=["The court lacks jurisdiction"],
        date="19th March, 2026",
        counsel_name="A. Barrister Esq.",
        counsel_firm="Law Chambers",
        counsel_address="1 Law Lane, Lagos",
    )


def _affidavit() -> SupportingAffidavit:
    return SupportingAffidavit(
        court_header="IN THE FEDERAL HIGH COURT",
        suit_number="FHC/L/CS/123/2026",
        parties="ABC v. XYZ",
        deponent_name="John Doe",
        deponent_description="Male, Nigerian, Businessman",
        deponent_capacity="Director of the Applicant",
        paragraphs=["I am the Director", "The Respondent failed to deliver"],
        jurat="Sworn to at Lagos this ____ day of ____ 20___",
    )


def _written_address() -> WrittenAddress:
    return WrittenAddress(
        court_header="IN THE FEDERAL HIGH COURT",
        suit_number="FHC/L/CS/123/2026",
        parties="ABC v. XYZ",
        title="Written Address in Support of Motion to Dismiss",
        introduction="This Written Address is filed in support of...",
        issues_for_determination=["Whether the court has jurisdiction?"],
        arguments=[
            ArgumentSection(
                issue_number=1,
                issue_text="Whether the court has jurisdiction?",
                argument_text="It is trite law that jurisdiction is fundamental...",
                cases_cited=[{"name": "Madukolu v. Nkemdilim", "citation": "(1962) 2 SCNLR 341"}],
                statutes_cited=[{"section": "Section 251", "act": "CFRN 1999"}],
            ),
        ],
        conclusion="We urge this Honourable Court to grant the application.",
        counsel_signature="A. Barrister Esq.\nLaw Chambers",
    )


def _generation_result(**overrides) -> GenerationResult:
    defaults = {
        "motion_paper": _motion_paper(),
        "affidavit": _affidavit(),
        "written_address": _written_address(),
        "citation_report": [],
        "strength_report": [],
        "counter_arguments": [],
        "readiness_report": {"ready": True, "warnings": []},
        "audit_log": [],
    }
    defaults.update(overrides)
    return GenerationResult(**defaults)


# ── VerificationStatus ─────────────────────────────────────────────────────────


class TestVerificationStatus:
    def test_values(self):
        assert VerificationStatus.FULLY_VERIFIED.value == "fully_verified"
        assert VerificationStatus.NOT_IN_CORPUS.value == "not_in_corpus"
        assert VerificationStatus.OVERRULED.value == "overruled"

    def test_is_string_enum(self):
        assert isinstance(VerificationStatus.CASE_EXISTS, str)
        assert VerificationStatus.CASE_EXISTS == "case_exists"


# ── ArgumentSection ────────────────────────────────────────────────────────────


class TestArgumentSection:
    def test_basic_creation(self):
        arg = ArgumentSection(
            issue_number=1,
            issue_text="Test issue?",
            argument_text="The law is clear...",
        )
        assert arg.issue_number == 1
        assert arg.cases_cited == []
        assert arg.statutes_cited == []

    def test_with_citations(self):
        arg = ArgumentSection(
            issue_number=2,
            issue_text="Another issue?",
            argument_text="As held in...",
            cases_cited=[{"name": "Case A v. B"}],
            statutes_cited=[{"section": "S.1", "act": "Act X"}],
        )
        assert len(arg.cases_cited) == 1
        assert len(arg.statutes_cited) == 1


# ── GenerationResult properties ────────────────────────────────────────────────


class TestGenerationResult:
    def test_unverified_citations(self):
        result = _generation_result(
            citation_report=[
                {"name": "A v. B", "status": "not_in_corpus"},
                {"name": "C v. D", "status": "fully_verified"},
            ]
        )
        assert len(result.unverified_citations) == 1
        assert result.unverified_citations[0]["name"] == "A v. B"

    def test_overruled_citations(self):
        result = _generation_result(
            citation_report=[
                {"name": "Old v. Case", "status": "overruled"},
                {"name": "Good v. Law", "status": "fully_verified"},
            ]
        )
        assert len(result.overruled_citations) == 1

    def test_misgrounded_citations(self):
        result = _generation_result(
            citation_report=[
                {
                    "name": "X v. Y",
                    "status": "case_verified",
                    "checks": {"attribution": {"verified": False}},
                },
                {
                    "name": "A v. B",
                    "status": "case_verified",
                    "checks": {"attribution": {"verified": True}},
                },
            ]
        )
        assert len(result.misgrounded_citations) == 1
        assert result.misgrounded_citations[0]["name"] == "X v. Y"

    def test_weak_arguments(self):
        result = _generation_result(
            strength_report=[
                {"issue_number": 1, "overall_score": 3.5},
                {"issue_number": 2, "overall_score": 7.0},
                {"issue_number": 3, "overall_score": None},
            ]
        )
        assert len(result.weak_arguments) == 1
        assert result.weak_arguments[0]["issue_number"] == 1

    def test_citation_summary_empty(self):
        result = _generation_result(citation_report=[])
        summary = result.citation_summary
        assert summary["total_citations"] == 0
        assert summary["verification_rate"] == 0.0

    def test_citation_summary_mixed(self):
        result = _generation_result(
            citation_report=[
                {"status": "fully_verified"},
                {"status": "fully_verified"},
                {"status": "case_verified", "checks": {"attribution": {"verified": True}}},
                {"status": "not_in_corpus"},
                {"status": "overruled"},
            ]
        )
        summary = result.citation_summary
        assert summary["total_citations"] == 5
        assert summary["fully_verified"] == 2
        assert summary["case_verified_principle_unconfirmed"] == 1
        assert summary["unverified"] == 1
        assert summary["overruled"] == 1
        assert summary["verification_rate"] == 3 / 5


# ── AuditLog ───────────────────────────────────────────────────────────────────


class TestAuditLog:
    def test_record_appends(self):
        log = AuditLog(draft_id="d1")
        log.record("event_a", {"key": "val"})
        log.record("event_b", {"n": 42})
        assert len(log.entries) == 2
        assert log.entries[0]["event_type"] == "event_a"
        assert log.entries[1]["data"]["n"] == 42

    def test_has_timestamps(self):
        log = AuditLog(draft_id="d2")
        log.record("test", {})
        assert "timestamp" in log.entries[0]
        assert isinstance(log.entries[0]["timestamp"], float)

    def test_draft_id_stored(self):
        log = AuditLog(draft_id="my-draft")
        log.record("x", {})
        assert log.entries[0]["draft_id"] == "my-draft"


# ── GenerationRequest ─────────────────────────────────────────────────────────


class TestGenerationRequest:
    def test_basic_creation(self):
        req = GenerationRequest(
            case_facts="The defendant breached the contract...",
            motion_type="motion_to_dismiss",
            court_name="Federal High Court",
            division="Lagos",
            location="Lagos",
            suit_number="FHC/L/CS/1/2026",
            applicant_name="ABC Ltd",
            applicant_description="Applicant",
            respondent_name="XYZ Ltd",
            respondent_description="Respondent",
            position="applicant",
            relief_sought="Dismissal of the suit",
            selected_cases=[],
        )
        assert req.motion_type == "motion_to_dismiss"
        assert req.statutes == []
        assert req.draft_id == ""
