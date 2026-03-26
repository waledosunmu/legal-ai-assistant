"""Unit tests for generation.templates — document rendering."""

from __future__ import annotations

import pytest

from generation.models import ArgumentSection, MotionPaper, SupportingAffidavit, WrittenAddress
from generation.templates import (
    FALLBACK_ISSUES,
    render_affidavit,
    render_motion_paper,
    render_written_address,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _motion_paper(**kw) -> MotionPaper:
    defaults = dict(
        court_name="IN THE FEDERAL HIGH COURT OF NIGERIA",
        division="IN THE LAGOS JUDICIAL DIVISION",
        location="HOLDEN AT LAGOS",
        suit_number="SUIT NO: FHC/L/CS/123/2026",
        applicant_name="ABC Manufacturing Ltd",
        applicant_description="Applicant",
        respondent_name="XYZ Supplies Ltd",
        respondent_description="Respondent",
        motion_type="MOTION ON NOTICE",
        brought_pursuant_to=["Order 26 Rule 1 of the FHC Rules", "Section 251 CFRN"],
        prayers=["An order dismissing the suit for want of jurisdiction"],
        grounds=["The court lacks jurisdiction over the subject matter"],
        date="19th March, 2026",
        counsel_name="A. Barrister Esq.",
        counsel_firm="Law & Associates",
        counsel_address="1 Law Lane, Lagos",
    )
    defaults.update(kw)
    return MotionPaper(**defaults)


def _affidavit(**kw) -> SupportingAffidavit:
    defaults = dict(
        court_header="IN THE FEDERAL HIGH COURT OF NIGERIA",
        suit_number="FHC/L/CS/123/2026",
        parties="ABC Manufacturing Ltd v. XYZ Supplies Ltd",
        deponent_name="John Doe",
        deponent_description="Male, Nigerian, Businessman, of 1 Main St, Lagos",
        deponent_capacity="Director of the Applicant company",
        paragraphs=[
            "I am the Managing Director of the Applicant company",
            "The Respondent commenced this suit on the 1st of January 2026",
            "The subject matter falls outside the jurisdiction of this Court",
        ],
        jurat="Sworn to at Lagos\nthis ______ day of _______ 20___",
    )
    defaults.update(kw)
    return SupportingAffidavit(**defaults)


def _written_address(**kw) -> WrittenAddress:
    defaults = dict(
        court_header="IN THE FEDERAL HIGH COURT OF NIGERIA",
        suit_number="FHC/L/CS/123/2026",
        parties="ABC Manufacturing Ltd v. XYZ Supplies Ltd",
        title="Written Address in Support of Motion to Dismiss",
        introduction="This Written Address is filed in support of the Applicant's Motion on Notice.",
        issues_for_determination=[
            "Whether this Honourable Court has jurisdiction over the subject matter?",
            "Whether the Applicant has disclosed a reasonable cause of action?",
        ],
        arguments=[
            ArgumentSection(
                issue_number=1,
                issue_text="Whether this Honourable Court has jurisdiction?",
                argument_text="It is trite law that the question of jurisdiction is fundamental...",
                cases_cited=[{"name": "Madukolu v. Nkemdilim"}],
            ),
            ArgumentSection(
                issue_number=2,
                issue_text="Whether a reasonable cause of action has been disclosed?",
                argument_text="A cause of action must disclose facts which give rise to a right...",
            ),
        ],
        conclusion="We urge this Honourable Court to grant the application.",
        counsel_signature="A. Barrister Esq.\nLaw & Associates",
    )
    defaults.update(kw)
    return WrittenAddress(**defaults)


# ── FALLBACK_ISSUES ────────────────────────────────────────────────────────────


class TestFallbackIssues:
    def test_all_motion_types_covered(self):
        expected = {
            "motion_to_dismiss",
            "interlocutory_injunction",
            "stay_of_proceedings",
            "summary_judgment",
            "extension_of_time",
        }
        assert set(FALLBACK_ISSUES.keys()) == expected

    def test_each_has_at_least_two_issues(self):
        for mt, issues in FALLBACK_ISSUES.items():
            assert len(issues) >= 2, f"{mt} has fewer than 2 fallback issues"

    def test_issues_are_questions(self):
        for mt, issues in FALLBACK_ISSUES.items():
            for issue in issues:
                assert issue.endswith("?"), f"Fallback issue for {mt} is not a question: {issue}"


# ── render_motion_paper ────────────────────────────────────────────────────────


class TestRenderMotionPaper:
    def test_contains_court_header(self):
        text = render_motion_paper(_motion_paper())
        assert "IN THE FEDERAL HIGH COURT OF NIGERIA" in text

    def test_contains_parties(self):
        text = render_motion_paper(_motion_paper())
        assert "ABC MANUFACTURING LTD" in text
        assert "XYZ SUPPLIES LTD" in text

    def test_contains_prayers(self):
        text = render_motion_paper(_motion_paper())
        assert "An order dismissing the suit" in text

    def test_contains_grounds(self):
        text = render_motion_paper(_motion_paper())
        assert "The court lacks jurisdiction" in text

    def test_contains_pursuant_to(self):
        text = render_motion_paper(_motion_paper())
        assert "Order 26 Rule 1" in text
        assert "Section 251 CFRN" in text

    def test_contains_counsel_block(self):
        text = render_motion_paper(_motion_paper())
        assert "A. Barrister Esq." in text
        assert "Law & Associates" in text

    def test_contains_catch_all_prayer(self):
        text = render_motion_paper(_motion_paper())
        assert "further order" in text

    def test_multiple_prayers_numbered(self):
        mp = _motion_paper(prayers=["First prayer", "Second prayer", "Third prayer"])
        text = render_motion_paper(mp)
        assert "1. First prayer" in text
        assert "2. Second prayer" in text
        assert "3. Third prayer" in text

    def test_suit_number(self):
        text = render_motion_paper(_motion_paper())
        assert "FHC/L/CS/123/2026" in text


# ── render_affidavit ───────────────────────────────────────────────────────────


class TestRenderAffidavit:
    def test_contains_deponent(self):
        text = render_affidavit(_affidavit())
        assert "John Doe" in text

    def test_paragraphs_prefixed_with_that(self):
        text = render_affidavit(_affidavit())
        assert "THAT I am the Managing Director" in text
        assert "THAT The Respondent commenced" in text

    def test_paragraphs_numbered(self):
        text = render_affidavit(_affidavit())
        assert "1. THAT" in text
        assert "2. THAT" in text
        assert "3. THAT" in text

    def test_contains_oath_act_paragraph(self):
        text = render_affidavit(_affidavit())
        assert "Oaths Act" in text

    def test_contains_jurat(self):
        text = render_affidavit(_affidavit())
        assert "Sworn to at Lagos" in text

    def test_exhibits_rendered(self):
        aff = _affidavit()
        aff.exhibits = [
            {"label": "Exhibit A", "description": "Copy of the contract"},
            {"label": "Exhibit B", "description": "Correspondence dated 1/1/2026"},
        ]
        text = render_affidavit(aff)
        assert "Exhibit A" in text
        assert "Copy of the contract" in text

    def test_no_exhibits_section_when_empty(self):
        text = render_affidavit(_affidavit())
        assert "EXHIBITS:" not in text


# ── render_written_address ─────────────────────────────────────────────────────


class TestRenderWrittenAddress:
    def test_contains_title(self):
        text = render_written_address(_written_address())
        assert "WRITTEN ADDRESS IN SUPPORT OF MOTION TO DISMISS" in text

    def test_contains_introduction(self):
        text = render_written_address(_written_address())
        assert "This Written Address is filed in support" in text

    def test_contains_issues(self):
        text = render_written_address(_written_address())
        assert "Whether this Honourable Court has jurisdiction" in text

    def test_issues_numbered(self):
        text = render_written_address(_written_address())
        assert "1." in text
        assert "2." in text

    def test_contains_arguments(self):
        text = render_written_address(_written_address())
        assert "ISSUE 1:" in text
        assert "It is trite law that the question of jurisdiction" in text

    def test_contains_conclusion(self):
        text = render_written_address(_written_address())
        assert "We urge this Honourable Court" in text

    def test_contains_counsel_signature(self):
        text = render_written_address(_written_address())
        assert "A. Barrister Esq." in text

    def test_section_headings_present(self):
        text = render_written_address(_written_address())
        assert "1.0 INTRODUCTION" in text
        assert "2.0 ISSUES FOR DETERMINATION" in text
        assert "3.0 ARGUMENTS" in text
        assert "4.0 CONCLUSION" in text
