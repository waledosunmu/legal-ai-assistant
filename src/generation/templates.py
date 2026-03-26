"""Nigerian court motion document templates.

Provides text assembly for the three documents in a motion filing:
  1. Motion Paper (Notice of Motion)
  2. Supporting Affidavit
  3. Written Address

Templates follow Federal High Court (Civil Procedure) Rules 2019 format.
"""

from __future__ import annotations

from generation.models import (
    ArgumentSection,
    MotionPaper,
    SupportingAffidavit,
    WrittenAddress,
)


# ── Fallback issues per motion type ────────────────────────────────────────────

FALLBACK_ISSUES: dict[str, list[str]] = {
    "motion_to_dismiss": [
        "Whether this Honourable Court has the jurisdiction to entertain this suit?",
        "Whether the Applicant has disclosed a reasonable cause of action against the Respondent?",
    ],
    "interlocutory_injunction": [
        "Whether the Applicant has established a prima facie case or a serious question to be tried?",
        "Whether damages would be an adequate remedy if the injunction is not granted?",
        "Whether the balance of convenience is in favour of granting the injunction?",
    ],
    "stay_of_proceedings": [
        "Whether the Applicant has shown sufficient cause for the proceedings to be stayed?",
        "Whether the Applicant will suffer irreparable harm if the stay is not granted?",
    ],
    "summary_judgment": [
        "Whether the Respondent has a real prospect of successfully defending the claim?",
        "Whether there is a compelling reason why the case should proceed to trial?",
    ],
    "extension_of_time": [
        "Whether the Applicant has provided satisfactory reasons for the failure to act within the prescribed time?",
        "Whether the Applicant's application has merit?",
        "Whether the grant of extension of time will occasion any injustice to the Respondent?",
    ],
}


# ── Template rendering ─────────────────────────────────────────────────────────


def render_motion_paper(mp: MotionPaper) -> str:
    """Render a MotionPaper dataclass into formatted text."""
    prayers_text = "\n".join(
        f"    {i}. {prayer}" for i, prayer in enumerate(mp.prayers, 1)
    )
    grounds_text = "\n".join(
        f"    {i}. {ground}" for i, ground in enumerate(mp.grounds, 1)
    )
    pursuant_text = "; ".join(mp.brought_pursuant_to)

    return f"""{mp.court_name}
{mp.division}
{mp.location}

{mp.suit_number}

BETWEEN:

{mp.applicant_name.upper()} ……………………………… {mp.applicant_description.upper()}

AND

{mp.respondent_name.upper()} ……………………………… {mp.respondent_description.upper()}

{'=' * 60}
{mp.motion_type.upper()}
{'=' * 60}

TAKE NOTICE that this Honourable Court will be moved on the ______ day of \
_______ 20___ at the hour of 9 o'clock in the forenoon or so soon thereafter \
as Counsel for the {mp.applicant_description} may be heard, praying this \
Honourable Court for the following orders:

PRAYERS:

{prayers_text}

    {len(mp.prayers) + 1}. And for such further order or other orders as this \
Honourable Court may deem fit to make in the circumstances.

GROUNDS:

{grounds_text}

This application is brought pursuant to {pursuant_text}.

Dated this {mp.date}

___________________________
{mp.counsel_name}
{mp.counsel_firm}
{mp.counsel_address}
Counsel to the {mp.applicant_description}
"""


def render_affidavit(aff: SupportingAffidavit) -> str:
    """Render a SupportingAffidavit dataclass into formatted text."""
    paragraphs_text = "\n\n".join(
        f"    {i}. THAT {para}" for i, para in enumerate(aff.paragraphs, 1)
    )

    exhibits_text = ""
    if aff.exhibits:
        exhibit_lines = "\n".join(
            f"    {e.get('label', '')}: {e.get('description', '')}"
            for e in aff.exhibits
        )
        exhibits_text = f"\n\nEXHIBITS:\n{exhibit_lines}"

    return f"""{aff.court_header}

{aff.suit_number}

{aff.parties}

{'=' * 60}
AFFIDAVIT IN SUPPORT OF {aff.court_header.split('—')[-1].strip() if '—' in aff.court_header else 'THE APPLICATION'}
{'=' * 60}

I, {aff.deponent_name}, {aff.deponent_description}, do hereby make oath and \
state as follows:

{paragraphs_text}

    {len(aff.paragraphs) + 1}. THAT I make this affidavit in good faith \
believing same to be true and correct and in accordance with the Oaths Act, \
Cap. O1, Laws of the Federation of Nigeria, 2004.
{exhibits_text}

{aff.jurat}

___________________________
DEPONENT
"""


def render_written_address(wa: WrittenAddress) -> str:
    """Render a WrittenAddress dataclass into formatted text."""
    issues_text = "\n".join(
        f"    {i}. {issue}"
        for i, issue in enumerate(wa.issues_for_determination, 1)
    )

    arguments_text = "\n\n".join(
        _render_argument_section(arg) for arg in wa.arguments
    )

    return f"""{wa.court_header}

{wa.suit_number}

{wa.parties}

{'=' * 60}
{wa.title.upper()}
{'=' * 60}

1.0 INTRODUCTION

{wa.introduction}

2.0 ISSUES FOR DETERMINATION

The following issue(s) arise for determination in this application:

{issues_text}

3.0 ARGUMENTS

{arguments_text}

4.0 CONCLUSION

{wa.conclusion}

{wa.counsel_signature}
"""


def _render_argument_section(arg: ArgumentSection) -> str:
    """Render a single argument section with its issue heading."""
    return f"""ISSUE {arg.issue_number}: {arg.issue_text}

{arg.argument_text}"""
