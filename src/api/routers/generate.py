"""POST /api/v1/generate route — motion document generation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AffidavitOut,
    ArgumentOut,
    CitationSummary,
    CitationVerificationResult,
    CounterArgument,
    GenerateRequest,
    GenerateResponse,
    MotionPaperOut,
    ReadinessReport,
    StrengthScore,
    WrittenAddressOut,
)
from generation.models import GenerationRequest
from generation.templates import render_affidavit, render_motion_paper, render_written_address

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_pipeline():
    """Lazy import to avoid circular dep at module load time."""
    from api.app import get_pipeline
    return get_pipeline()


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate a complete motion (Motion Paper + Affidavit + Written Address)."""
    pipeline = _get_pipeline()

    # Convert API schema to internal request
    gen_request = GenerationRequest(
        case_facts=request.case_facts,
        motion_type=request.motion_type,
        court_name=request.court_name,
        division=request.division,
        location=request.location,
        suit_number=request.suit_number,
        applicant_name=request.applicant_name,
        applicant_description=request.applicant_description,
        respondent_name=request.respondent_name,
        respondent_description=request.respondent_description,
        position=request.position,
        relief_sought=request.relief_sought,
        selected_cases=[c.model_dump() for c in request.selected_cases],
        statutes=[s.model_dump() for s in request.statutes],
        counsel_name=request.counsel_name,
        counsel_firm=request.counsel_firm,
        counsel_address=request.counsel_address,
        date=request.date,
        deponent_name=request.deponent_name,
        deponent_description=request.deponent_description,
        deponent_capacity=request.deponent_capacity,
    )

    try:
        result = await pipeline.generate(gen_request)
    except Exception as exc:
        logger.error("generate.handler_failed exc=%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed. Please try again.")

    # Render document text
    mp_text = render_motion_paper(result.motion_paper)
    aff_text = render_affidavit(result.affidavit)
    wa_text = render_written_address(result.written_address)

    return GenerateResponse(
        motion_paper=MotionPaperOut(
            court_name=result.motion_paper.court_name,
            suit_number=result.motion_paper.suit_number,
            applicant_name=result.motion_paper.applicant_name,
            respondent_name=result.motion_paper.respondent_name,
            motion_type=result.motion_paper.motion_type,
            prayers=result.motion_paper.prayers,
            grounds=result.motion_paper.grounds,
            rendered_text=mp_text,
        ),
        affidavit=AffidavitOut(
            deponent_name=result.affidavit.deponent_name,
            paragraphs=result.affidavit.paragraphs,
            rendered_text=aff_text,
        ),
        written_address=WrittenAddressOut(
            title=result.written_address.title,
            introduction=result.written_address.introduction,
            issues_for_determination=result.written_address.issues_for_determination,
            arguments=[
                ArgumentOut(
                    issue_number=arg.issue_number,
                    issue_text=arg.issue_text,
                    argument_text=arg.argument_text,
                    cases_cited=arg.cases_cited,
                    statutes_cited=arg.statutes_cited,
                )
                for arg in result.written_address.arguments
            ],
            conclusion=result.written_address.conclusion,
            rendered_text=wa_text,
        ),
        citation_report=[
            CitationVerificationResult(
                name=c.get("name"),
                citation=c.get("citation"),
                case_id=c.get("case_id"),
                status=c.get("status", "unknown"),
                verified=c.get("verified", False),
                warning=c.get("warning"),
                checks=c.get("checks", {}),
            )
            for c in result.citation_report
        ],
        citation_summary=CitationSummary(**result.citation_summary),
        strength_report=[
            StrengthScore(
                issue_number=s.get("issue_number", 0),
                issue_text=s.get("issue_text", ""),
                legal_soundness=s.get("legal_soundness"),
                factual_applicability=s.get("factual_applicability"),
                authority_strength=s.get("authority_strength"),
                vulnerability=s.get("vulnerability"),
                overall_score=s.get("overall_score"),
                weaknesses=s.get("weaknesses", []),
            )
            for s in result.strength_report
        ],
        counter_arguments=[
            CounterArgument(
                issue_number=ca.get("issue_number", 0),
                counter_argument=ca.get("counter_argument", ""),
                potential_authority=ca.get("potential_authority", ""),
                suggested_rebuttal=ca.get("suggested_rebuttal", ""),
            )
            for ca in result.counter_arguments
        ],
        readiness=ReadinessReport(**result.readiness_report),
    )
