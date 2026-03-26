"""Pydantic request/response schemas for the search and generation APIs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────────

MotionType = Literal[
    "motion_to_dismiss",
    "interlocutory_injunction",
    "stay_of_proceedings",
    "summary_judgment",
    "extension_of_time",
]

AreaOfLaw = Literal[
    "constitutional_law",
    "criminal_law",
    "contract_law",
    "land_law",
    "company_law",
    "family_law",
    "electoral_law",
    "administrative_law",
    "tort_law",
    "taxation_law",
]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=5000)
    motion_type: MotionType | None = None
    court_filter: list[str] | None = None
    year_min: int | None = Field(default=None, ge=1900, le=2100)
    year_max: int | None = Field(default=None, ge=1900, le=2100)
    area_of_law: AreaOfLaw | None = None
    max_results: int = Field(default=10, ge=1, le=20)
    include_statutes: bool = True


# ── Response ──────────────────────────────────────────────────────────────────


class MatchedSegment(BaseModel):
    type: str
    content: str


class CaseResult(BaseModel):
    case_id: str
    case_name: str
    case_name_short: str
    citation: str | None
    court: str
    year: int | None
    relevance_score: float
    relevance_explanation: str
    authority_score: int
    times_cited: int
    matched_segment: MatchedSegment
    verification_status: str


class QueryAnalysis(BaseModel):
    detected_motion_type: str | None
    detected_concepts: list[str]
    case_references_found: list[str]


class SearchMetadata(BaseModel):
    total_time_ms: int
    results_returned: int
    stage_timings_ms: dict[str, int] = Field(default_factory=dict)
    stage_counts: dict[str, int] = Field(default_factory=dict)
    cache: dict[str, bool] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    cases: list[CaseResult]
    statutes: list[dict]
    query_analysis: QueryAnalysis
    search_metadata: SearchMetadata


# ── Generation request ─────────────────────────────────────────────────────────


class SelectedCase(BaseModel):
    case_id: str
    case_name: str
    case_name_short: str = ""
    citation: str | None = None
    court: str = ""
    year: int | None = None
    matched_segment: MatchedSegment | None = None


class StatuteRef(BaseModel):
    section: str
    act: str = ""
    title: str = ""
    content: str = ""


class GenerateRequest(BaseModel):
    case_facts: str = Field(..., min_length=50, max_length=20000)
    motion_type: MotionType
    court_name: str = Field(..., min_length=3, max_length=200)
    division: str = Field(..., min_length=2, max_length=200)
    location: str = Field(..., min_length=2, max_length=200)
    suit_number: str = Field(..., min_length=3, max_length=100)
    applicant_name: str = Field(..., min_length=2, max_length=300)
    applicant_description: str = Field(default="Applicant", max_length=200)
    respondent_name: str = Field(..., min_length=2, max_length=300)
    respondent_description: str = Field(default="Respondent", max_length=200)
    position: Literal["applicant", "respondent"] = "applicant"
    relief_sought: str = Field(..., min_length=10, max_length=5000)
    selected_cases: list[SelectedCase]
    statutes: list[StatuteRef] = Field(default_factory=list)
    counsel_name: str = ""
    counsel_firm: str = ""
    counsel_address: str = ""
    date: str = ""
    deponent_name: str = ""
    deponent_description: str = ""
    deponent_capacity: str = ""


# ── Generation response ───────────────────────────────────────────────────────


class CitationVerificationResult(BaseModel):
    name: str | None = None
    citation: str | None = None
    case_id: str | None = None
    status: str
    verified: bool = False
    warning: str | None = None
    checks: dict = Field(default_factory=dict)


class StrengthScore(BaseModel):
    issue_number: int
    issue_text: str = ""
    legal_soundness: float | None = None
    factual_applicability: float | None = None
    authority_strength: float | None = None
    vulnerability: float | None = None
    overall_score: float | None = None
    weaknesses: list[str] = Field(default_factory=list)


class CounterArgument(BaseModel):
    issue_number: int
    counter_argument: str = ""
    potential_authority: str = ""
    suggested_rebuttal: str = ""


class ArgumentOut(BaseModel):
    issue_number: int
    issue_text: str
    argument_text: str
    cases_cited: list[dict] = Field(default_factory=list)
    statutes_cited: list[dict] = Field(default_factory=list)


class MotionPaperOut(BaseModel):
    court_name: str
    suit_number: str
    applicant_name: str
    respondent_name: str
    motion_type: str
    prayers: list[str]
    grounds: list[str]
    rendered_text: str


class AffidavitOut(BaseModel):
    deponent_name: str
    paragraphs: list[str]
    rendered_text: str


class WrittenAddressOut(BaseModel):
    title: str
    introduction: str
    issues_for_determination: list[str]
    arguments: list[ArgumentOut]
    conclusion: str
    rendered_text: str


class CitationSummary(BaseModel):
    total_citations: int
    fully_verified: int
    case_verified_principle_unconfirmed: int
    unverified: int
    overruled: int
    misgrounded: int
    verification_rate: float


class ReadinessReport(BaseModel):
    ready: bool
    warnings: list[str] = Field(default_factory=list)
    cases_count: int = 0
    statutes_count: int = 0


class GenerateResponse(BaseModel):
    motion_paper: MotionPaperOut
    affidavit: AffidavitOut
    written_address: WrittenAddressOut
    citation_report: list[CitationVerificationResult]
    citation_summary: CitationSummary
    strength_report: list[StrengthScore]
    counter_arguments: list[CounterArgument]
    readiness: ReadinessReport
