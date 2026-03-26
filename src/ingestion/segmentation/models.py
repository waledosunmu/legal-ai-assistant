"""Segment type definitions and data models for judgment segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SegmentType(StrEnum):
    """Types of meaningful segments within a Nigerian court judgment."""

    CAPTION = "caption"  # Court, parties, suit number header
    BACKGROUND = "background"  # Procedural history / how case got here
    FACTS = "facts"  # Statement of facts
    ISSUES = "issues"  # Issues for determination
    SUBMISSION = "submission"  # Counsel's arguments
    ANALYSIS = "analysis"  # Court's analysis / reasoning
    HOLDING = "holding"  # Court's determination on each issue
    RATIO = "ratio"  # The binding principle of law
    OBITER = "obiter"  # Non-binding observations
    ORDERS = "orders"  # Final orders / disposition
    DISSENT = "dissent"  # Dissenting opinion
    CONCURRENCE = "concurrence"  # Concurring opinion
    CITED_CASE = "cited_case"  # Block quoting / referencing another case


@dataclass
class JudgmentSegment:
    """A meaningful segment extracted from a judgment."""

    segment_type: SegmentType
    content: str
    position: int  # Paragraph index in the original text
    confidence: float  # 0.0–1.0 classification confidence
    issue_number: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SegmentedJudgment:
    """A fully segmented judgment ready for storage and embedding."""

    case_id: str
    segments: list[JudgmentSegment]
    issues_for_determination: list[str]
    holdings: list[dict]  # [{issue, determination, reasoning}]
    ratio_decidendi: str | None
    orders: list[str]
    cases_cited: list[dict]  # [{name, citation, context, treatment}]
    statutes_cited: list[str]
