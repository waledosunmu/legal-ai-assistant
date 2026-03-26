"""Shared data models for the Phase 1 retrieval engine.

All retrieval components use these types to pass data between stages, enabling
clean dependency injection and easy mocking in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Query understanding ────────────────────────────────────────────────────────


@dataclass
class ParsedQuery:
    """Result of QueryParser.parse() — structured understanding of a search query."""

    original: str
    motion_type: str | None = None  # "motion_to_dismiss" | "interlocutory_injunction" | ...
    detected_concepts: list[str] = field(default_factory=list)
    case_references: list[str] = field(default_factory=list)  # raw citation strings
    area_of_law: str | None = None
    confidence: float = 0.5  # 0–1; Layer 3 (LLM) triggered if < 0.5
    step_back_query: str | None = None  # LLM-generated broader restatement


@dataclass
class ExpandedQueries:
    """Result of QueryExpander.expand() — multiple query variants + embeddings."""

    dense_texts: list[str] = field(default_factory=list)  # up to 3 text variants
    dense_embeddings: list[list[float]] = field(default_factory=list)  # one per text
    sparse_texts: list[str] = field(default_factory=list)  # up to 3 keyword strings


# ── Retrieval candidates ───────────────────────────────────────────────────────


@dataclass
class CandidateResult:
    """A single case segment surfaced by Stage 1 and scored by Stage 2."""

    case_id: str  # UUID string
    segment_id: str  # UUID string
    segment_type: str  # e.g. "RATIO", "HOLDING", "ANALYSIS"
    content: str  # segment text
    court: str = ""
    year: int | None = None
    opinion_type: str = "LEAD"
    dense_rank: int | None = None  # rank within dense results (1-indexed); None if not found
    sparse_rank: int | None = None  # rank within sparse results (1-indexed); None if not found
    fusion_score: float = 0.0  # raw RRF score
    boosted_score: float = 0.0  # after legal metadata boosts


# ── Final search results ───────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A fully ranked and explained result returned to the caller."""

    case_id: str
    case_name: str
    case_name_short: str
    citation: str | None
    court: str
    year: int | None
    relevance_score: float  # final blended score (LLM + fusion)
    relevance_explanation: str  # ≤ 150 char explanation from LLM reranker
    authority_score: int  # from case_authority_scores materialized view
    times_cited: int
    matched_segment: dict  # {"type": ..., "content": ...}
    verification_status: str = "verified"  # "verified" | "unverified"


# ── Retrieval configuration ────────────────────────────────────────────────────


@dataclass
class RetrievalConfig:
    """Tunable parameters for the retrieval pipeline."""

    # Stage 1 — search limits
    dense_limit: int = 60  # candidates per dense variant
    sparse_limit: int = 60  # candidates per sparse variant

    # Stage 2 — RRF + boosting
    rrf_k: int = 60
    top_candidates: int = 30  # candidates passed to reranker

    # Stage 3 — reranking
    top_results: int = 15  # final results before slicing to max_results
    llm_weight: float = 0.70
    fusion_weight: float = 0.30

    # Court boosting
    court_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "NGSC": 1.15,
            "NGCA": 1.08,
            "NGFCHC": 1.0,
            "NGLAHC": 1.0,
            "NGKNHC": 1.0,
            "NGBAHC": 1.0,
            "NGBEHC": 1.0,
        }
    )

    # Segment type boosting
    segment_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "RATIO": 1.20,
            "HOLDING": 1.15,
            "ANALYSIS": 1.0,
            "FACTS": 0.9,
            "ISSUE": 1.05,
            "OBITER": 0.8,
            "ORDER": 0.95,
            "INTRODUCTION": 0.85,
            "CAPTION": 0.7,
        }
    )

    # Opinion type boosting
    opinion_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "LEAD": 1.0,
            "CONCURRING": 0.7,
            "DISSENTING": 0.4,
        }
    )

    # Recency decay — half-life in years
    recency_half_life: int = 15

    # Authority score — max additive boost (10%)
    authority_max_boost: float = 0.10

    # LLM trigger threshold (Layer 3 in QueryParser)
    llm_confidence_threshold: float = 0.5
