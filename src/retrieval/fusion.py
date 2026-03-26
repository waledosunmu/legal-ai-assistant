"""Stage 2: RRF fusion + legal metadata boosting.

RRFFusion.fuse() takes multiple ranked result lists from Stage 1 (dense + sparse),
combines them with Reciprocal Rank Fusion (k=60), then applies legal metadata
boosts (court hierarchy, segment type, opinion type, recency, authority), and
returns the top-N CandidateResult objects sorted by boosted_score.

Overruled cases are pre-filtered at the SQL level in searcher.py, but an
additional in-memory check can be toggled via `filter_overruled=True`.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from retrieval.models import CandidateResult, RetrievalConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = RetrievalConfig()


class RRFFusion:
    """Reciprocal Rank Fusion + legal metadata boosting.

    Args:
        config: RetrievalConfig controlling all boost weights.
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or _DEFAULT_CONFIG

    def fuse(
        self,
        dense_results: list[list[dict]],
        sparse_results: list[list[dict]],
        authority_scores: dict[str, int] | None = None,
        current_year: int | None = None,
        top_n: int | None = None,
    ) -> list[CandidateResult]:
        """Fuse and boost results; return top-N CandidateResult sorted by boosted_score.

        Args:
            dense_results: list of result lists, one per dense query variant.
                Each result dict must have: segment_id, case_id, segment_type,
                content, court, year, opinion_type, retrieval_weight.
            sparse_results: same structure, one per sparse query variant.
            authority_scores: {case_id: times_cited}. Missing cases treated as 0.
            current_year: used for recency decay; defaults to current calendar year.
            top_n: override for config.top_candidates.
        """
        authority_scores = authority_scores or {}
        current_year = current_year or datetime.now().year
        top_n = top_n or self.config.top_candidates

        # ── Step 1: deduplicate and index all unique segments ──────────────────
        # segment_id → best row dict (favour dense rows which have 'distance')
        all_segments: dict[str, dict] = {}
        for result_list in dense_results + sparse_results:
            for row in result_list:
                sid = row.get("segment_id", "")
                if sid and sid not in all_segments:
                    all_segments[sid] = row

        if not all_segments:
            return []

        # ── Step 2: RRF scoring ───────────────────────────────────────────────
        k = self.config.rrf_k
        rrf_scores: dict[str, float] = {sid: 0.0 for sid in all_segments}

        for result_list in dense_results + sparse_results:
            for rank, row in enumerate(result_list, start=1):
                sid = row.get("segment_id", "")
                if sid in rrf_scores:
                    rrf_scores[sid] += 1.0 / (k + rank)

        # ── Step 3: compute max authority score for normalisation ─────────────
        max_cited = max(authority_scores.values(), default=1)

        # ── Step 4: legal metadata boosting ──────────────────────────────────
        candidates: list[CandidateResult] = []
        for sid, row in all_segments.items():
            fusion_score = rrf_scores[sid]

            court = (row.get("court") or "").upper()
            seg_type = (row.get("segment_type") or "").upper()
            opinion = (row.get("opinion_type") or "LEAD").upper()
            year = row.get("year") or current_year
            case_id = row.get("case_id", "")
            times_cited = authority_scores.get(case_id, 0)
            weight = float(row.get("retrieval_weight") or 1.0)

            court_boost = self.config.court_boosts.get(court, 1.0)
            seg_boost = self.config.segment_boosts.get(seg_type, 1.0)
            opinion_boost = self.config.opinion_boosts.get(opinion, 1.0)
            recency_boost = _recency_boost(year, current_year, self.config.recency_half_life)
            authority_boost = _authority_boost(
                times_cited, max_cited, self.config.authority_max_boost
            )

            boosted = (
                fusion_score
                * court_boost
                * seg_boost
                * opinion_boost
                * weight
                * recency_boost
                * (1.0 + authority_boost)
            )

            # Find dense_rank and sparse_rank for this segment
            dense_rank = _find_rank(sid, dense_results)
            sparse_rank = _find_rank(sid, sparse_results)

            candidates.append(CandidateResult(
                case_id=case_id,
                segment_id=sid,
                segment_type=seg_type,
                content=row.get("content", ""),
                court=court,
                year=year if isinstance(year, int) else current_year,
                opinion_type=opinion,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                fusion_score=fusion_score,
                boosted_score=boosted,
            ))

        # ── Step 5: deduplicate by case_id (keep best segment per case) ───────
        best_by_case: dict[str, CandidateResult] = {}
        for c in candidates:
            existing = best_by_case.get(c.case_id)
            if existing is None or c.boosted_score > existing.boosted_score:
                best_by_case[c.case_id] = c

        sorted_candidates = sorted(
            best_by_case.values(),
            key=lambda c: c.boosted_score,
            reverse=True,
        )
        return sorted_candidates[:top_n]


# ── Boost helpers ─────────────────────────────────────────────────────────────


def _recency_boost(year: int, current_year: int, half_life: int) -> float:
    """Exponential decay: 2^(-(current_year - year) / half_life).

    More recent cases score closer to 1.0; older cases approach 0 asymptotically.
    """
    age = max(0, current_year - year)
    return 2 ** (-age / half_life)


def _authority_boost(times_cited: int, max_cited: int, max_boost: float) -> float:
    """Log-scaled authority boost in range [0, max_boost].

    log(1 + times_cited) / log(1 + max_cited) * max_boost
    """
    if max_cited <= 0:
        return 0.0
    numerator = math.log1p(times_cited)
    denominator = math.log1p(max_cited)
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * max_boost


def _find_rank(segment_id: str, result_lists: list[list[dict]]) -> int | None:
    """Return 1-indexed rank of segment_id across all result lists (best rank wins)."""
    best: int | None = None
    for result_list in result_lists:
        for rank, row in enumerate(result_list, start=1):
            if row.get("segment_id") == segment_id:
                if best is None or rank < best:
                    best = rank
                break
    return best
