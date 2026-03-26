"""Stage 3: LLM reranking with Claude Haiku.

LLMReranker.rerank() takes the top-N CandidateResults from fusion, sends them
to Claude Haiku in a single batch prompt, and returns SearchResult objects with:
  - relevance_score = 0.70 × llm_score + 0.30 × fusion_score
  - relevance_explanation (≤ 150 chars)
  - case metadata (name, citation, court, year, authority)

If the Haiku call fails, we gracefully fall back to fusion_score-only ranking.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.models import CandidateResult, RetrievalConfig, SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = RetrievalConfig()

_RERANK_PROMPT = """You are a Nigerian legal research assistant. Evaluate how relevant each case is for the given query.

For each case, score it on three dimensions (0.0 to 1.0):
1. factual_similarity: How similar are the facts to the query?
2. principle_applicability: How applicable is the legal principle to the query?
3. authority_strength: How authoritative is this case for Nigerian courts?

Also write a short explanation (max 100 characters) of why this case is relevant.

Query: {query}

Cases to evaluate:
{cases_json}

Return ONLY a JSON array with one object per case in the same order:
[
  {{"id": "<segment_id>", "factual_similarity": 0.8, "principle_applicability": 0.9, "authority_strength": 0.7, "explanation": "short explanation"}},
  ...
]"""


class LLMReranker:
    """Rerank CandidateResults using Claude Haiku scoring.

    Args:
        anthropic_client: anthropic.AsyncAnthropic instance (or compatible mock).
            If None, reranking falls back to fusion_score ordering.
        config: RetrievalConfig controlling blend weights.
    """

    def __init__(self, anthropic_client=None, config: RetrievalConfig | None = None) -> None:
        self._anthropic = anthropic_client
        self.config = config or _DEFAULT_CONFIG

    async def rerank(
        self,
        query: str,
        candidates: list[CandidateResult],
        case_metadata: dict[str, dict],
        top_n: int = 15,
    ) -> list[SearchResult]:
        """Rerank candidates and return SearchResult list sorted by relevance_score.

        Args:
            query: original user query
            candidates: top-N fusion candidates (sorted by boosted_score)
            case_metadata: {case_id: {case_name, citation, court, year,
                             times_cited, authority_score}}
            top_n: number of results to return
        """
        if not candidates:
            return []

        # Attempt LLM reranking
        llm_scores: dict[str, float] = {}
        explanations: dict[str, str] = {}

        if self._anthropic:
            llm_scores, explanations = await self._call_haiku(query, candidates)

        # Build SearchResult list
        results: list[SearchResult] = []
        max_fusion = max((c.boosted_score for c in candidates), default=1.0)

        for candidate in candidates:
            meta = case_metadata.get(candidate.case_id, {})
            case_name = meta.get("case_name", "Unknown")
            citation = meta.get("citation")
            court = meta.get("court") or candidate.court
            year = meta.get("year") or candidate.year
            times_cited = meta.get("times_cited", 0)
            authority_score = meta.get("authority_score", 0)

            llm_score = llm_scores.get(candidate.segment_id)
            norm_fusion = candidate.boosted_score / max_fusion if max_fusion > 0 else 0.0

            if llm_score is not None:
                relevance_score = (
                    self.config.llm_weight * llm_score
                    + self.config.fusion_weight * norm_fusion
                )
            else:
                relevance_score = norm_fusion

            explanation = explanations.get(candidate.segment_id, "")

            results.append(SearchResult(
                case_id=candidate.case_id,
                case_name=case_name,
                case_name_short=_short_name(case_name),
                citation=citation,
                court=court,
                year=year,
                relevance_score=min(relevance_score, 1.0),
                relevance_explanation=explanation[:150],
                authority_score=authority_score,
                times_cited=times_cited,
                matched_segment={
                    "type": candidate.segment_type,
                    "content": candidate.content[:500],
                },
                verification_status="verified",
            ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_n]

    async def _call_haiku(
        self,
        query: str,
        candidates: list[CandidateResult],
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Call Claude Haiku and parse structured scoring response.

        Returns (llm_scores, explanations) where keys are segment_ids.
        """
        cases_json = json.dumps([
            {
                "id": c.segment_id,
                "court": c.court,
                "year": c.year,
                "segment_type": c.segment_type,
                "content": c.content[:300],
            }
            for c in candidates
        ], indent=2)

        prompt = _RERANK_PROMPT.format(query=query, cases_json=cases_json)

        try:
            settings = _get_settings()
            message = await self._anthropic.messages.create(
                model=settings.extraction_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("reranker.haiku_failed exc=%s", exc)
            return {}, {}

        llm_scores: dict[str, float] = {}
        explanations: dict[str, str] = {}

        for item in data:
            if not isinstance(item, dict):
                continue
            sid = item.get("id", "")
            if not sid:
                continue
            # Average the three scoring dimensions
            scores = [
                item.get("factual_similarity", 0.5),
                item.get("principle_applicability", 0.5),
                item.get("authority_strength", 0.5),
            ]
            llm_scores[sid] = sum(scores) / len(scores)
            explanations[sid] = str(item.get("explanation", ""))[:150]

        return llm_scores, explanations


# ── Helpers ────────────────────────────────────────────────────────────────────


def _short_name(case_name: str, max_length: int = 60) -> str:
    """Shorten a case name for display."""
    if len(case_name) <= max_length:
        return case_name
    # Try to cut at " v. " or " V. "
    parts = re.split(r"\s+[vV]\.\s+", case_name, maxsplit=1)
    if len(parts) == 2:
        appellant = parts[0].strip()[:25]
        respondent = parts[1].strip()[:25]
        return f"{appellant} v. {respondent}"
    return case_name[:max_length] + "..."


def _get_settings():
    from config import settings
    return settings
