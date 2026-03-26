"""RetrievalEngine — orchestrates the full 3-stage retrieval pipeline.

Pipeline:
  1. Parse query (QueryParser)
  2. Expand to multiple variants + embeddings (QueryExpander)
  3. Stage 1: dense + sparse + exact search in parallel (Searchers)
  4. Fetch authority scores from DB
  5. Stage 2: RRF fusion + legal metadata boosting (RRFFusion)
  6. Fetch case metadata for candidates
  7. Stage 3: LLM reranking (LLMReranker)
  8. Statute retrieval in parallel with stages 3-7
  9. Return structured response dict

Usage::

    engine = await create_engine()
    result = await engine.search("grounds for dismissal for want of jurisdiction",
                                  motion_type="motion_to_dismiss")
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.fusion import RRFFusion
from retrieval.models import CandidateResult, ParsedQuery, RetrievalConfig, SearchResult
from retrieval.query_expander import QueryExpander
from retrieval.query_parser import QueryParser
from retrieval.reranker import LLMReranker
from retrieval.searcher import DenseSearcher, ExactSearcher, SparseSearcher
from retrieval.statutes import StatuteRetriever

logger = logging.getLogger(__name__)

# ── Authority score SQL ────────────────────────────────────────────────────────

_AUTHORITY_SQL = """
SELECT case_id::text, times_cited, authority_score
FROM case_authority_scores
WHERE case_id = ANY($1::uuid[])
"""

_CASE_METADATA_SQL = """
SELECT
    id::text    AS case_id,
    case_name,
    citation,
    court::text AS court,
    year,
    status::text AS status
FROM cases
WHERE id = ANY($1::uuid[])
"""


class RetrievalEngine:
    """Full retrieval pipeline: parse → expand → search → fuse → rerank.

    Inject all dependencies so each component is independently testable.
    """

    def __init__(
        self,
        parser: QueryParser,
        expander: QueryExpander,
        dense: DenseSearcher,
        sparse: SparseSearcher,
        exact: ExactSearcher,
        fusion: RRFFusion,
        reranker: LLMReranker,
        statutes: StatuteRetriever,
        config: RetrievalConfig | None = None,
        cache: Any = None,
    ) -> None:
        self.parser = parser
        self.expander = expander
        self.dense = dense
        self.sparse = sparse
        self.exact = exact
        self.fusion = fusion
        self.reranker = reranker
        self.statutes = statutes
        self.config = config or RetrievalConfig()
        self.cache = cache

    async def close(self) -> None:
        """Release any owned resources held by the engine."""
        if self.cache is not None:
            await self.cache.close()

    async def search(
        self,
        query: str,
        motion_type: str | None = None,
        court_filter: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        area_of_law: str | None = None,
        max_results: int = 10,
        include_statutes: bool = True,
    ) -> dict:
        """Execute full retrieval pipeline and return API-ready response dict."""
        from db import get_connection

        t_start = time.perf_counter()
        stage_timings_ms: dict[str, int] = {}
        stage_counts: dict[str, int] = {}
        cache_status = {
            "parsed_query": False,
            "candidate_results": False,
        }

        def _elapsed_ms(start: float) -> int:
            return int((time.perf_counter() - start) * 1000)

        # ── Step 1: Parse query ────────────────────────────────────────────────
        parse_started = time.perf_counter()
        parsed: ParsedQuery | None = None
        if self.cache is not None:
            cached_parsed = await self.cache.get_parsed(query)
            if isinstance(cached_parsed, dict):
                try:
                    parsed = ParsedQuery(**cached_parsed)
                    cache_status["parsed_query"] = True
                except TypeError:
                    parsed = None
        if parsed is None:
            parsed = await self.parser.parse(query)
            if self.cache is not None:
                await self.cache.set_parsed(query, asdict(parsed))
        stage_timings_ms["parse"] = _elapsed_ms(parse_started)

        # Allow caller to override detected motion type
        if motion_type:
            parsed.motion_type = motion_type

        # ── Step 2: Expand query ───────────────────────────────────────────────
        expand_started = time.perf_counter()
        expanded = await self.expander.expand(parsed)
        stage_timings_ms["expand"] = _elapsed_ms(expand_started)
        stage_counts["dense_variants"] = len(expanded.dense_embeddings)
        stage_counts["sparse_variants"] = len(expanded.sparse_texts)
        stage_counts["exact_case_references"] = min(len(parsed.case_references), 2)

        candidates: list[CandidateResult] | None = None
        authority_scores: dict[str, int] = {}

        cache_key: str | None = None
        if self.cache is not None:
            cache_lookup_started = time.perf_counter()
            cache_key = self.cache.candidates_key(
                query,
                {
                    "motion_type": parsed.motion_type,
                    "court_filter": court_filter,
                    "year_min": year_min,
                    "year_max": year_max,
                    "area_of_law": area_of_law,
                },
            )
            cached_candidates = await self.cache.get_candidates(cache_key)
            stage_timings_ms["candidate_cache_lookup"] = _elapsed_ms(cache_lookup_started)
            if isinstance(cached_candidates, list) and cached_candidates:
                try:
                    candidates = [CandidateResult(**item) for item in cached_candidates]
                    cache_status["candidate_results"] = True
                except TypeError:
                    candidates = None

        # ── Step 3: Stage 1 — parallel search ─────────────────────────────────
        async with get_connection() as conn:
            if candidates is None:
                search_started = time.perf_counter()
                search_tasks = []

                # Dense search — each task acquires its own connection so they can
                # run truly in parallel (asyncpg connections are not reentrant).
                for emb in expanded.dense_embeddings:
                    if emb:

                        async def _dense(e=emb):
                            async with get_connection() as _c:
                                return await self.dense.search(
                                    _c,
                                    e,
                                    limit=self.config.dense_limit,
                                    court_codes=court_filter,
                                    year_min=year_min,
                                    year_max=year_max,
                                )

                        search_tasks.append(_dense())

                # Sparse search — each task acquires its own connection
                for text in expanded.sparse_texts:

                    async def _sparse(t=text):
                        async with get_connection() as _c:
                            return await self.sparse.search(
                                _c,
                                t,
                                limit=self.config.sparse_limit,
                                court_codes=court_filter,
                                year_min=year_min,
                                year_max=year_max,
                            )

                    search_tasks.append(_sparse())

                # Exact search — each task acquires its own connection
                for ref in parsed.case_references[:2]:

                    async def _exact(r=ref):
                        async with get_connection() as _c:
                            return await self.exact.search(_c, r)

                    search_tasks.append(_exact())

                stage_counts["stage1_tasks"] = len(search_tasks)
                all_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                stage_timings_ms["stage1_search"] = _elapsed_ms(search_started)

                # Separate dense vs sparse vs exact results
                n_dense = len(expanded.dense_embeddings)
                n_sparse = len(expanded.sparse_texts)

                dense_results: list[list[dict]] = []
                sparse_results: list[list[dict]] = []

                for i, res in enumerate(all_results):
                    if isinstance(res, Exception):
                        logger.warning("engine.search_task_failed i=%d exc=%s", i, res)
                        res = []
                    if i < n_dense:
                        dense_results.append(res)  # type: ignore[arg-type]
                    elif i < n_dense + n_sparse:
                        sparse_results.append(res)  # type: ignore[arg-type]
                    else:
                        sparse_results.append(res)  # type: ignore[arg-type]

                # ── Step 4: Fetch authority scores ─────────────────────────────
                authority_started = time.perf_counter()
                all_candidate_ids = _collect_case_ids(dense_results + sparse_results)
                authority_scores = await self._fetch_authority_scores(conn, all_candidate_ids)
                stage_timings_ms["authority_lookup"] = _elapsed_ms(authority_started)

                # ── Step 5: Stage 2 — RRF fusion + boosting ───────────────────
                fusion_started = time.perf_counter()
                candidates = self.fusion.fuse(
                    dense_results=dense_results,
                    sparse_results=sparse_results,
                    authority_scores=authority_scores,
                    current_year=datetime.now().year,
                    top_n=self.config.top_candidates,
                )
                stage_timings_ms["fusion"] = _elapsed_ms(fusion_started)
                if self.cache is not None and cache_key is not None and candidates:
                    await self.cache.set_candidates(
                        cache_key,
                        [asdict(candidate) for candidate in candidates],
                    )
            else:
                stage_counts["stage1_tasks"] = 0
                authority_started = time.perf_counter()
                authority_scores = await self._fetch_authority_scores(
                    conn,
                    [candidate.case_id for candidate in candidates],
                )
                stage_timings_ms["authority_lookup"] = _elapsed_ms(authority_started)

            if not candidates:
                stage_timings_ms["total"] = _elapsed_ms(t_start)
                return _empty_response(
                    query,
                    parsed,
                    t_start,
                    stage_timings_ms=stage_timings_ms,
                    stage_counts=stage_counts,
                    cache_status=cache_status,
                )

            stage_counts["candidate_count"] = len(candidates)

            # ── Step 6: Fetch case metadata for candidates ─────────────────────
            metadata_started = time.perf_counter()
            candidate_case_ids = [c.case_id for c in candidates]
            case_metadata = await self._fetch_case_metadata(conn, candidate_case_ids)
            stage_timings_ms["metadata_lookup"] = _elapsed_ms(metadata_started)

            # Merge authority scores into metadata
            for cid, meta in case_metadata.items():
                meta["authority_score"] = authority_scores.get(cid, 0)
                meta["times_cited"] = authority_scores.get(cid, 0)

            # ── Step 7: Stage 3 — LLM reranking (+ statute retrieval in parallel)
            parallel_started = time.perf_counter()
            statute_embedding = expanded.dense_embeddings[0] if expanded.dense_embeddings else None
            rerank_task = self.reranker.rerank(
                query=query,
                candidates=candidates,
                case_metadata=case_metadata,
                top_n=self.config.top_results,
            )
            statute_task = (
                self.statutes.retrieve(conn, parsed.motion_type, statute_embedding)
                if include_statutes
                else _noop()
            )

            ranked_results, statute_results = await asyncio.gather(
                rerank_task, statute_task, return_exceptions=True
            )
            stage_timings_ms["rerank_and_statutes"] = _elapsed_ms(parallel_started)

        # Handle exceptions
        if isinstance(ranked_results, Exception):
            logger.error("engine.rerank_failed exc=%s", ranked_results)
            ranked_results = _fusion_fallback(candidates, case_metadata, max_results)
        if isinstance(statute_results, Exception):
            statute_results = []

        # Trim to max_results
        final_results: list[SearchResult] = ranked_results[:max_results]  # type: ignore

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        stage_timings_ms["total"] = elapsed_ms
        stage_counts["results_returned"] = len(final_results)
        logger.info(
            "engine.search_complete query=%.40s results=%d elapsed_ms=%d cache_parsed=%s cache_candidates=%s",
            query,
            len(final_results),
            elapsed_ms,
            cache_status["parsed_query"],
            cache_status["candidate_results"],
        )

        return {
            "cases": [_result_to_dict(r) for r in final_results],
            "statutes": statute_results or [],
            "query_analysis": {
                "detected_motion_type": parsed.motion_type,
                "detected_concepts": parsed.detected_concepts,
                "case_references_found": parsed.case_references,
            },
            "search_metadata": {
                "total_time_ms": elapsed_ms,
                "results_returned": len(final_results),
                "stage_timings_ms": stage_timings_ms,
                "stage_counts": stage_counts,
                "cache": cache_status,
            },
        }

    async def _fetch_authority_scores(self, conn, case_ids: list[str]) -> dict[str, int]:
        """Return {case_id: times_cited} for the given case IDs."""
        if not case_ids:
            return {}
        try:
            rows = await conn.fetch(_AUTHORITY_SQL, case_ids)
            return {str(r["case_id"]): int(r["times_cited"]) for r in rows}
        except Exception as exc:
            logger.debug("engine.authority_scores_failed exc=%s", exc)
            return {}

    async def _fetch_case_metadata(self, conn, case_ids: list[str]) -> dict[str, dict]:
        """Return {case_id: metadata_dict} for display in results."""
        if not case_ids:
            return {}
        try:
            rows = await conn.fetch(_CASE_METADATA_SQL, case_ids)
            return {
                str(r["case_id"]): {
                    "case_name": r["case_name"],
                    "citation": r["citation"],
                    "court": r["court"],
                    "year": r["year"],
                    "status": r["status"],
                }
                for r in rows
            }
        except Exception as exc:
            logger.debug("engine.case_metadata_failed exc=%s", exc)
            return {}


# ── Engine factory ─────────────────────────────────────────────────────────────


async def create_engine(
    enable_hyde: bool = False,
    enable_step_back: bool = False,
) -> RetrievalEngine:
    """Wire up all retrieval components from settings.

    Call once at startup; the engine is stateless and safe to reuse.
    """
    import anthropic
    import voyageai

    from config import settings
    from retrieval.cache import LegalRetrievalCache

    voyage_client = voyageai.AsyncClient(api_key=settings.embedding_api_key)  # type: ignore[attr-defined]
    anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    cache = LegalRetrievalCache(settings.redis_url)

    try:
        await cache.connect()
    except Exception as exc:
        logger.warning("engine.cache_connect_failed exc=%s", exc)
        cache = None

    parser = QueryParser(anthropic_client=anthropic_client)
    expander = QueryExpander(
        voyage_client=voyage_client,
        embedding_model=settings.embedding_model,
        anthropic_client=anthropic_client,
        cache=cache,
        enable_hyde=enable_hyde,
        enable_step_back=enable_step_back,
    )

    return RetrievalEngine(
        parser=parser,
        expander=expander,
        dense=DenseSearcher(),
        sparse=SparseSearcher(),
        exact=ExactSearcher(),
        fusion=RRFFusion(),
        reranker=LLMReranker(anthropic_client=anthropic_client),
        statutes=StatuteRetriever(),
        cache=cache,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _collect_case_ids(result_lists: list[list[dict]]) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for result_list in result_lists:
        for row in result_list:
            cid = row.get("case_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return ids


def _empty_response(
    query: str,
    parsed,
    t_start: float,
    stage_timings_ms: dict[str, int] | None = None,
    stage_counts: dict[str, int] | None = None,
    cache_status: dict[str, bool] | None = None,
) -> dict:
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    return {
        "cases": [],
        "statutes": [],
        "query_analysis": {
            "detected_motion_type": parsed.motion_type,
            "detected_concepts": parsed.detected_concepts,
            "case_references_found": parsed.case_references,
        },
        "search_metadata": {
            "total_time_ms": elapsed_ms,
            "results_returned": 0,
            "stage_timings_ms": stage_timings_ms or {},
            "stage_counts": stage_counts or {},
            "cache": cache_status or {},
        },
    }


def _fusion_fallback(candidates, case_metadata: dict, max_results: int) -> list[SearchResult]:
    """Return fusion-ordered results when reranker fails."""
    if not candidates:
        return []

    max_fusion = max((candidate.boosted_score for candidate in candidates), default=1.0)
    results: list[SearchResult] = []

    for candidate in candidates[:max_results]:
        meta = case_metadata.get(candidate.case_id, {})
        case_name = meta.get("case_name", "Unknown")
        citation = meta.get("citation")
        court = meta.get("court") or candidate.court
        year = meta.get("year") or candidate.year
        authority_score = meta.get("authority_score", 0)
        times_cited = meta.get("times_cited", 0)
        relevance_score = candidate.boosted_score / max_fusion if max_fusion > 0 else 0.0

        results.append(
            SearchResult(
                case_id=candidate.case_id,
                case_name=case_name,
                case_name_short=_short_name(case_name),
                citation=citation,
                court=court,
                year=year,
                relevance_score=min(relevance_score, 1.0),
                relevance_explanation="Fusion-only fallback result",
                authority_score=authority_score,
                times_cited=times_cited,
                matched_segment={
                    "type": candidate.segment_type,
                    "content": candidate.content[:500],
                },
                verification_status="verified",
            )
        )

    results.sort(key=lambda result: result.relevance_score, reverse=True)
    return results[:max_results]


def _result_to_dict(result: SearchResult) -> dict:
    return {
        "case_id": result.case_id,
        "case_name": result.case_name,
        "case_name_short": result.case_name_short,
        "citation": result.citation,
        "court": result.court,
        "year": result.year,
        "relevance_score": round(result.relevance_score, 4),
        "relevance_explanation": result.relevance_explanation,
        "verification_status": result.verification_status,
        "authority_score": result.authority_score,
        "times_cited": result.times_cited,
        "matched_segment": result.matched_segment,
    }


async def _noop() -> list:
    return []


def _short_name(case_name: str, max_length: int = 60) -> str:
    if len(case_name) <= max_length:
        return case_name
    return case_name[:max_length] + "..."
