"""POST /api/v1/search route."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_engine():
    """Lazy import to avoid circular dep at module load time."""
    from api.app import get_engine
    return get_engine()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Execute a legal case search and return ranked results."""
    engine = _get_engine()
    try:
        result = await engine.search(
            query=request.query,
            motion_type=request.motion_type,
            court_filter=request.court_filter,
            year_min=request.year_min,
            year_max=request.year_max,
            area_of_law=request.area_of_law,
            max_results=request.max_results,
            include_statutes=request.include_statutes,
        )
    except Exception as exc:
        logger.error("search.handler_failed exc=%s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed. Please try again.")

    return SearchResponse(**result)
