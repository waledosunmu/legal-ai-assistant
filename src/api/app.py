"""FastAPI application factory for the Legal AI Assistant API.

Usage::

    uv run uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers.search import router as search_router
from api.routers.generate import router as generate_router

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────

_engine = None
_pipeline = None


def get_engine():
    """Return the singleton RetrievalEngine (initialised during lifespan)."""
    if _engine is None:
        raise RuntimeError("Engine not initialised — did lifespan run?")
    return _engine


def get_pipeline():
    """Return the singleton MotionGenerationPipeline (initialised during lifespan)."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised — did lifespan run?")
    return _pipeline


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: warm up DB pool + build retrieval/generation engines. Tear-down: close."""
    global _engine, _pipeline

    logger.info("api.startup: initialising DB pool, retrieval engine, generation pipeline")
    from db import get_pool
    from retrieval.engine import create_engine
    from generation.pipeline import MotionGenerationPipeline
    from generation.verification import CitationVerifier
    from config import settings
    import anthropic

    pool = await get_pool()          # warm up asyncpg connection pool
    _engine = await create_engine()

    anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    verifier = CitationVerifier(db_pool=pool)
    _pipeline = MotionGenerationPipeline(
        anthropic_client=anthropic_client,
        verifier=verifier,
        generation_model=settings.generation_model,
        extraction_model=settings.extraction_model,
    )
    logger.info("api.startup: ready")

    yield

    logger.info("api.shutdown")
    if _engine is not None:
        await _engine.close()
    _engine = None
    _pipeline = None


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal AI Assistant API",
        version="0.1.0",
        description="Nigerian case law search powered by 3-stage retrieval + LLM reranking.",
        lifespan=lifespan,
    )

    app.include_router(search_router, prefix="/api/v1", tags=["search"])
    app.include_router(generate_router, prefix="/api/v1", tags=["generation"])

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
