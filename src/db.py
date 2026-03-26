"""asyncpg connection pool — used by ingestion pipeline and API layer."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
import structlog

from config import settings

logger = structlog.get_logger(__name__)

_pool: asyncpg.Pool | None = None


async def create_pool() -> asyncpg.Pool:
    """Create and return the shared connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.asyncpg_url,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        logger.info("db.pool_created", min_size=2, max_size=10)
    return _pool


async def close_pool() -> None:
    """Close the shared connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("db.pool_closed")


async def get_pool() -> asyncpg.Pool:
    """Return the shared pool, creating it if necessary."""
    if _pool is None:
        await create_pool()
    assert _pool is not None
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Acquire a connection from the pool as an async context manager."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn  # type: ignore[misc]
