"""4-layer Redis cache for the retrieval engine.

Layers:
  L1 — parsed query (24 h)      key: ``qp:{sha256(query)}``
  L2 — query embedding (7 d)    key: ``emb:{sha256(text)}``
  L3 — pre-rerank candidates (1 h)  key: ``ret:{sha256(query+filters)}``
  L4 — semantic cache (30 m)    key: ``sem:{sha256(query)}``
       Stores (embedding, result) pairs; a hit requires cosine_sim ≥ 0.95

Usage::

    cache = LegalRetrievalCache(redis_url="redis://localhost:6379/0")
    await cache.connect()

    parsed = await cache.get_parsed("my query")
    if parsed is None:
        parsed = ...
        await cache.set_parsed("my query", parsed)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# TTLs in seconds
_TTL_L1 = 60 * 60 * 24        # 24 h  — parsed query
_TTL_L2 = 60 * 60 * 24 * 7    # 7 d   — embedding vector
_TTL_L3 = 60 * 60              # 1 h   — pre-rerank candidates
_TTL_L4 = 60 * 30              # 30 m  — semantic cache

_SEM_THRESHOLD = 0.95           # cosine similarity threshold for L4 hit
_SEM_MAX_ENTRIES = 500          # maximum stored semantic cache entries


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class LegalRetrievalCache:
    """Redis-backed 4-layer cache for the legal retrieval pipeline.

    Args:
        redis_url: URL passed to ``redis.asyncio.from_url()``.
    """

    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._redis: Any = None  # redis.asyncio.Redis, set on connect()

    async def connect(self) -> None:
        """Open the Redis connection pool. Call once at startup."""
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(self._url, decode_responses=True)

    async def close(self) -> None:
        """Close the Redis connection pool. Call at shutdown."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    # ── L1 — Parsed query ─────────────────────────────────────────────────────

    async def get_parsed(self, query: str) -> dict | None:
        """Return cached ParsedQuery fields as dict, or None on miss."""
        key = f"qp:{_sha(query)}"
        return await self._get_json(key)

    async def set_parsed(self, query: str, data: dict) -> None:
        key = f"qp:{_sha(query)}"
        await self._set_json(key, data, _TTL_L1)

    # ── L2 — Query embedding ──────────────────────────────────────────────────

    async def get_embedding(self, text: str) -> list[float] | None:
        key = f"emb:{_sha(text)}"
        raw = await self._get_json(key)
        return raw if isinstance(raw, list) else None

    async def set_embedding(self, text: str, embedding: list[float]) -> None:
        key = f"emb:{_sha(text)}"
        await self._set_json(key, embedding, _TTL_L2)

    # ── L3 — Pre-rerank candidates ────────────────────────────────────────────

    def candidates_key(self, query: str, filters: dict) -> str:
        """Deterministic cache key encoding query + search filters."""
        payload = json.dumps({"q": query, **filters}, sort_keys=True)
        return f"ret:{_sha(payload)}"

    async def get_candidates(self, cache_key: str) -> list[dict] | None:
        raw = await self._get_json(cache_key)
        return raw if isinstance(raw, list) else None

    async def set_candidates(self, cache_key: str, candidates: list[dict]) -> None:
        await self._set_json(cache_key, candidates, _TTL_L3)

    # ── L4 — Semantic cache ───────────────────────────────────────────────────

    async def get_semantic(
        self, query_embedding: list[float]
    ) -> list[dict] | None:
        """Return a cached search result if a semantically close query exists."""
        if self._redis is None:
            return None
        try:
            keys = await self._redis.keys("sem:*")
            for key in keys[:_SEM_MAX_ENTRIES]:
                raw = await self._redis.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)
                stored_emb = entry.get("embedding")
                if not stored_emb:
                    continue
                sim = _cosine(query_embedding, stored_emb)
                if sim >= _SEM_THRESHOLD:
                    logger.debug("cache.l4_hit sim=%.3f key=%s", sim, key)
                    return entry.get("result")
        except Exception as exc:
            logger.debug("cache.l4_get_failed exc=%s", exc)
        return None

    async def set_semantic(
        self, query: str, query_embedding: list[float], result: list[dict]
    ) -> None:
        key = f"sem:{_sha(query)}"
        entry = {"embedding": query_embedding, "result": result}
        await self._set_json(key, entry, _TTL_L4)

    # ── Invalidation ──────────────────────────────────────────────────────────

    async def invalidate_candidates_and_semantic(self) -> None:
        """Purge L3 + L4 caches (call when new cases are ingested)."""
        if self._redis is None:
            return
        try:
            l3_keys = await self._redis.keys("ret:*")
            l4_keys = await self._redis.keys("sem:*")
            keys_to_delete = l3_keys + l4_keys
            if keys_to_delete:
                await self._redis.delete(*keys_to_delete)
                logger.info("cache.invalidated count=%d", len(keys_to_delete))
        except Exception as exc:
            logger.warning("cache.invalidation_failed exc=%s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_json(self, key: str) -> Any:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.debug("cache.get_failed key=%s exc=%s", key, exc)
            return None

    async def _set_json(self, key: str, value: Any, ttl: int) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(key, json.dumps(value), ex=ttl)
        except Exception as exc:
            logger.debug("cache.set_failed key=%s exc=%s", key, exc)
