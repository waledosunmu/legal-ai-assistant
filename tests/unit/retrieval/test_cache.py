"""Unit tests for src/retrieval/cache.py"""

from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval.cache import (
    LegalRetrievalCache,
    _cosine,
    _sha,
    _TTL_L1,
    _TTL_L2,
    _TTL_L3,
    _TTL_L4,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_cache() -> tuple[LegalRetrievalCache, MagicMock]:
    """Return a cache whose Redis client is fully mocked."""
    cache = LegalRetrievalCache(redis_url="redis://localhost:6379/0")
    redis_mock = MagicMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock()
    redis_mock.delete = AsyncMock()
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.aclose = AsyncMock()
    cache._redis = redis_mock
    return cache, redis_mock


# ── _sha helper ───────────────────────────────────────────────────────────────


class TestSha:
    def test_deterministic(self) -> None:
        assert _sha("hello") == _sha("hello")

    def test_different_inputs_different_hashes(self) -> None:
        assert _sha("abc") != _sha("xyz")

    def test_length_is_32(self) -> None:
        assert len(_sha("anything")) == 32


# ── _cosine helper ────────────────────────────────────────────────────────────


class TestCosine:
    def test_identical_vectors_return_one(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self) -> None:
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_magnitude_returns_zero(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


# ── L1 — Parsed query ─────────────────────────────────────────────────────────


class TestL1Parsed:
    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self) -> None:
        cache, redis = _make_cache()
        result = await cache.get_parsed("unknown query")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_hit_returns_data(self) -> None:
        cache, redis = _make_cache()
        data = {"motion_type": "motion_to_dismiss", "confidence": 0.8}
        redis.get = AsyncMock(return_value=json.dumps(data))
        result = await cache.get_parsed("test query")
        assert result == data

    @pytest.mark.asyncio
    async def test_set_uses_correct_key_and_ttl(self) -> None:
        cache, redis = _make_cache()
        data = {"motion_type": None}
        await cache.set_parsed("test query", data)
        key_used = redis.set.call_args[0][0]
        ttl_used = redis.set.call_args[1]["ex"]
        assert key_used == f"qp:{_sha('test query')}"
        assert ttl_used == _TTL_L1

    @pytest.mark.asyncio
    async def test_redis_error_returns_none(self) -> None:
        cache, redis = _make_cache()
        redis.get = AsyncMock(side_effect=Exception("Redis down"))
        result = await cache.get_parsed("query")
        assert result is None


# ── L2 — Embedding ────────────────────────────────────────────────────────────


class TestL2Embedding:
    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self) -> None:
        cache, redis = _make_cache()
        result = await cache.get_embedding("some text")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_float_list(self) -> None:
        cache, redis = _make_cache()
        emb = [0.1, 0.2, 0.3]
        redis.get = AsyncMock(return_value=json.dumps(emb))
        result = await cache.get_embedding("some text")
        assert result == pytest.approx(emb)

    @pytest.mark.asyncio
    async def test_set_uses_correct_key_and_ttl(self) -> None:
        cache, redis = _make_cache()
        await cache.set_embedding("some text", [0.1, 0.2])
        key_used = redis.set.call_args[0][0]
        ttl_used = redis.set.call_args[1]["ex"]
        assert key_used == f"emb:{_sha('some text')}"
        assert ttl_used == _TTL_L2

    @pytest.mark.asyncio
    async def test_non_list_cached_value_returns_none(self) -> None:
        cache, redis = _make_cache()
        redis.get = AsyncMock(return_value=json.dumps({"bad": "value"}))
        result = await cache.get_embedding("text")
        assert result is None


# ── L3 — Candidates ───────────────────────────────────────────────────────────


class TestL3Candidates:
    def test_candidates_key_deterministic(self) -> None:
        cache = LegalRetrievalCache("redis://localhost")
        k1 = cache.candidates_key("query", {"court_filter": ["NGSC"]})
        k2 = cache.candidates_key("query", {"court_filter": ["NGSC"]})
        assert k1 == k2

    def test_candidates_key_differs_on_filter_change(self) -> None:
        cache = LegalRetrievalCache("redis://localhost")
        k1 = cache.candidates_key("query", {"court_filter": ["NGSC"]})
        k2 = cache.candidates_key("query", {"court_filter": ["NGCA"]})
        assert k1 != k2

    def test_candidates_key_starts_with_ret(self) -> None:
        cache = LegalRetrievalCache("redis://localhost")
        assert cache.candidates_key("q", {}).startswith("ret:")

    @pytest.mark.asyncio
    async def test_get_candidates_miss(self) -> None:
        cache, redis = _make_cache()
        result = await cache.get_candidates("ret:abc123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_candidates_hit(self) -> None:
        cache, redis = _make_cache()
        stored = [{"case_id": "c1"}, {"case_id": "c2"}]
        redis.get = AsyncMock(return_value=json.dumps(stored))
        result = await cache.get_candidates("ret:abc123")
        assert result == stored

    @pytest.mark.asyncio
    async def test_set_candidates_uses_l3_ttl(self) -> None:
        cache, redis = _make_cache()
        await cache.set_candidates("ret:key1", [{"case_id": "c1"}])
        ttl_used = redis.set.call_args[1]["ex"]
        assert ttl_used == _TTL_L3

    @pytest.mark.asyncio
    async def test_non_list_candidates_returns_none(self) -> None:
        cache, redis = _make_cache()
        redis.get = AsyncMock(return_value=json.dumps("oops"))
        result = await cache.get_candidates("ret:key1")
        assert result is None


# ── L4 — Semantic cache ───────────────────────────────────────────────────────


class TestL4Semantic:
    @pytest.mark.asyncio
    async def test_no_keys_returns_none(self) -> None:
        cache, redis = _make_cache()
        redis.keys = AsyncMock(return_value=[])
        result = await cache.get_semantic([0.1] * 1024)
        assert result is None

    @pytest.mark.asyncio
    async def test_similar_query_returns_result(self) -> None:
        cache, redis = _make_cache()
        emb = [1.0, 0.0, 0.0]
        stored_result = [{"case_id": "c1"}]
        entry = json.dumps({"embedding": emb, "result": stored_result})
        redis.keys = AsyncMock(return_value=["sem:abc"])
        redis.get = AsyncMock(return_value=entry)

        # Querying with the exact same embedding → cosine = 1.0 (above 0.95)
        result = await cache.get_semantic(emb)
        assert result == stored_result

    @pytest.mark.asyncio
    async def test_dissimilar_query_returns_none(self) -> None:
        cache, redis = _make_cache()
        stored_emb = [1.0, 0.0, 0.0]
        query_emb = [0.0, 1.0, 0.0]  # orthogonal → cosine = 0.0
        entry = json.dumps({"embedding": stored_emb, "result": [{"case_id": "c1"}]})
        redis.keys = AsyncMock(return_value=["sem:abc"])
        redis.get = AsyncMock(return_value=entry)

        result = await cache.get_semantic(query_emb)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_semantic_stores_embedding_and_result(self) -> None:
        cache, redis = _make_cache()
        emb = [0.5, 0.5]
        result = [{"case_id": "c2"}]
        await cache.set_semantic("my query", emb, result)
        key_used = redis.set.call_args[0][0]
        ttl_used = redis.set.call_args[1]["ex"]
        stored = json.loads(redis.set.call_args[0][1])
        assert key_used == f"sem:{_sha('my query')}"
        assert ttl_used == _TTL_L4
        assert stored["embedding"] == emb
        assert stored["result"] == result

    @pytest.mark.asyncio
    async def test_redis_error_in_get_semantic_returns_none(self) -> None:
        cache, redis = _make_cache()
        redis.keys = AsyncMock(side_effect=Exception("Redis gone"))
        result = await cache.get_semantic([0.1, 0.2])
        assert result is None


# ── Invalidation ──────────────────────────────────────────────────────────────


class TestInvalidation:
    @pytest.mark.asyncio
    async def test_invalidation_deletes_l3_and_l4_keys(self) -> None:
        cache, redis = _make_cache()
        redis.keys = AsyncMock(side_effect=[
            ["ret:key1", "ret:key2"],   # L3 keys
            ["sem:key3"],               # L4 keys
        ])
        await cache.invalidate_candidates_and_semantic()
        redis.delete.assert_called_once_with("ret:key1", "ret:key2", "sem:key3")

    @pytest.mark.asyncio
    async def test_invalidation_no_keys_no_delete(self) -> None:
        cache, redis = _make_cache()
        redis.keys = AsyncMock(return_value=[])
        await cache.invalidate_candidates_and_semantic()
        redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidation_redis_error_does_not_raise(self) -> None:
        cache, redis = _make_cache()
        redis.keys = AsyncMock(side_effect=Exception("Redis timeout"))
        # Should not raise
        await cache.invalidate_candidates_and_semantic()


# ── No-Redis (disconnected) behaviour ─────────────────────────────────────────


class TestDisconnected:
    @pytest.mark.asyncio
    async def test_get_parsed_without_connect_returns_none(self) -> None:
        cache = LegalRetrievalCache("redis://localhost")
        # _redis is None — should silently return None
        result = await cache.get_parsed("query")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_parsed_without_connect_does_not_raise(self) -> None:
        cache = LegalRetrievalCache("redis://localhost")
        await cache.set_parsed("query", {"data": 1})

    @pytest.mark.asyncio
    async def test_get_semantic_without_connect_returns_none(self) -> None:
        cache = LegalRetrievalCache("redis://localhost")
        result = await cache.get_semantic([0.1, 0.2])
        assert result is None


# ── connect / close ───────────────────────────────────────────────────────────


class TestConnectClose:
    @pytest.mark.asyncio
    async def test_close_calls_aclose(self) -> None:
        cache, redis = _make_cache()
        await cache.close()
        redis.aclose.assert_called_once()
        assert cache._redis is None

    @pytest.mark.asyncio
    async def test_close_when_not_connected_does_nothing(self) -> None:
        cache = LegalRetrievalCache("redis://localhost")
        await cache.close()  # should not raise
