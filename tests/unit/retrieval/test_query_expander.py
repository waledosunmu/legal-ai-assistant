"""Unit tests for src/retrieval/query_expander.py"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import TextBlock

from retrieval.models import ParsedQuery
from retrieval.query_expander import QueryExpander, _make_step_back


def _make_voyage(embeddings: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.embeddings = embeddings
    client = MagicMock()
    client.embed = AsyncMock(return_value=resp)
    return client


def _make_anthropic(text: str) -> MagicMock:
    msg = MagicMock(content=[TextBlock(type="text", text=text)])
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=msg)
    return client


def _parsed(
    original: str = "test query",
    motion_type: str | None = None,
    concepts: list[str] | None = None,
    case_refs: list[str] | None = None,
    step_back: str | None = None,
) -> ParsedQuery:
    return ParsedQuery(
        original=original,
        motion_type=motion_type,
        detected_concepts=concepts or [],
        case_references=case_refs or [],
        step_back_query=step_back,
    )


class TestDenseExpansion:
    @pytest.mark.asyncio
    async def test_always_includes_original(self) -> None:
        voyage = _make_voyage([[0.1] * 1024])
        expander = QueryExpander(voyage_client=voyage, anthropic_client=None)
        parsed = _parsed("motion to dismiss for want of jurisdiction")
        result = await expander.expand(parsed)
        assert parsed.original in result.dense_texts

    @pytest.mark.asyncio
    async def test_step_back_added_from_parsed(self) -> None:
        voyage = _make_voyage([[0.1] * 1024, [0.2] * 1024])
        expander = QueryExpander(voyage_client=voyage, anthropic_client=None, enable_step_back=True)
        parsed = _parsed("test", step_back="underlying legal principle")
        result = await expander.expand(parsed)
        assert "underlying legal principle" in result.dense_texts

    @pytest.mark.asyncio
    async def test_hyde_generated_when_anthropic_provided(self) -> None:
        voyage = _make_voyage([[0.1] * 1024, [0.2] * 1024, [0.3] * 1024])
        anthropic = _make_anthropic("The court held that jurisdiction requires...")
        expander = QueryExpander(
            voyage_client=voyage,
            anthropic_client=anthropic,
            enable_step_back=True,
            enable_hyde=True,
        )
        # Provide a step_back so we already have 2 variants before HyDE
        parsed = _parsed("grounds for dismissal", step_back="conditions for dismissal of a suit")
        result = await expander.expand(parsed)
        anthropic.messages.create.assert_called_once()
        assert len(result.dense_texts) == 3

    @pytest.mark.asyncio
    async def test_hyde_skipped_without_anthropic(self) -> None:
        voyage = _make_voyage([[0.1] * 1024, [0.2] * 1024])
        expander = QueryExpander(voyage_client=voyage, anthropic_client=None)
        parsed = _parsed("some query", motion_type="motion_to_dismiss")
        result = await expander.expand(parsed)
        assert len(result.dense_texts) <= 2

    @pytest.mark.asyncio
    async def test_dense_embeddings_match_texts(self) -> None:
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        voyage = _make_voyage(embeddings)
        expander = QueryExpander(voyage_client=voyage, anthropic_client=None)
        parsed = _parsed("test query", step_back="broader principle")
        result = await expander.expand(parsed)
        assert len(result.dense_embeddings) == len(result.dense_texts)

    @pytest.mark.asyncio
    async def test_voyage_called_with_query_input_type(self) -> None:
        voyage = _make_voyage([[0.1] * 1024])
        expander = QueryExpander(voyage_client=voyage)
        await expander.expand(_parsed("test"))
        call_kwargs = voyage.embed.call_args
        assert (
            call_kwargs.kwargs.get("input_type") == "query"
            or call_kwargs[1].get("input_type") == "query"
            or "query" in str(call_kwargs)
        )

    @pytest.mark.asyncio
    async def test_hyde_failure_falls_back_gracefully(self) -> None:
        voyage = _make_voyage([[0.1] * 1024, [0.2] * 1024])
        anthropic = MagicMock()
        anthropic.messages = MagicMock()
        anthropic.messages.create = AsyncMock(side_effect=Exception("API error"))
        expander = QueryExpander(voyage_client=voyage, anthropic_client=anthropic)
        parsed = _parsed("test query", step_back="broader principle")
        result = await expander.expand(parsed)  # should not raise
        assert len(result.dense_texts) >= 1

    @pytest.mark.asyncio
    async def test_max_three_dense_variants(self) -> None:
        voyage = _make_voyage([[0.1] * 1024] * 3)
        anthropic = _make_anthropic("hypothetical holding text here")
        expander = QueryExpander(voyage_client=voyage, anthropic_client=anthropic)
        parsed = _parsed("test", step_back="step back text")
        result = await expander.expand(parsed)
        assert len(result.dense_texts) <= 3

    @pytest.mark.asyncio
    async def test_embedding_cache_hit_skips_voyage_call(self) -> None:
        voyage = _make_voyage([[0.1] * 1024])
        cache = MagicMock()
        cache.get_embedding = AsyncMock(return_value=[0.9] * 1024)
        cache.set_embedding = AsyncMock()
        expander = QueryExpander(voyage_client=voyage, cache=cache)

        result = await expander.expand(_parsed("test query"))

        voyage.embed.assert_not_called()
        assert result.dense_embeddings[0] == [0.9] * 1024

    @pytest.mark.asyncio
    async def test_embedding_cache_miss_is_backfilled(self) -> None:
        voyage = _make_voyage([[0.1] * 1024])
        cache = MagicMock()
        cache.get_embedding = AsyncMock(return_value=None)
        cache.set_embedding = AsyncMock()
        expander = QueryExpander(voyage_client=voyage, cache=cache)

        await expander.expand(_parsed("test query"))

        voyage.embed.assert_called_once()
        cache.set_embedding.assert_called_once()


class TestSparseExpansion:
    @pytest.mark.asyncio
    async def test_original_always_in_sparse(self) -> None:
        voyage = _make_voyage([[0.1] * 1024])
        expander = QueryExpander(voyage_client=voyage)
        parsed = _parsed("application for injunction")
        result = await expander.expand(parsed)
        assert parsed.original in result.sparse_texts

    @pytest.mark.asyncio
    async def test_concept_keywords_added(self) -> None:
        voyage = _make_voyage([[0.1] * 1024, [0.2] * 1024])
        expander = QueryExpander(voyage_client=voyage)
        parsed = _parsed("test", concepts=["jurisdiction", "estoppel"])
        result = await expander.expand(parsed)
        assert len(result.sparse_texts) >= 2

    @pytest.mark.asyncio
    async def test_citation_added_as_third_variant(self) -> None:
        voyage = _make_voyage([[0.1] * 1024] * 3)
        expander = QueryExpander(voyage_client=voyage)
        parsed = _parsed("test", concepts=["jurisdiction"], case_refs=["(2000) 3 NWLR 1"])
        result = await expander.expand(parsed)
        assert any("NWLR" in s for s in result.sparse_texts)

    @pytest.mark.asyncio
    async def test_max_three_sparse_variants(self) -> None:
        voyage = _make_voyage([[0.1] * 1024] * 3)
        expander = QueryExpander(voyage_client=voyage)
        parsed = _parsed(
            "test", concepts=["jurisdiction", "estoppel"], case_refs=["(2000) 1 NWLR 5"]
        )
        result = await expander.expand(parsed)
        assert len(result.sparse_texts) <= 3


class TestStepBackGeneration:
    def test_step_back_uses_motion_type(self) -> None:
        parsed = _parsed(motion_type="motion_to_dismiss")
        result = _make_step_back(parsed)
        assert "dismissal" in result.lower() or "dismiss" in result.lower()

    def test_step_back_includes_concepts(self) -> None:
        parsed = _parsed(concepts=["jurisdiction", "estoppel"])
        result = _make_step_back(parsed)
        assert "jurisdiction" in result or "estoppel" in result

    def test_step_back_fallback_to_original(self) -> None:
        parsed = _parsed(original="the query", motion_type=None, concepts=[])
        result = _make_step_back(parsed)
        assert result == "the query"
