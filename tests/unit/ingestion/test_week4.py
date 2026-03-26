"""Tests for Phase 0 Week 4: chunker, embedder, db_loader, benchmark evaluator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.benchmark.builder import (
    BenchmarkQuery,
    EmbeddingEvaluator,
    NLRBBuilder,
)
from ingestion.embedding.chunker import EmbeddingChunk, LegalTextChunker
from ingestion.embedding.embedder import CorpusEmbedder
from ingestion.loaders.db_loader import BulkCaseLoader
from ingestion.segmentation.models import SegmentType

# ── LegalTextChunker ──────────────────────────────────────────────────────────


def _judgment(**kwargs) -> dict:
    """Build a minimal segmented judgment dict for testing."""
    defaults = {
        "case_id": "case_001",
        "case_name": "Malami v. Ohikhuare",
        "court": "NGSC",
        "year": 2020,
        "citation": "(2020) 1 NWLR (Pt.1748) 1",
        "area_of_law": ["land_law"],
        "ratio_decidendi": None,
        "holdings": [],
        "segments": [],
    }
    defaults.update(kwargs)
    return defaults


class TestLegalTextChunkerRatio:
    def test_ratio_creates_one_chunk(self):
        chunker = LegalTextChunker()
        j = _judgment(ratio_decidendi="The binding principle is X.")
        chunks = chunker.chunk(j)
        ratio_chunks = [c for c in chunks if c.segment_type == "ratio"]
        assert len(ratio_chunks) == 1

    def test_ratio_chunk_id_format(self):
        chunker = LegalTextChunker()
        j = _judgment(ratio_decidendi="Binding principle.")
        chunks = chunker.chunk(j)
        assert chunks[0].chunk_id == "case_001__ratio_0"

    def test_no_ratio_no_ratio_chunk(self):
        chunker = LegalTextChunker()
        j = _judgment(ratio_decidendi=None)
        chunks = chunker.chunk(j)
        assert not any(c.segment_type == "ratio" for c in chunks)


class TestLegalTextChunkerHoldings:
    def test_one_chunk_per_holding(self):
        chunker = LegalTextChunker()
        holdings = [
            {"issue": "Issue 1", "determination": "yes", "reasoning": "reason A"},
            {"issue": "Issue 2", "determination": "no", "reasoning": "reason B"},
        ]
        j = _judgment(holdings=holdings)
        chunks = chunker.chunk(j)
        holding_chunks = [c for c in chunks if c.segment_type == "holding"]
        assert len(holding_chunks) == 2

    def test_holding_content_contains_issue(self):
        chunker = LegalTextChunker()
        j = _judgment(holdings=[{"issue": "Standing", "determination": "yes", "reasoning": "r"}])
        chunks = chunker.chunk(j)
        holding = next(c for c in chunks if c.segment_type == "holding")
        assert "Standing" in holding.content

    def test_no_holdings_no_holding_chunks(self):
        chunker = LegalTextChunker()
        j = _judgment(holdings=[])
        chunks = chunker.chunk(j)
        assert not any(c.segment_type == "holding" for c in chunks)


class TestLegalTextChunkerSegments:
    def _make_seg(self, seg_type: str, content: str) -> dict:
        return {"segment_type": seg_type, "content": content}

    def test_facts_segment_produces_facts_chunk(self):
        chunker = LegalTextChunker()
        j = _judgment(segments=[self._make_seg("facts", "The plaintiff sued the defendant.")])
        chunks = chunker.chunk(j)
        assert any(c.segment_type == "facts" for c in chunks)

    def test_background_segment_produces_facts_chunk(self):
        chunker = LegalTextChunker()
        j = _judgment(
            segments=[self._make_seg("background", "The case arose from a land dispute.")]
        )
        chunks = chunker.chunk(j)
        assert any(c.segment_type == "facts" for c in chunks)

    def test_only_first_facts_segment_included(self):
        chunker = LegalTextChunker()
        segs = [
            self._make_seg("facts", "First facts."),
            self._make_seg("facts", "Second facts."),
        ]
        j = _judgment(segments=segs)
        chunks = chunker.chunk(j)
        facts_chunks = [c for c in chunks if c.segment_type == "facts"]
        assert len(facts_chunks) == 1

    def test_facts_content_truncated_to_2000_chars(self):
        chunker = LegalTextChunker()
        long_content = "word " * 1000  # 5000 chars
        j = _judgment(segments=[self._make_seg("facts", long_content)])
        chunks = chunker.chunk(j)
        facts = next(c for c in chunks if c.segment_type == "facts")
        assert len(facts.content) <= 2000

    def test_short_analysis_produces_one_chunk(self):
        chunker = LegalTextChunker()
        j = _judgment(segments=[self._make_seg("analysis", "Short analysis text.")])
        chunks = chunker.chunk(j)
        analysis = [c for c in chunks if c.segment_type == "analysis"]
        assert len(analysis) == 1

    def test_long_analysis_produces_multiple_chunks(self):
        chunker = LegalTextChunker(max_chunk_words=10, overlap_words=2)
        long_text = " ".join([f"word{i}" for i in range(30)])
        j = _judgment(segments=[self._make_seg("analysis", long_text)])
        chunks = chunker.chunk(j)
        analysis = [c for c in chunks if c.segment_type == "analysis"]
        assert len(analysis) > 1

    def test_overlapping_chunks_share_content(self):
        chunker = LegalTextChunker(max_chunk_words=5, overlap_words=2)
        text = "alpha beta gamma delta epsilon zeta eta theta"
        j = _judgment(segments=[self._make_seg("analysis", text)])
        chunks = chunker.chunk(j)
        analysis = [c for c in chunks if c.segment_type == "analysis"]
        if len(analysis) >= 2:
            # The last words of chunk 0 should appear at the start of chunk 1
            last_words_0 = set(analysis[0].content.split()[-2:])
            first_words_1 = set(analysis[1].content.split()[:2])
            assert last_words_0 & first_words_1  # overlap exists


class TestLegalTextChunkerMetadata:
    def test_metadata_preserved_in_chunks(self):
        chunker = LegalTextChunker()
        j = _judgment(
            ratio_decidendi="Principle.",
            court="NGSC",
            year=2021,
            citation="(2021) 1 SC 1",
            case_name="A v. B",
            area_of_law=["contract"],
        )
        chunks = chunker.chunk(j)
        chunk = chunks[0]
        assert chunk.court == "NGSC"
        assert chunk.year == 2021
        assert chunk.citation == "(2021) 1 SC 1"
        assert chunk.case_name == "A v. B"
        assert chunk.area_of_law == ["contract"]

    def test_empty_judgment_returns_no_chunks(self):
        chunker = LegalTextChunker()
        j = _judgment()
        chunks = chunker.chunk(j)
        assert chunks == []


class TestLegalTextChunkerRetrievalWeight:
    def test_ratio_has_highest_weight(self):
        chunker = LegalTextChunker()
        assert chunker.retrieval_weight("ratio") > chunker.retrieval_weight("analysis")

    def test_holding_higher_than_facts(self):
        chunker = LegalTextChunker()
        assert chunker.retrieval_weight("holding") > chunker.retrieval_weight("facts")

    def test_caption_has_low_weight(self):
        chunker = LegalTextChunker()
        assert chunker.retrieval_weight("caption") < 1.0


# ── CorpusEmbedder ────────────────────────────────────────────────────────────


def _make_voyage_client(embeddings: list[list[float]]) -> MagicMock:
    """Return a mock voyageai.AsyncClient."""
    mock_response = MagicMock()
    mock_response.embeddings = embeddings
    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=mock_response)
    return mock_client


def _make_chunk(chunk_id: str, embedding=None) -> EmbeddingChunk:
    return EmbeddingChunk(
        chunk_id=chunk_id,
        case_id="case_001",
        segment_type="analysis",
        content="Some legal text.",
        embedding=embedding,
    )


class TestCorpusEmbedder:
    @pytest.mark.asyncio
    async def test_embed_chunks_populates_embeddings(self):
        mock_client = _make_voyage_client([[0.1, 0.2, 0.3]])
        embedder = CorpusEmbedder(_client=mock_client)
        chunks = [_make_chunk("c_0")]
        result = await embedder.embed_chunks(chunks)
        assert result[0].embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_already_embedded_chunks_skipped(self):
        mock_client = _make_voyage_client([])
        embedder = CorpusEmbedder(_client=mock_client)
        pre_embedded = _make_chunk("c_0", embedding=[1.0, 2.0])
        result = await embedder.embed_chunks([pre_embedded])
        mock_client.embed.assert_not_called()
        assert result[0].embedding == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_batching_splits_into_multiple_api_calls(self):
        # batch_size=1, 3 chunks → 3 API calls
        mock_client = _make_voyage_client([])
        mock_client.embed = AsyncMock(
            side_effect=[
                MagicMock(embeddings=[[0.0]]),
                MagicMock(embeddings=[[1.0]]),
                MagicMock(embeddings=[[2.0]]),
            ]
        )
        embedder = CorpusEmbedder(_client=mock_client, batch_size=1)
        chunks = [_make_chunk(f"c_{i}") for i in range(3)]
        result = await embedder.embed_chunks(chunks)
        assert mock_client.embed.call_count == 3
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_result_count_matches_input(self):
        n = 5
        vecs = [[float(i)] for i in range(n)]
        mock_client = _make_voyage_client(vecs)
        embedder = CorpusEmbedder(_client=mock_client)
        chunks = [_make_chunk(f"c_{i}") for i in range(n)]
        result = await embedder.embed_chunks(chunks)
        assert len(result) == n

    @pytest.mark.asyncio
    async def test_embed_file_roundtrip(self, tmp_path):
        """Write chunks to disk, embed, read back — embedding should be populated."""
        chunk = _make_chunk("chunk_0")
        chunks_path = tmp_path / "input.jsonl"
        # Write chunk without embedding
        with chunks_path.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "case_id": chunk.case_id,
                        "segment_type": chunk.segment_type,
                        "content": chunk.content,
                        "embedding": None,
                        "court": None,
                        "year": None,
                        "area_of_law": [],
                        "case_name": None,
                        "citation": None,
                    }
                )
                + "\n"
            )

        mock_client = _make_voyage_client([[0.5, 0.6]])
        embedder = CorpusEmbedder(_client=mock_client)
        output_path = tmp_path / "output.jsonl"
        n = await embedder.embed_file(chunks_path, output_path)
        assert n == 1
        with output_path.open() as f:
            saved = json.loads(f.readline())
        assert saved["embedding"] == [0.5, 0.6]


# ── BulkCaseLoader mappings (pure, no DB) ─────────────────────────────────────


class TestBulkCaseLoaderMappings:
    def test_map_court_ngsc(self):
        assert BulkCaseLoader.map_court("NGSC") == "SUPREME_COURT"

    def test_map_court_ngca(self):
        assert BulkCaseLoader.map_court("NGCA") == "COURT_OF_APPEAL"

    def test_map_court_case_insensitive(self):
        assert BulkCaseLoader.map_court("ngsc") == "SUPREME_COURT"

    def test_map_court_unknown_raises(self):
        with pytest.raises(ValueError):
            BulkCaseLoader.map_court("UNKNOWN")

    def test_map_segment_type_ratio(self):
        assert BulkCaseLoader.map_segment_type(SegmentType.RATIO.value) == "RATIO"

    def test_map_segment_type_holding(self):
        assert BulkCaseLoader.map_segment_type(SegmentType.HOLDING.value) == "HOLDING"

    def test_map_segment_type_background_falls_back_to_analysis(self):
        assert BulkCaseLoader.map_segment_type(SegmentType.BACKGROUND.value) == "ANALYSIS"

    def test_map_segment_type_unknown_falls_back_to_analysis(self):
        assert BulkCaseLoader.map_segment_type("nonexistent_type") == "ANALYSIS"

    def test_map_segment_type_chunker_string_ratio(self):
        # LegalTextChunker produces lowercase "ratio", not enum value
        assert BulkCaseLoader.map_segment_type("ratio") == "RATIO"


# ── EmbeddingEvaluator ────────────────────────────────────────────────────────


class TestEmbeddingEvaluatorRecall:
    def test_perfect_recall(self):
        ev = EmbeddingEvaluator()
        assert ev._recall(["a", "b"], ["a", "b"]) == 1.0

    def test_zero_recall(self):
        ev = EmbeddingEvaluator()
        assert ev._recall(["c", "d"], ["a", "b"]) == 0.0

    def test_partial_recall(self):
        ev = EmbeddingEvaluator()
        assert ev._recall(["a", "c"], ["a", "b"]) == 0.5

    def test_empty_relevant(self):
        ev = EmbeddingEvaluator()
        assert ev._recall(["a"], []) == 0.0


class TestEmbeddingEvaluatorMRR:
    def test_first_result_relevant(self):
        ev = EmbeddingEvaluator()
        assert ev._reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_second_result_relevant(self):
        ev = EmbeddingEvaluator()
        assert ev._reciprocal_rank(["x", "a"], ["a"]) == 0.5

    def test_no_relevant_result(self):
        ev = EmbeddingEvaluator()
        assert ev._reciprocal_rank(["x", "y"], ["a"]) == 0.0


class TestEmbeddingEvaluatorNDCG:
    def test_perfect_ndcg(self):
        ev = EmbeddingEvaluator()
        result = ev._ndcg(["a", "b"], ["a", "b"], k=2)
        assert abs(result - 1.0) < 1e-6

    def test_zero_ndcg(self):
        ev = EmbeddingEvaluator()
        result = ev._ndcg(["x", "y"], ["a", "b"], k=2)
        assert result == 0.0

    def test_ndcg_higher_when_relevant_first(self):
        ev = EmbeddingEvaluator()
        good = ev._ndcg(["a", "x"], ["a"], k=2)
        poor = ev._ndcg(["x", "a"], ["a"], k=2)
        assert good > poor


class TestEmbeddingEvaluatorEvaluate:
    def test_evaluate_returns_aggregate_metrics(self):
        ev = EmbeddingEvaluator()
        queries = [
            BenchmarkQuery("q1", "query 1", ["a"]),
            BenchmarkQuery("q2", "query 2", ["b"]),
        ]
        # retrieve_fn always returns the relevant doc first
        result = ev.evaluate(
            queries,
            retrieve_fn=lambda q, k: ["a"] if "1" in q else ["b"],
            k=5,
        )
        assert result["recall_at_k"] == 1.0
        assert result["mrr"] == 1.0
        assert result["num_queries"] == 2

    def test_evaluate_empty_queries(self):
        ev = EmbeddingEvaluator()
        result = ev.evaluate([], retrieve_fn=lambda q, k: [], k=5)
        assert result["recall_at_k"] == 0.0
        assert result["num_queries"] == 0


# ── NLRBBuilder ───────────────────────────────────────────────────────────────


class TestNLRBBuilder:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "benchmark.jsonl"
        builder = NLRBBuilder(path)
        queries = [
            BenchmarkQuery("q1", "What is locus standi?", ["case_a", "case_b"], "constitutional"),
            BenchmarkQuery("q2", "Breach of contract remedies", ["case_c"], "contract"),
        ]
        builder.save(queries)
        loaded = builder.load()
        assert len(loaded) == 2
        assert loaded[0].query_id == "q1"
        assert loaded[1].relevant_case_ids == ["case_c"]

    def test_load_missing_file_returns_empty(self, tmp_path):
        builder = NLRBBuilder(tmp_path / "nonexistent.jsonl")
        assert builder.load() == []

    def test_append_adds_to_existing(self, tmp_path):
        path = tmp_path / "bench.jsonl"
        builder = NLRBBuilder(path)
        builder.save([BenchmarkQuery("q1", "query", ["c1"])])
        builder.append(BenchmarkQuery("q2", "query 2", ["c2"]))
        loaded = builder.load()
        assert len(loaded) == 2
        assert loaded[1].query_id == "q2"
