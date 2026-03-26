"""Segment-aware chunking for embedding: turns segmented judgments into EmbeddingChunks."""

from __future__ import annotations

from dataclasses import dataclass, field

from ingestion.segmentation.models import SegmentType


@dataclass
class EmbeddingChunk:
    """A single text chunk ready for embedding and vector storage."""

    chunk_id: str  # Unique: "{case_id}__{segment_type}_{index}"
    case_id: str
    segment_type: str  # Matches DB segment_type enum string
    content: str
    embedding: list[float] | None = None

    # Metadata stored alongside the vector for filtered retrieval
    court: str | None = None
    year: int | None = None
    area_of_law: list[str] = field(default_factory=list)
    case_name: str | None = None
    citation: str | None = None


# ── Retrieval weight per segment type ─────────────────────────────────────────
# Higher weight = more likely to appear at the top of search results.
_RETRIEVAL_WEIGHTS: dict[str, float] = {
    "ratio": 2.0,
    "holding": 1.8,
    "obiter": 1.3,
    "orders": 1.3,
    "issues": 1.2,
    "facts": 1.0,
    "analysis": 1.0,
    "caption": 0.5,
    "background": 0.7,
}


class LegalTextChunker:
    """
    Convert a segmented judgment dict into a list of ``EmbeddingChunk`` objects.

    Chunking strategy (highest retrieval value first):

    - **Ratio decidendi** → single chunk.
    - **Each holding** → one chunk (issue + determination + reasoning).
    - **Facts / Background** → single chunk, first 2,000 characters.
    - **Analysis sections** → overlapping word-based chunks of
      ``max_chunk_words`` words with ``overlap_words`` overlap.
    - Other segment types (caption, orders, obiter) → single chunks.

    Input format ``segmented_judgment`` dict keys:

    ``case_id``, ``court``, ``year``, ``area_of_law``, ``case_name``,
    ``citation``, ``ratio_decidendi``, ``holdings``, ``segments``
    (list of ``{segment_type, content}`` dicts).
    """

    def __init__(
        self,
        max_chunk_words: int = 512,
        overlap_words: int = 100,
    ) -> None:
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words

    def chunk(self, segmented_judgment: dict) -> list[EmbeddingChunk]:
        """
        Build all chunks for a single judgment.

        Returns chunks ordered: ratio → holdings → facts → analysis.
        """
        case_id = segmented_judgment["case_id"]
        meta = {
            "court": segmented_judgment.get("court"),
            "year": segmented_judgment.get("year"),
            "area_of_law": segmented_judgment.get("area_of_law") or [],
            "case_name": segmented_judgment.get("case_name"),
            "citation": segmented_judgment.get("citation"),
        }

        chunks: list[EmbeddingChunk] = []

        # 1. Ratio decidendi — highest priority
        ratio = segmented_judgment.get("ratio_decidendi")
        if ratio:
            chunks.append(self._make_chunk(case_id, "ratio", ratio, 0, meta))

        # 2. Holdings — one chunk per holding
        for i, holding in enumerate(segmented_judgment.get("holdings", [])):
            content = (
                f"Issue: {holding.get('issue', '')}\n"
                f"Determination: {holding.get('determination', '')}\n"
                f"Reasoning: {holding.get('reasoning', '')}"
            ).strip()
            chunks.append(self._make_chunk(case_id, "holding", content, i, meta))

        # 3. Other segments from the segments list
        facts_added = False
        for seg in segmented_judgment.get("segments", []):
            seg_type = seg.get("segment_type") if isinstance(seg, dict) else seg.segment_type.value
            content = (seg.get("content") or "") if isinstance(seg, dict) else (seg.content or "")

            if seg_type in (SegmentType.FACTS.value, SegmentType.BACKGROUND.value):
                if not facts_added:
                    chunks.append(self._make_chunk(case_id, "facts", content[:2000], 0, meta))
                    facts_added = True

            elif seg_type == SegmentType.ANALYSIS.value:
                analysis_chunks = self._split_long(case_id, "analysis", content, meta)
                chunks.extend(analysis_chunks)

            elif seg_type in (
                SegmentType.ORDERS.value,
                SegmentType.OBITER.value,
                SegmentType.CAPTION.value,
            ):
                chunks.append(self._make_chunk(case_id, seg_type, content, 0, meta))

        return chunks

    def retrieval_weight(self, segment_type: str) -> float:
        """Return the retrieval boost weight for a segment type."""
        return _RETRIEVAL_WEIGHTS.get(segment_type, 1.0)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_chunk(
        self,
        case_id: str,
        seg_type: str,
        content: str,
        index: int,
        meta: dict,
    ) -> EmbeddingChunk:
        return EmbeddingChunk(
            chunk_id=f"{case_id}__{seg_type}_{index}",
            case_id=case_id,
            segment_type=seg_type,
            content=content,
            **meta,
        )

    def _split_long(
        self,
        case_id: str,
        seg_type: str,
        text: str,
        meta: dict,
    ) -> list[EmbeddingChunk]:
        """Split long text into overlapping word-based chunks."""
        words = text.split()
        if len(words) <= self.max_chunk_words:
            return [self._make_chunk(case_id, seg_type, text, 0, meta)]

        chunks: list[EmbeddingChunk] = []
        start = 0
        idx = 0
        while start < len(words):
            end = start + self.max_chunk_words
            chunk_text = " ".join(words[start:end])
            chunks.append(self._make_chunk(case_id, seg_type, chunk_text, idx, meta))
            start = end - self.overlap_words
            idx += 1
        return chunks
