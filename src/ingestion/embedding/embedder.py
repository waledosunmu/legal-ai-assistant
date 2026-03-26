"""Batched, resumable corpus embedder using Voyage AI."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import structlog
import voyageai

from ingestion.embedding.chunker import EmbeddingChunk

logger = structlog.get_logger(__name__)


class CorpusEmbedder:
    """
    Embed ``EmbeddingChunk`` objects in batches using the Voyage AI API.

    Uses ``voyageai.AsyncClient`` so it fits naturally into the async
    ingestion pipeline.  The embedding model defaults to ``voyage-law-2``
    which is fine-tuned on legal text.

    Resumability: chunks whose ``embedding`` field is already populated
    are skipped automatically — re-run the pipeline at any time without
    paying for re-embedding.

    Args:
        model:       Voyage AI model identifier.
        api_key:     Voyage AI API key.  Falls back to ``VOYAGE_API_KEY``
                     environment variable if omitted.
        batch_size:  Number of texts per API call.  Voyage's rate limit
                     is 128 texts / request for ``voyage-law-2``.
        _client:     Optional pre-built ``voyageai.AsyncClient`` for tests.
    """

    def __init__(
        self,
        model: str = "voyage-law-2",
        api_key: str | None = None,
        batch_size: int = 128,
        sleep_between_batches: float = 0.0,
        _client: voyageai.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.sleep_between_batches = sleep_between_batches
        self._client: voyageai.AsyncClient = _client or voyageai.AsyncClient(
            api_key=api_key
        )

    async def embed_chunks(
        self,
        chunks: list[EmbeddingChunk],
    ) -> list[EmbeddingChunk]:
        """
        Return a new list of chunks with ``embedding`` populated.

        Already-embedded chunks (``embedding is not None``) are passed
        through unchanged — this is the resumability guarantee.
        """
        to_embed = [c for c in chunks if c.embedding is None]
        already_done = [c for c in chunks if c.embedding is not None]

        logger.info(
            "embedder.start",
            total=len(chunks),
            to_embed=len(to_embed),
            skipped=len(already_done),
            model=self.model,
        )

        embedded: list[EmbeddingChunk] = []
        for i in range(0, len(to_embed), self.batch_size):
            batch = to_embed[i : i + self.batch_size]
            texts = [c.content for c in batch]
            # Retry once on rate limit with a 60-second back-off
            for attempt in range(2):
                try:
                    vectors = await self._embed_batch(texts)
                    break
                except voyageai.error.RateLimitError:
                    if attempt == 0:
                        wait = 60
                        logger.warning("embedder.rate_limit", wait_seconds=wait, batch=i)
                        await asyncio.sleep(wait)
                    else:
                        raise
            for chunk, vec in zip(batch, vectors):
                embedded.append(replace(chunk, embedding=vec))
            logger.debug(
                "embedder.batch_done",
                batch_start=i,
                batch_end=min(i + self.batch_size, len(to_embed)),
            )
            if self.sleep_between_batches > 0 and i + self.batch_size < len(to_embed):
                await asyncio.sleep(self.sleep_between_batches)

        # Return in original order: already-done first, then newly embedded
        return already_done + embedded

    async def embed_file(
        self,
        chunks_path: Path,
        output_path: Path,
    ) -> int:
        """
        Embed all chunks in a JSONL file and write results to ``output_path``.

        Returns the number of newly embedded chunks.
        """
        import json

        chunks: list[EmbeddingChunk] = []
        if chunks_path.exists():
            with chunks_path.open(encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    chunks.append(
                        EmbeddingChunk(
                            chunk_id=data["chunk_id"],
                            case_id=data["case_id"],
                            segment_type=data["segment_type"],
                            content=data["content"],
                            embedding=data.get("embedding"),
                            court=data.get("court"),
                            year=data.get("year"),
                            area_of_law=data.get("area_of_law", []),
                            case_name=data.get("case_name"),
                            citation=data.get("citation"),
                        )
                    )

        result = await self.embed_chunks(chunks)
        newly_embedded = sum(
            1
            for orig, new in zip(chunks, result)
            if orig.embedding is None and new.embedding is not None
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for chunk in result:
                record = {
                    "chunk_id":     chunk.chunk_id,
                    "case_id":      chunk.case_id,
                    "segment_type": chunk.segment_type,
                    "content":      chunk.content,
                    "embedding":    chunk.embedding,
                    "court":        chunk.court,
                    "year":         chunk.year,
                    "area_of_law":  chunk.area_of_law,
                    "case_name":    chunk.case_name,
                    "citation":     chunk.citation,
                }
                f.write(json.dumps(record) + "\n")

        logger.info("embedder.file_done", path=str(output_path), newly_embedded=newly_embedded)
        return newly_embedded

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the Voyage AI API for a single batch."""
        response = await self._client.embed(
            texts,
            model=self.model,
            input_type="document",
        )
        return response.embeddings
