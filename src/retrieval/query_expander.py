"""Query expansion for the retrieval engine.

Generates multiple query variants for both dense (vector) and sparse (keyword)
search to improve recall via multi-query deduplication.

Dense variants (up to 3):
  1. Original query — embedded with input_type="query"
  2. Step-back query — broader restatement of the legal principle
  3. HyDE — a hypothetical relevant holding, embedded as a document surrogate

Sparse variants (up to 3):
  1. Extracted keywords (concatenation of detected concepts + case refs)
  2. Legal concept phrases from ParsedQuery.detected_concepts
  3. Exact citation string (if case references found)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.models import ExpandedQueries, ParsedQuery

logger = logging.getLogger(__name__)

# ── HyDE prompt template ───────────────────────────────────────────────────────

_HYDE_PROMPT = """You are a Nigerian legal expert. Write a brief (3–5 sentence) holding or ratio decidendi that a Nigerian court would write when ruling on the following legal issue. Be specific, cite the relevant legal principle, and write as if this is from an actual judgment.

Legal issue: {query}

Holding:"""


class QueryExpander:
    """Generate dense embeddings and sparse keyword variants from a ParsedQuery.

    Args:
        voyage_client: voyageai.AsyncClient instance (or compatible mock).
        embedding_model: Model name to use for embeddings.
        anthropic_client: Optional Anthropic client for HyDE generation.
            If None, HyDE is skipped (only 2 dense variants produced).
    """

    def __init__(
        self,
        voyage_client,
        embedding_model: str = "voyage-law-2",
        anthropic_client=None,
        cache: Any = None,
        enable_hyde: bool = False,
        enable_step_back: bool = False,
    ) -> None:
        self._voyage = voyage_client
        self._model = embedding_model
        self._anthropic = anthropic_client
        self._cache = cache
        self._enable_hyde = enable_hyde
        self._enable_step_back = enable_step_back

    async def expand(self, parsed: ParsedQuery) -> ExpandedQueries:
        """Return dense embeddings + sparse text variants."""
        dense_texts = await self._build_dense_texts(parsed)
        dense_embeddings = await self._embed_texts(dense_texts)
        sparse_texts = self._build_sparse_texts(parsed)

        return ExpandedQueries(
            dense_texts=dense_texts,
            dense_embeddings=dense_embeddings,
            sparse_texts=sparse_texts,
        )

    # ── Dense expansion ────────────────────────────────────────────────────────

    async def _build_dense_texts(self, parsed: ParsedQuery) -> list[str]:
        """Build up to 3 dense query texts."""
        texts = [parsed.original]

        # Variant 2: step-back (restatement of the underlying legal principle)
        if self._enable_step_back:
            step_back = parsed.step_back_query or _make_step_back(parsed)
            if step_back and step_back != parsed.original:
                texts.append(step_back)

        # Variant 3: HyDE — conditional on Anthropic client availability and flag
        if self._enable_hyde and self._anthropic and len(texts) < 3:
            hyde = await self._generate_hyde(parsed.original)
            if hyde:
                texts.append(hyde)

        return texts[:3]

    async def _generate_hyde(self, query: str) -> str | None:
        """Generate a hypothetical document embedding (HyDE) via Claude Haiku."""
        try:
            settings = _get_settings()
            message = await self._anthropic.messages.create(
                model=settings.extraction_model,
                max_tokens=256,
                messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
            )
            return message.content[0].text.strip()
        except Exception as exc:
            logger.debug("query_expander.hyde_failed exc=%s", exc)
            return None

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of query texts with Voyage AI (input_type='query')."""
        if not texts:
            return []

        embeddings: list[list[float] | None] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        if self._cache is not None:
            for index, text in enumerate(texts):
                cached = await self._cache.get_embedding(text)
                if cached is not None:
                    embeddings[index] = cached
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(index)
        else:
            uncached_texts = list(texts)
            uncached_indices = list(range(len(texts)))

        if not uncached_texts:
            return [embedding or [] for embedding in embeddings]

        try:
            response = await self._voyage.embed(
                uncached_texts,
                model=self._model,
                input_type="query",
            )
            for index, embedding in zip(uncached_indices, response.embeddings, strict=False):
                embeddings[index] = embedding
                if self._cache is not None:
                    await self._cache.set_embedding(texts[index], embedding)
        except Exception as exc:
            logger.error("query_expander.embed_failed exc=%s", exc)

        return [embedding or [] for embedding in embeddings]

    # ── Sparse expansion ───────────────────────────────────────────────────────

    def _build_sparse_texts(self, parsed: ParsedQuery) -> list[str]:
        """Build up to 3 sparse search texts."""
        texts: list[str] = []

        # Variant 1: original query (main websearch_to_tsquery input)
        texts.append(parsed.original)

        # Variant 2: concept keyword string
        if parsed.detected_concepts:
            # Use human-readable phrases instead of internal concept keys
            from retrieval.query_parser import LEGAL_CONCEPTS
            concept_phrases: list[str] = []
            for concept in parsed.detected_concepts[:5]:
                keywords = LEGAL_CONCEPTS.get(concept, [])
                if keywords:
                    concept_phrases.append(keywords[0])  # lead keyword per concept
            if concept_phrases:
                texts.append(" ".join(concept_phrases))

        # Variant 3: exact citation string (if any)
        if parsed.case_references and len(texts) < 3:
            texts.append(parsed.case_references[0])

        return texts[:3]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_step_back(parsed: ParsedQuery) -> str:
    """Generate a rule-based step-back query from parsed concepts/motion."""
    parts: list[str] = []
    if parsed.motion_type:
        motion_desc = {
            "motion_to_dismiss": "grounds for dismissal of a suit",
            "interlocutory_injunction": "conditions for granting interlocutory injunction",
            "stay_of_proceedings": "requirements for stay of proceedings",
            "summary_judgment": "requirements for summary judgment",
            "extension_of_time": "principles for granting extension of time",
        }.get(parsed.motion_type, "")
        if motion_desc:
            parts.append(motion_desc)
    if parsed.detected_concepts:
        parts.extend(parsed.detected_concepts[:2])
    if parsed.area_of_law:
        parts.append(parsed.area_of_law.replace("_", " "))
    return " ".join(parts) if parts else parsed.original


def _get_settings():
    from config import settings
    return settings
