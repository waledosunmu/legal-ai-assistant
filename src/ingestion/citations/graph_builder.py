"""Citation graph construction: edge building, fuzzy resolution, authority scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import structlog

from ingestion.citations.extractor import CitationTreatmentClassifier, ExtractedCitation

logger = structlog.get_logger(__name__)

# Minimum fuzzy-match ratio (0-1) to consider two case names the same
_FUZZY_MATCH_THRESHOLD: float = 0.75

# Authority score damping factor (similar to PageRank's alpha)
_DAMPING: float = 0.85


@dataclass
class CitationEdge:
    """A directed edge in the citation graph."""

    citing_case_id: str  # The case that cites
    cited_case_id: str | None  # Resolved corpus ID (None if unknown)
    cited_case_name: str  # Name as it appears in the citing judgment
    cited_citation: str  # Raw citation string
    treatment: str  # followed / distinguished / overruled / mentioned
    context: str  # Sentence where the citation appears


class CitationGraphBuilder:
    """
    Build a citation graph from extracted citation objects.

    Responsibilities:
    - Convert ``ExtractedCitation`` objects into ``CitationEdge`` records.
    - Fuzzy-resolve cited case names against a known case registry
      (dict mapping case_id → case_name) to link edges where possible.
    - Compute simple authority scores (normalised in-degree) for cases
      that appear as cited targets.
    """

    def __init__(self) -> None:
        self._classifier = CitationTreatmentClassifier()

    def build_edges(
        self,
        citing_case_id: str,
        citations: list[ExtractedCitation],
        case_registry: dict[str, str],
    ) -> list[CitationEdge]:
        """
        Build citation edges for a single citing judgment.

        Args:
            citing_case_id: Unique ID of the judgment containing these citations.
            citations: List of ``ExtractedCitation`` objects from that judgment.
            case_registry: Mapping of ``{case_id: case_name}`` for all known
                cases in the corpus, used for fuzzy resolution.

        Returns:
            List of ``CitationEdge`` objects, one per unique citation.
        """
        edges: list[CitationEdge] = []
        seen: set[str] = set()

        for citation in citations:
            # Deduplicate by raw citation text within the same judgment
            if citation.raw_text in seen:
                continue
            seen.add(citation.raw_text)

            treatment = self._classifier.classify(citation.context)
            cited_case_id = self._resolve_to_case_id(citation, case_registry)

            edges.append(
                CitationEdge(
                    citing_case_id=citing_case_id,
                    cited_case_id=cited_case_id,
                    cited_case_name=citation.case_name or "",
                    cited_citation=citation.full_citation,
                    treatment=treatment,
                    context=citation.context,
                )
            )

        logger.debug(
            "citation_graph.edges_built",
            citing=citing_case_id,
            total=len(edges),
            resolved=sum(1 for e in edges if e.cited_case_id is not None),
        )
        return edges

    def compute_authority_scores(
        self,
        all_edges: list[CitationEdge],
    ) -> dict[str, float]:
        """
        Compute normalised authority scores from citation in-degree.

        Score = (number of times the case is cited) / (max citations in corpus).
        Cases that are never cited as a *target* are not included.

        This is a simple proxy for authority; a full PageRank calculation
        can replace it once the corpus is large enough.
        """
        in_degree: dict[str, int] = {}
        for edge in all_edges:
            if edge.cited_case_id is not None:
                in_degree[edge.cited_case_id] = in_degree.get(edge.cited_case_id, 0) + 1

        if not in_degree:
            return {}

        max_count = max(in_degree.values())
        return {case_id: round(count / max_count, 4) for case_id, count in in_degree.items()}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _resolve_to_case_id(
        self,
        citation: ExtractedCitation,
        case_registry: dict[str, str],
    ) -> str | None:
        """
        Attempt to resolve a citation to a known corpus case ID.

        Strategy (in order):
        1. Exact citation string match against registry values.
        2. Fuzzy case-name match using SequenceMatcher.

        Returns the case ID string on success, or ``None`` if the cited
        case is not found in the corpus.
        """
        if not case_registry:
            return None

        # 1. Try exact match on citation string
        target_citation = self._normalise(citation.raw_text)
        for case_id, case_name in case_registry.items():
            if target_citation in self._normalise(case_name):
                return case_id

        # 2. Fuzzy match on case name
        if not citation.case_name:
            return None

        target_name = self._normalise(citation.case_name)
        best_id: str | None = None
        best_ratio: float = 0.0

        for case_id, case_name in case_registry.items():
            ratio = SequenceMatcher(
                None,
                target_name,
                self._normalise(case_name),
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = case_id

        if best_ratio >= _FUZZY_MATCH_THRESHOLD:
            return best_id
        return None

    @staticmethod
    def _normalise(text: str) -> str:
        """Lower-case and collapse whitespace for comparison."""
        return re.sub(r"\s+", " ", text.lower().strip())
