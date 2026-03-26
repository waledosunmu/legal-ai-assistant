"""Nigerian Legal Retrieval Benchmark (NLRB) builder and evaluator."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkQuery:
    """A single retrieval benchmark query."""

    query_id: str
    query_text: str
    relevant_case_ids: list[str]  # Ground-truth relevant case IDs
    area_of_law: str | None = None
    notes: str | None = None


@dataclass
class EvaluationResult:
    """Retrieval metrics for a single query."""

    query_id: str
    recall_at_k: float  # Fraction of relevant docs in top-K
    reciprocal_rank: float  # 1 / rank of first relevant result (0 if none)
    ndcg_at_k: float  # Normalised discounted cumulative gain at K
    retrieved_ids: list[str] = field(default_factory=list)


class NLRBBuilder:
    """
    Build and persist the Nigerian Legal Retrieval Benchmark dataset.

    The benchmark is a collection of natural-language queries paired with
    known relevant case IDs.  It is stored as a JSONL file so it can be
    extended incrementally as the corpus grows.
    """

    def __init__(self, benchmark_path: Path) -> None:
        self.path = benchmark_path

    def load(self) -> list[BenchmarkQuery]:
        """Load existing benchmark queries from disk."""
        if not self.path.exists():
            return []
        queries: list[BenchmarkQuery] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                queries.append(
                    BenchmarkQuery(
                        query_id=data["query_id"],
                        query_text=data["query_text"],
                        relevant_case_ids=data["relevant_case_ids"],
                        area_of_law=data.get("area_of_law"),
                        notes=data.get("notes"),
                    )
                )
        logger.info("nlrb.loaded", path=str(self.path), count=len(queries))
        return queries

    def save(self, queries: list[BenchmarkQuery]) -> None:
        """Persist benchmark queries to disk (overwrites existing file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for q in queries:
                record = {
                    "query_id": q.query_id,
                    "query_text": q.query_text,
                    "relevant_case_ids": q.relevant_case_ids,
                    "area_of_law": q.area_of_law,
                    "notes": q.notes,
                }
                f.write(json.dumps(record) + "\n")
        logger.info("nlrb.saved", path=str(self.path), count=len(queries))

    def append(self, query: BenchmarkQuery) -> None:
        """Append a single query to the benchmark file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            record = {
                "query_id": query.query_id,
                "query_text": query.query_text,
                "relevant_case_ids": query.relevant_case_ids,
                "area_of_law": query.area_of_law,
                "notes": query.notes,
            }
            f.write(json.dumps(record) + "\n")


class EmbeddingEvaluator:
    """
    Evaluate embedding model quality against the NLRB benchmark.

    Pass a ``retrieve_fn`` callable that takes ``(query_text, k)`` and
    returns a list of ``case_id`` strings (top-K results).

    Metrics computed:
    - **Recall@K**: fraction of known-relevant cases appearing in top-K.
    - **MRR**: mean reciprocal rank of the first relevant result.
    - **NDCG@K**: normalised discounted cumulative gain at position K.
    """

    def evaluate(
        self,
        queries: list[BenchmarkQuery],
        retrieve_fn: Callable[[str, int], list[str]],
        k: int = 10,
    ) -> dict:
        """
        Run evaluation and return an aggregate metrics dict.

        Returns::

            {
                "recall_at_k":  float,   # mean across queries
                "mrr":          float,
                "ndcg_at_k":    float,
                "k":            int,
                "num_queries":  int,
                "per_query":    list[EvaluationResult],
            }
        """
        results: list[EvaluationResult] = []

        for q in queries:
            retrieved = retrieve_fn(q.query_text, k)
            results.append(
                EvaluationResult(
                    query_id=q.query_id,
                    recall_at_k=self._recall(retrieved, q.relevant_case_ids),
                    reciprocal_rank=self._reciprocal_rank(retrieved, q.relevant_case_ids),
                    ndcg_at_k=self._ndcg(retrieved, q.relevant_case_ids, k),
                    retrieved_ids=retrieved,
                )
            )

        n = len(results) or 1
        summary = {
            "recall_at_k": round(sum(r.recall_at_k for r in results) / n, 4),
            "mrr": round(sum(r.reciprocal_rank for r in results) / n, 4),
            "ndcg_at_k": round(sum(r.ndcg_at_k for r in results) / n, 4),
            "k": k,
            "num_queries": len(results),
            "per_query": results,
        }

        logger.info(
            "evaluator.done",
            recall=summary["recall_at_k"],
            mrr=summary["mrr"],
            ndcg=summary["ndcg_at_k"],
            k=k,
            queries=len(results),
        )
        return summary

    # ── Metric helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _recall(retrieved: list[str], relevant: list[str]) -> float:
        """Recall@K = |retrieved ∩ relevant| / |relevant|."""
        if not relevant:
            return 0.0
        hits = sum(1 for r in retrieved if r in relevant)
        return hits / len(relevant)

    @staticmethod
    def _reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
        """1 / (rank of first relevant result), or 0.0 if none found."""
        relevant_set = set(relevant)
        for rank, case_id in enumerate(retrieved, start=1):
            if case_id in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
        """
        NDCG@K using binary relevance (1 if relevant, 0 otherwise).

        DCG  = Σ rel_i / log2(i + 1)   for i in 1..K
        IDCG = Σ 1 / log2(i + 1)       for i in 1..min(|relevant|, K)
        NDCG = DCG / IDCG
        """
        import math

        relevant_set = set(relevant)
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, case_id in enumerate(retrieved[:k])
            if case_id in relevant_set
        )
        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0
