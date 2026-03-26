#!/usr/bin/env python3
"""CLI entry point for embedding model evaluation.

Usage examples::

    # Run retrieval benchmark on embedded corpus
    python scripts/evaluate.py run --benchmark data/quality/benchmark.jsonl

    # Create a new benchmark query interactively
    python scripts/evaluate.py add-query

    # Show benchmark summary
    python scripts/evaluate.py status --benchmark data/quality/benchmark.jsonl
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import click
import structlog

from evaluation.benchmark.builder import (
    BenchmarkQuery,
    EmbeddingEvaluator,
    NLRBBuilder,
)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

_DEFAULT_BENCHMARK = Path("data/quality/benchmark.jsonl")


@click.group()
def cli() -> None:
    """Embedding evaluation CLI."""


# ── run ───────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--benchmark",
    default=str(_DEFAULT_BENCHMARK),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the NLRB benchmark JSONL file.",
)
@click.option("--k", default=10, show_default=True, type=int, help="Recall cutoff K.")
@click.option(
    "--segments-dir",
    default="data/processed/segments",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory containing *_embedded.jsonl files.",
)
def run(benchmark: Path, k: int, segments_dir: Path) -> None:
    """Evaluate retrieval quality against the benchmark."""
    builder = NLRBBuilder(benchmark)
    queries = builder.load()

    if not queries:
        click.echo("No benchmark queries found. Use 'add-query' to create some.")
        return

    # Build in-memory index from embedded chunk files
    import numpy as np

    all_chunks: list[dict] = []
    for path in sorted(Path(segments_dir).glob("*_embedded.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("embedding"):
                    all_chunks.append(data)

    if not all_chunks:
        click.echo("No embedded chunks found. Run 'embed run' first.")
        return

    # Translate benchmark relevant_case_ids (DB UUIDs) → AKN slugs using
    # cases.source_id, which was populated by scripts/backfill_source_id.py.
    import asyncio

    import asyncpg

    from config import settings

    async def _build_uuid_to_slug() -> dict[str, str]:
        all_uuids = {uid for q in queries for uid in q.relevant_case_ids}
        try:
            conn = await asyncpg.connect(settings.asyncpg_url)
            rows = await conn.fetch(
                "SELECT id::text, source_id FROM cases"
                " WHERE id = ANY($1::uuid[]) AND source_id IS NOT NULL",
                list(all_uuids),
            )
            await conn.close()
            return {str(r["id"]): r["source_id"] for r in rows}
        except Exception:
            return {}

    uuid_to_slug = asyncio.run(_build_uuid_to_slug())

    def _translate_ids(ids: list[str]) -> list[str]:
        return [uuid_to_slug.get(i, i) for i in ids]

    # Patch benchmark queries to use slug IDs for the embedding eval
    from dataclasses import replace as dc_replace

    translated_queries = [
        dc_replace(q, relevant_case_ids=_translate_ids(q.relevant_case_ids)) for q in queries
    ]

    # Build numpy matrix for cosine similarity search
    matrix = np.array([c["embedding"] for c in all_chunks], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-9)

    def retrieve_fn(query_text: str, top_k: int) -> list[str]:
        """Naive cosine similarity retrieval using in-memory numpy index."""
        import asyncio

        from config import settings
        from ingestion.embedding.embedder import CorpusEmbedder

        embedder = CorpusEmbedder(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
        )

        async def _embed_query() -> list[float]:
            response = await embedder._client.embed(
                [query_text],
                model=embedder.model,
                input_type="query",
            )
            return response.embeddings[0]

        q_vec = np.array(asyncio.run(_embed_query()), dtype=np.float32)
        q_vec /= max(np.linalg.norm(q_vec), 1e-9)
        scores = matrix @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [all_chunks[i]["case_id"] for i in top_indices]

    evaluator = EmbeddingEvaluator()
    results = evaluator.evaluate(translated_queries, retrieve_fn, k=k)

    # Remove per_query list for JSON output (too verbose)
    output = {k: v for k, v in results.items() if k != "per_query"}
    click.echo(json.dumps(output, indent=2))


# ── run-engine ───────────────────────────────────────────────────────────────


@cli.command("run-engine")
@click.option(
    "--benchmark",
    default=str(_DEFAULT_BENCHMARK),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the NLRB benchmark JSONL file.",
)
@click.option("--k", default=10, show_default=True, type=int, help="Recall cutoff K.")
@click.option(
    "--no-hyde", "disable_hyde", is_flag=True, default=False, help="Disable HyDE expansion."
)
@click.option(
    "--no-step-back",
    "disable_step_back",
    is_flag=True,
    default=False,
    help="Disable step-back expansion.",
)
def run_engine(benchmark: Path, k: int, disable_hyde: bool, disable_step_back: bool) -> None:
    """Evaluate the live retrieval engine against the benchmark."""
    import asyncio

    from evaluation.benchmark.builder import EmbeddingEvaluator, NLRBBuilder
    from retrieval.engine import create_engine

    queries = NLRBBuilder(benchmark).load()
    if not queries:
        click.echo("No benchmark queries found. Use 'add-query' to create some.")
        return

    async def _run() -> dict:
        engine = await create_engine(
            enable_hyde=not disable_hyde,
            enable_step_back=not disable_step_back,
        )

        async def _retrieve(query_text: str, top_k: int) -> list[str]:
            result = await engine.search(
                query=query_text,
                max_results=top_k,
                include_statutes=False,
            )
            return [case["case_id"] for case in result["cases"]]

        try:
            evaluator = EmbeddingEvaluator()
            per_query_results = []
            for query in queries:
                retrieved_ids = await _retrieve(query.query_text, k)
                per_query_results.append((query, retrieved_ids))

            def _retrieve_from_snapshot(query_text: str, top_k: int) -> list[str]:
                for query, retrieved_ids in per_query_results:
                    if query.query_text == query_text:
                        return retrieved_ids[:top_k]
                return []

            return evaluator.evaluate(queries, _retrieve_from_snapshot, k=k)
        finally:
            await engine.close()

    results = asyncio.run(_run())
    output = {key: value for key, value in results.items() if key != "per_query"}
    click.echo(json.dumps(output, indent=2))


# ── add-query ─────────────────────────────────────────────────────────────────


@cli.command("add-query")
@click.option(
    "--benchmark",
    default=str(_DEFAULT_BENCHMARK),
    show_default=True,
    type=click.Path(path_type=Path),
)
def add_query(benchmark: Path) -> None:
    """Interactively add a new benchmark query."""
    query_text = click.prompt("Query text")
    relevant_ids_raw = click.prompt("Relevant case IDs (comma-separated)")
    relevant_ids = [r.strip() for r in relevant_ids_raw.split(",") if r.strip()]
    area = click.prompt("Area of law (optional)", default="", show_default=False)
    notes = click.prompt("Notes (optional)", default="", show_default=False)

    query = BenchmarkQuery(
        query_id=str(uuid.uuid4()),
        query_text=query_text,
        relevant_case_ids=relevant_ids,
        area_of_law=area or None,
        notes=notes or None,
    )

    NLRBBuilder(benchmark).append(query)
    click.echo(f"Added query '{query.query_id}' to {benchmark}")


# ── status ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--benchmark",
    default=str(_DEFAULT_BENCHMARK),
    show_default=True,
    type=click.Path(path_type=Path),
)
def status(benchmark: Path) -> None:
    """Show benchmark summary."""
    builder = NLRBBuilder(Path(benchmark))
    queries = builder.load()
    by_area: dict[str, int] = {}
    for q in queries:
        area = q.area_of_law or "untagged"
        by_area[area] = by_area.get(area, 0) + 1

    summary = {"total_queries": len(queries), "by_area_of_law": by_area}
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli()
