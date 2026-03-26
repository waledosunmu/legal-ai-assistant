"""Fix embedding dimension 1536→1024 and add citation metadata column.

voyage-law-2 produces 1024-dimensional vectors, not 1536. This migration:
  1. Drops the HNSW indexes on embedding columns (required before type change)
  2. Alters case_segments.embedding and statute_segments.embedding to vector(1024)
  3. Rebuilds the HNSW indexes
  4. Adds a metadata JSONB column to citation_graph for storing cited_case_name
     and citation_text from unresolved edges

Revision ID: 002
Revises: 001
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Drop HNSW indexes before altering column types ─────────────────────────
    op.execute("DROP INDEX IF EXISTS idx_segments_embedding")
    op.execute("DROP INDEX IF EXISTS idx_statutes_embedding")

    # ── Fix embedding dimensions ───────────────────────────────────────────────
    # The columns are currently empty (no data loaded yet), so no USING clause
    # is needed for the cast — but we include it for safety in case of partial runs.
    op.execute(
        "ALTER TABLE case_segments "
        "ALTER COLUMN embedding TYPE vector(1024) "
        "USING NULL"  # existing rows have NULL embeddings; safe to truncate
    )
    op.execute(
        "ALTER TABLE statute_segments " "ALTER COLUMN embedding TYPE vector(1024) " "USING NULL"
    )

    # ── Rebuild HNSW indexes ───────────────────────────────────────────────────
    op.execute("""
        CREATE INDEX idx_segments_embedding ON case_segments
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute("""
        CREATE INDEX idx_statutes_embedding ON statute_segments
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ── Add metadata column to citation_graph ─────────────────────────────────
    op.execute("ALTER TABLE citation_graph " "ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_segments_embedding")
    op.execute("DROP INDEX IF EXISTS idx_statutes_embedding")

    op.execute(
        "ALTER TABLE case_segments " "ALTER COLUMN embedding TYPE vector(1536) " "USING NULL"
    )
    op.execute(
        "ALTER TABLE statute_segments " "ALTER COLUMN embedding TYPE vector(1536) " "USING NULL"
    )

    op.execute("""
        CREATE INDEX idx_segments_embedding ON case_segments
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute("""
        CREATE INDEX idx_statutes_embedding ON statute_segments
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    op.execute("ALTER TABLE citation_graph DROP COLUMN IF EXISTS metadata")
