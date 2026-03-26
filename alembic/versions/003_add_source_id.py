"""Add source_id column to cases for AKN slug tracking.

Adds a nullable, unique `source_id TEXT` column to `cases`. This stores the
AKN slug used by the ingestion pipeline (e.g. "akn_ng_judgment_ngsc_2017_23"),
making UUID↔slug translation a direct DB lookup rather than a fragile
citation cross-reference.

Populated separately by scripts/backfill_source_id.py after migration.

Revision ID: 003
Revises: 002
"""

import sqlalchemy as sa

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cases", sa.Column("source_id", sa.Text(), nullable=True))
    op.create_index("idx_cases_source_id", "cases", ["source_id"], unique=True)


def downgrade() -> None:
    op.drop_index("idx_cases_source_id", table_name="cases")
    op.drop_column("cases", "source_id")
