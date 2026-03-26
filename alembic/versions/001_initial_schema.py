"""Initial schema — cases, segments, citations, statutes, templates, users, billing

Revision ID: 001
Revises:
Create Date: 2026-03-11
"""

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Extensions ────────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")  # fuzzy citation matching
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ── Enums ─────────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TYPE court_enum AS ENUM (
            'SUPREME_COURT',
            'COURT_OF_APPEAL',
            'FEDERAL_HIGH_COURT',
            'LAGOS_HIGH_COURT',
            'KANO_HIGH_COURT',
            'BAUCHI_HIGH_COURT',
            'BENUE_HIGH_COURT',
            'EBONYI_HIGH_COURT',
            'OTHER_STATE_HC'
        )
    """)

    op.execute("""
        CREATE TYPE case_status AS ENUM (
            'active',
            'overruled',
            'distinguished'
        )
    """)

    op.execute("""
        CREATE TYPE segment_type AS ENUM (
            'FACTS',
            'ISSUE',
            'HOLDING',
            'RATIO',
            'OBITER',
            'ORDER',
            'ANALYSIS',
            'CAPTION',
            'INTRODUCTION'
        )
    """)

    op.execute("""
        CREATE TYPE citation_treatment AS ENUM (
            'FOLLOWED',
            'DISTINGUISHED',
            'OVERRULED',
            'MENTIONED',
            'APPROVED',
            'CRITICISED'
        )
    """)

    op.execute("""
        CREATE TYPE opinion_type AS ENUM (
            'LEAD',
            'CONCURRING',
            'DISSENTING'
        )
    """)

    # ── cases ─────────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE cases (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            case_name       TEXT NOT NULL,
            citation        TEXT UNIQUE,
            lpelr_citation  TEXT,
            court           court_enum NOT NULL,
            judicial_division TEXT,
            year            INTEGER NOT NULL,
            judges          TEXT[],
            lead_judge      TEXT,
            area_of_law     TEXT[],
            full_text       TEXT NOT NULL,
            status          case_status NOT NULL DEFAULT 'active',
            overruled_by    UUID REFERENCES cases(id),
            source_url      TEXT,
            source          TEXT NOT NULL DEFAULT 'nigerialii',
            jurisdiction    TEXT NOT NULL DEFAULT 'NG',
            ingested_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_verified_at TIMESTAMP WITH TIME ZONE,
            search_vector   TSVECTOR
        )
    """)

    # ── case_segments ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE case_segments (
            id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            case_id          UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
            segment_type     segment_type NOT NULL,
            content          TEXT NOT NULL,
            embedding        vector(1536),
            content_tsv      TSVECTOR,
            issue_number     INTEGER,
            opinion_author   TEXT,
            opinion_type     opinion_type NOT NULL DEFAULT 'LEAD',
            retrieval_weight FLOAT NOT NULL DEFAULT 1.0,
            metadata         JSONB DEFAULT '{}'
        )
    """)

    # ── citation_graph ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE citation_graph (
            citing_case_id  UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
            cited_case_id   UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
            context         TEXT,
            treatment       citation_treatment NOT NULL DEFAULT 'MENTIONED',
            PRIMARY KEY (citing_case_id, cited_case_id)
        )
    """)

    # ── statute_segments ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE statute_segments (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            title       TEXT NOT NULL,
            short_title TEXT,
            section     TEXT,
            content     TEXT NOT NULL,
            embedding   vector(1536),
            year        INTEGER,
            status      TEXT NOT NULL DEFAULT 'in_force',
            jurisdiction TEXT NOT NULL DEFAULT 'NG',
            source      TEXT NOT NULL DEFAULT 'laws_africa'
        )
    """)

    # ── motion_templates ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE motion_templates (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            motion_type     TEXT NOT NULL,
            court           court_enum NOT NULL,
            template_json   JSONB NOT NULL,
            rules_reference TEXT[],
            jurisdiction    TEXT NOT NULL DEFAULT 'NG',
            version         INTEGER NOT NULL DEFAULT 1,
            active          BOOLEAN NOT NULL DEFAULT TRUE,
            UNIQUE (motion_type, court, jurisdiction, version)
        )
    """)

    # ── users ─────────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE users (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            email           TEXT UNIQUE NOT NULL,
            password_hash   TEXT NOT NULL,
            full_name       TEXT,
            nba_number      TEXT,
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_login_at   TIMESTAMP WITH TIME ZONE
        )
    """)

    # ── drafts ────────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE drafts (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            motion_type     TEXT NOT NULL,
            court           court_enum,
            case_facts      TEXT,
            selected_cases  UUID[],
            selected_statutes UUID[],
            generated_motion  JSONB,
            generated_affidavit JSONB,
            generated_address   JSONB,
            citations_verified  JSONB,
            audit_log       JSONB DEFAULT '[]',
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # ── draft_versions ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE draft_versions (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            draft_id    UUID NOT NULL REFERENCES drafts(id) ON DELETE CASCADE,
            version     INTEGER NOT NULL,
            content     JSONB NOT NULL,
            created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE (draft_id, version)
        )
    """)

    # ── feedback ──────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE feedback (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            draft_id    UUID REFERENCES drafts(id) ON DELETE SET NULL,
            user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            rating      INTEGER CHECK (rating BETWEEN 1 AND 5),
            comment     TEXT,
            created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # ── subscriptions ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE subscriptions (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            plan            TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active',
            motions_included INTEGER NOT NULL DEFAULT 0,
            motions_used    INTEGER NOT NULL DEFAULT 0,
            current_period_start TIMESTAMP WITH TIME ZONE,
            current_period_end   TIMESTAMP WITH TIME ZONE,
            paystack_sub_code TEXT,
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # ── motion_credits ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE motion_credits (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            credits     INTEGER NOT NULL DEFAULT 0,
            updated_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # ── transactions ──────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE transactions (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            amount_kobo     INTEGER NOT NULL,
            currency        TEXT NOT NULL DEFAULT 'NGN',
            type            TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            paystack_ref    TEXT UNIQUE,
            metadata        JSONB DEFAULT '{}',
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # ── Materialised view: case authority scores ───────────────────────────────
    op.execute("""
        CREATE MATERIALIZED VIEW case_authority_scores AS
        SELECT
            cited_case_id AS case_id,
            COUNT(*) AS times_cited,
            COUNT(*) AS authority_score,
            NOW() AS last_refreshed
        FROM citation_graph
        GROUP BY cited_case_id
        WITH DATA
    """)

    # ── Indexes ───────────────────────────────────────────────────────────────

    # Vector search — HNSW for fast approximate nearest neighbour
    op.execute("""
        CREATE INDEX idx_segments_embedding
        ON case_segments
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # Full-text search
    op.execute("CREATE INDEX idx_segments_tsv ON case_segments USING GIN (content_tsv)")
    op.execute("CREATE INDEX idx_cases_search ON cases USING GIN (search_vector)")

    # Metadata filters
    op.execute("CREATE INDEX idx_cases_court ON cases (court)")
    op.execute("CREATE INDEX idx_cases_year ON cases (year)")
    op.execute("CREATE INDEX idx_cases_status ON cases (status)")
    op.execute("CREATE INDEX idx_cases_jurisdiction ON cases (jurisdiction)")
    op.execute("CREATE INDEX idx_cases_area_of_law ON cases USING GIN (area_of_law)")

    # Citation graph lookups
    op.execute("CREATE INDEX idx_citation_cited ON citation_graph (cited_case_id)")
    op.execute("CREATE INDEX idx_citation_citing ON citation_graph (citing_case_id)")

    # Authority scores materialised view
    op.execute("CREATE UNIQUE INDEX idx_authority_scores_case ON case_authority_scores (case_id)")

    # Segments by case
    op.execute("CREATE INDEX idx_segments_case ON case_segments (case_id)")
    op.execute("CREATE INDEX idx_segments_type ON case_segments (segment_type)")

    # Statute search
    op.execute("""
        CREATE INDEX idx_statutes_embedding
        ON statute_segments
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # Trigram index on case_name for fuzzy citation matching
    op.execute("CREATE INDEX idx_cases_name_trgm ON cases USING GIN (case_name gin_trgm_ops)")


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS case_authority_scores")
    op.execute("DROP TABLE IF EXISTS transactions")
    op.execute("DROP TABLE IF EXISTS motion_credits")
    op.execute("DROP TABLE IF EXISTS subscriptions")
    op.execute("DROP TABLE IF EXISTS feedback")
    op.execute("DROP TABLE IF EXISTS draft_versions")
    op.execute("DROP TABLE IF EXISTS drafts")
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TABLE IF EXISTS motion_templates")
    op.execute("DROP TABLE IF EXISTS statute_segments")
    op.execute("DROP TABLE IF EXISTS citation_graph")
    op.execute("DROP TABLE IF EXISTS case_segments")
    op.execute("DROP TABLE IF EXISTS cases")
    op.execute("DROP TYPE IF EXISTS opinion_type")
    op.execute("DROP TYPE IF EXISTS citation_treatment")
    op.execute("DROP TYPE IF EXISTS segment_type")
    op.execute("DROP TYPE IF EXISTS case_status")
    op.execute("DROP TYPE IF EXISTS court_enum")
