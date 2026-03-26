# Legal AI Assistant

An AI-powered Nigerian legal motion drafting assistant. Given a legal issue and facts, it retrieves relevant Nigerian case law and generates complete court motion papers — Motion Paper, Supporting Affidavit, and Written Address — compliant with Federal High Court (Civil Procedure) Rules 2019.

---

## Architecture Overview

```
User Query / Facts
       ↓
  Retrieval Engine                    Generation Engine
  ─────────────────                   ─────────────────
  Query Parser (Haiku)                Issue Formulation (Sonnet)
  Query Expander                      Argument Generation (Sonnet)
  Dense Search (pgvector)             Citation Verification
  Sparse Search (tsvector)    ──→     Strength Assessment (Haiku)
  Exact Search (citation)             Supporting Sections (Haiku)
  RRF Fusion + Boosts                 Template Assembly
  LLM Reranker (Haiku)                     ↓
       ↓                          Motion Paper + Affidavit
  SearchResult[]                    + Written Address
```

**Stack:** Python 3.12 · FastAPI · PostgreSQL 16 + pgvector · Redis · Voyage AI (`voyage-law-2`) · Anthropic Claude

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Data Foundation (crawling, parsing, segmentation, embedding) | ✅ Complete |
| 1 | Retrieval Engine + API | ✅ Complete |
| 2 | Motion Drafting Generation | ✅ Complete |
| 3 | Frontend | Planned |

**Corpus (as of 2026-03-26):** 1,667 cases · 24,891 embedded segments
- NigeriaLII: 1,286 cases (SC, CA, Federal HC, Lagos HC)
- NWLR Online: 381 cases (60 parts discovered, ingestion ongoing)

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose
- Anthropic API key
- Voyage AI API key
- NWLR Online subscription (for corpus expansion)

---

## Setup

### 1. Install dependencies

```bash
uv sync --extra dev
uv run playwright install chromium
uv run pre-commit install
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

Required variables:

```env
DATABASE_URL=postgresql+asyncpg://legalai:legalai@localhost:5432/legalai
POSTGRES_DB=legalai
POSTGRES_USER=legalai
POSTGRES_PASSWORD=legalai
REDIS_URL=redis://localhost:6379/0
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_API_KEY=pa-...           # Voyage AI key
EMBEDDING_MODEL=voyage-law-2
NWLR_EMAIL=your@email.com          # NWLR Online subscription
NWLR_PASSWORD=your-password
```

### 3. Start infrastructure

```bash
make dev        # starts PostgreSQL 16 + pgvector + Redis via Docker Compose
make migrate    # applies Alembic migrations
```

### 4. Run tests

```bash
make test
# 469 unit tests across ingestion, retrieval, generation, and API layers
```

---

## Data Ingestion

The corpus is built from two sources:

### NigeriaLII (free, public)

```bash
# Crawl cases from a court (NGSC, NGCA, NGFCHC, NGLAHC)
uv run python scripts/crawl.py --court NGSC fetch --limit 100

# Parse fetched HTML
uv run python scripts/parse.py

# Load into PostgreSQL
uv run python scripts/load.py run

# Embed (Voyage AI)
uv run python scripts/embed.py run

# Build tsvector index
uv run python scripts/embed.py tsvector

# Show pipeline status
uv run python scripts/load.py status
```

### NWLR Online (subscription-based, ~15,507 cases across 2034 parts)

```bash
# Discover case IDs for a range of parts (saves to data/nwlr_manifest.json)
uv run python scripts/nwlr_crawl.py discover --parts-from 2000 --parts-to 2034

# Fetch metadata + HTML for all discovered cases
uv run python scripts/nwlr_crawl.py fetch

# Parse cached HTML into NWLR.jsonl
uv run python scripts/nwlr_crawl.py parse

# Load into PostgreSQL
uv run python scripts/nwlr_crawl.py load

# Embed into pgvector
uv run python scripts/nwlr_crawl.py embed

# Check pipeline status
uv run python scripts/nwlr_crawl.py status
```

**Rate limiting:** Cloudflare blocks rapid requests. Default rate limit is 3s with ±50% jitter. Use `--rate-limit 3.0` (already the default). A 403 response triggers an automatic 90s back-off + retry.

**Discovery strategy:** NWLR has no listing API. The crawler linearly scans pages 1–700 per part to find the first case, then chain-follows via `page_end + 1`. Null-probed pages are cached to disk, making interrupted runs resumable without redundant API calls.

---

## API

Start the development server:

```bash
uvicorn api.app:app --reload --port 8000
```

### Search endpoint

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "grounds for dismissal for want of jurisdiction",
    "limit": 10,
    "motion_type": "motion_to_dismiss"
  }'
```

### Generate endpoint

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "motion_type": "motion_to_dismiss",
    "facts": "The claimant filed suit in the Federal High Court challenging a state governor election...",
    "client_name": "Respondent",
    "counsel_name": "Chukwuemeka Obi Esq.",
    "suit_number": "FHC/ABJ/CS/001/2026"
  }'
```

Response includes:
- `motion_paper` — formal Motion on Notice
- `supporting_affidavit` — sworn affidavit with numbered paragraphs
- `written_address` — structured legal argument with cited cases
- `citations_used` — list of verified Nigerian case citations
- `strength_assessment` — AI assessment of argument strength

---

## Retrieval Pipeline

```
Query → QueryParser (Haiku/rules/regex)
      → QueryExpander
           dense: [original, step-back*¹, HyDE*²]
           sparse: [original, concepts, case_refs]
      → DenseSearcher (pgvector cosine)
        SparseSearcher (tsvector websearch)
        ExactSearcher (citation match)
      → RRFusion (k=60)
           + authority_score boost
           + court-tier boost (SC > CA > HC)
           + recency boost
      → LLMReranker (Haiku, 0.70×LLM + 0.30×fusion)
      → SearchResult[]
```

*¹ Step-back disabled by default (hurts MRR at <5,000 cases — re-enable at corpus scale)*
*² HyDE disabled by default (same reason)*

**4-layer Redis cache:**
- L1: Parsed query — 24h
- L2: Embeddings — 7 days
- L3: Candidates — 1h
- L4: Semantic (cosine ≥ 0.95) — 30 minutes

### Retrieval benchmark

```bash
# Evaluate raw embedding retrieval
uv run python scripts/evaluate.py run --benchmark data/quality/benchmark.jsonl

# Evaluate full engine pipeline
uv run python scripts/evaluate.py run-engine --benchmark data/quality/benchmark.jsonl
```

Current baseline (30-query benchmark, 1,667 cases):

| Variant | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|
| Full pipeline (no step-back/HyDE) | 13.4% | 16.9% | 11.3% |
| Raw embedding | 5.0% | 5.0% | 3.6% |

Scores are corpus-size-limited. Expected to improve significantly at >5,000 cases.

---

## Database

```bash
# Connect via psql
docker compose exec postgres psql -U legalai -d legalai

# Useful queries
SELECT court, COUNT(*) FROM cases GROUP BY court ORDER BY 2 DESC;
SELECT source, COUNT(*) FROM cases GROUP BY source;
SELECT COUNT(*) FROM case_segments WHERE embedding IS NOT NULL;
```

### Schema highlights

| Table | Purpose |
|---|---|
| `cases` | Full judgment metadata — citation, court, parties, judges, year, area_of_law, tsvector |
| `case_segments` | Chunked judgment text with `VECTOR(1024)` embeddings and retrieval weights |
| `citation_graph` | Resolved inter-case citations with authority scores |
| `statute_segments` | Legislation sections (Laws.Africa) with embeddings |
| `motion_templates` | Reference motion type templates |

---

## Database Migrations

```bash
make migrate                    # Apply all pending migrations
make migrate-down               # Roll back one migration
uv run alembic history          # Show migration history
uv run alembic revision --autogenerate -m "description"  # Create new migration
```

---

## Development

```bash
make lint       # Ruff linting
make fmt        # Ruff format + autofix
make typecheck  # Pyright type checking
make test       # Full test suite
```

### Project structure

```
src/
├── api/            # FastAPI app, routers, schemas
├── ingestion/      # Crawlers, parsers, segmenters, embedders, DB loaders
│   ├── sources/    # NigeriaLII, NWLR Online, CoA, Laws.Africa, PDF
│   ├── parsing/    # Text cleaning, metadata extraction
│   ├── citations/  # Citation extraction + graph building
│   ├── segmentation/  # 3-pass segmenter (structural → NLP → LLM)
│   ├── embedding/  # Chunking + Voyage AI embedding
│   └── loaders/    # AsyncPG bulk insert
├── retrieval/      # RAG engine (parser, expander, searcher, fusion, reranker, cache)
└── generation/     # Motion drafting pipeline (models, templates, verification)

scripts/            # Operational CLI tools
alembic/            # Database migrations
tests/              # 469 unit tests
data/               # Raw + processed data (gitignored)
```

---

## Environment Variables Reference

| Variable | Description | Required |
|---|---|---|
| `DATABASE_URL` | AsyncPG connection URL | Yes |
| `REDIS_URL` | Redis connection URL | Yes |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Yes |
| `EMBEDDING_API_KEY` | Voyage AI API key | Yes |
| `EMBEDDING_MODEL` | Voyage AI model (default: `voyage-law-2`) | Yes |
| `NWLR_EMAIL` | NWLR Online account email | For NWLR ingestion |
| `NWLR_PASSWORD` | NWLR Online account password | For NWLR ingestion |
| `PAYSTACK_SECRET_KEY` | Paystack payments (Phase 3) | No |
| `JWT_SECRET` | Auth token signing (Phase 3) | No |
| `SENTRY_DSN` | Error monitoring (production) | No |
