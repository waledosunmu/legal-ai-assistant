.PHONY: dev down migrate crawl embed evaluate test lint fmt typecheck install

# ── Local dev environment ──────────────────────────────────────────────────────
dev:
	docker-compose up -d
	@echo "✓ PostgreSQL + Redis running. Run 'make migrate' to apply schema."

down:
	docker-compose down

# ── Database ───────────────────────────────────────────────────────────────────
migrate:
	uv run alembic upgrade head

migrate-down:
	uv run alembic downgrade -1

migrate-history:
	uv run alembic history --verbose

# ── Data pipeline ──────────────────────────────────────────────────────────────
crawl:
	uv run python scripts/crawl.py $(ARGS)

embed:
	uv run python scripts/embed.py $(ARGS)

evaluate:
	uv run python scripts/evaluate.py $(ARGS)

# ── Testing ────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	uv run ruff check src/ tests/ scripts/

fmt:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

typecheck:
	uv run pyright src/

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	uv sync --extra dev
	uv run playwright install chromium
	uv run pre-commit install
