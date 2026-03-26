from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://legalai:legalai@localhost:5432/legalai"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── AI APIs ───────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "voyage-law-2"  # Updated after Week 4 evaluation
    embedding_dimensions: int = 1024  # voyage-law-2 outputs 1024-dim vectors

    # Generation models
    generation_model: str = "claude-sonnet-4-6"
    extraction_model: str = "claude-haiku-4-5-20251001"

    # ── Payments ──────────────────────────────────────────────────────────────
    paystack_secret_key: str = ""
    paystack_public_key: str = ""

    # ── Analytics & Monitoring ─────────────────────────────────────────────────
    posthog_api_key: str = ""
    sentry_dsn: str = ""

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret: str = "change-me-in-production"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # ── NWLR Online ───────────────────────────────────────────────────────────
    nwlr_email: str = ""
    nwlr_password: str = ""

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Confidence threshold below which Layer 3 (LLM) is triggered in QueryParser
    llm_confidence_threshold: float = 0.5

    # ── App ───────────────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def asyncpg_url(self) -> str:
        """Return database URL formatted for raw asyncpg (without SQLAlchemy prefix)."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


settings = Settings()
