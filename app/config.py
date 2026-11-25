import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    # LLM (any OpenAI-compatible API)
    llm_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", "not-set"))
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_max_concurrency: int = field(default_factory=lambda: int(_env("LLM_MAX_CONCURRENCY", "64")))

    # Redis
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    cache_ttl_seconds: int = field(default_factory=lambda: int(_env("CACHE_TTL_SECONDS", "86400")))

    # Token batching: small requests arriving within the window are merged into one LLM call
    batch_window_ms: float = field(default_factory=lambda: float(_env("BATCH_WINDOW_MS", "10")))
    batch_max_items: int = field(default_factory=lambda: int(_env("BATCH_MAX_ITEMS", "8")))
    batch_max_tokens: int = field(default_factory=lambda: int(_env("BATCH_MAX_TOKENS", "2000")))
    small_request_tokens: int = field(default_factory=lambda: int(_env("SMALL_REQUEST_TOKENS", "300")))

    # Ingestion
    ingest_workers: int = field(default_factory=lambda: int(_env("INGEST_WORKERS", "4")))


settings = Settings()
