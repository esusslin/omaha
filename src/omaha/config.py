"""Application settings. Everything injectable, nothing read from ambient globals."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: str = Field(default="local", description="local | staging | production")
    log_level: str = "INFO"

    # database
    database_url: str = "postgresql+psycopg://omaha:omaha_local_dev@localhost:5433/omaha"

    # storage for raw originals (Phase 1 keeps them on disk)
    data_dir: str = "./data"

    # scheduler — disable when running >1 replica, or two collectors race
    scheduler_enabled: bool = True

    # ingestion
    user_agent: str = "omaha/0.1 (+https://github.com/esusslin/omaha)"
    request_timeout_seconds: float = 30.0
    max_retries: int = 3

    # Minimum gap between requests to the same host. Conditional requests keep repeat
    # polls cheap, but a backfill walks dozens of *new* article URLs per club, where
    # ETags buy nothing — so pace them. One request per second per host is well under
    # what a club CDN notices, and the whole backfill is still bounded by article count.
    min_request_interval_seconds: float = 1.0

    # --- extraction (Phase 4) ---
    anthropic_api_key: str = ""
    """Empty means extraction is unavailable and the pipeline says so rather than
    failing mid-batch. Same shape as the embedding model being absent on the host."""
    extract_model: str = "claude-haiku-4-5-20251001"
    """Haiku: the task is span extraction against a fixed schema, not reasoning. At
    roughly two dollars for a full-corpus pass, re-running after a prompt change is
    cheap enough that `extractor_version` is a usable iteration loop rather than a
    ceremony."""
    extract_batch_size: int = 50
    """Chunks per scheduled run. Bounds the cost and the wall time of a single job so a
    backfill can't monopolise the hourly slot."""

    # staleness thresholds, in seconds, used by /health
    # a source overdue by more than this reports ok=false
    staleness_grace_seconds: int = 3600 * 6

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
