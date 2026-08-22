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

    # staleness thresholds, in seconds, used by /health
    # a source overdue by more than this reports ok=false
    staleness_grace_seconds: int = 3600 * 6

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
