"""Service configuration, loaded from the environment / .env once at startup."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

import business_case as bc


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    groq_api_key: SecretStr
    groq_model: str = bc.DEFAULT_MODEL
    groq_timeout_seconds: float = 90.0
    groq_max_retries: int = 2

    # Guards against a caller pointing our API key at an arbitrary (expensive) model.
    allowed_models: tuple[str, ...] = (bc.DEFAULT_MODEL,)

    max_upload_bytes: int = 10 * 1024 * 1024
    max_grid_years: int = 40

    # Each fired risk trigger costs one serial LLM call, so /risks/assess can hold a
    # worker thread for a minute or more. 40 (the AnyIO default) starves quickly.
    thread_limit: int = 64


@lru_cache
def get_settings() -> Settings:
    return Settings()
