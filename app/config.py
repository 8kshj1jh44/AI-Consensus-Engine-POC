from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    openai_api_key: str
    anthropic_api_key: str
    anthropic_workspace_id: str | None = None

    gemini_model: str = "gemini-3.8-flash"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5-20251001"

    arbiter_model_provider: str = "anthropic"
    anthropic_arbiter_model: str = "claude-sonnet-5-5"
    openai_arbiter_model: str = "gpt-4o"

    request_timeout_seconds: float = 12.0
    arbiter_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
