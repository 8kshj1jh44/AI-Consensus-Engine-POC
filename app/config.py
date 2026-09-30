from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    openai_api_key: str
    anthropic_api_key: str
    anthropic_workspace_id: str | None = None

    arbiter_model_provider: str = "anthropic"
    request_timeout_seconds: float = 12.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()