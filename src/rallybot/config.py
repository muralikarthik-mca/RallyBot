"""Typed environment configuration for RallyBot."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load and validate settings from the environment and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    app_name: str = "RallyBot"
    app_env: Literal["development", "testing", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: str = "sqlite:///./data/rallybot.db"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    ollama_generation_timeout_seconds: float = Field(default=60.0, gt=0, le=300, allow_inf_nan=False)
    discord_bot_token: SecretStr | None = None
    discord_guild_id: int | None = Field(default=None, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, immutable application settings."""
    return Settings()
