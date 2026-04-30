from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-backed runtime settings for the chatbot backend."""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    model_path: Path = Field(default=Path("backend/models"), alias="MODEL_PATH")
    default_model: str = Field(default="gemma-2b", alias="DEFAULT_MODEL")
    max_tokens: int = Field(default=512, alias="MAX_TOKENS", ge=64, le=4096)
    request_body_limit_bytes: int = Field(
        default=64 * 1024,
        alias="REQUEST_BODY_LIMIT_BYTES",
        ge=1024,
        le=1024 * 1024,
    )
    rate_limit_per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE", ge=1, le=500)
    skip_model_load: bool = Field(default=False, alias="SKIP_MODEL_LOAD")


class SettingsProvider:
    """Provides a cached singleton settings instance."""

    def __init__(self) -> None:
        """Initialize internal cache."""
        self._settings: AppSettings | None = None

    def get_settings(self) -> AppSettings:
        """Load and cache validated app settings.

        Returns:
            AppSettings: Validated runtime settings.
        """
        if self._settings is None:
            self._settings = AppSettings()
        return self._settings


settings_provider = SettingsProvider()
settings = settings_provider.get_settings()
