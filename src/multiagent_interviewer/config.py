"""Application configuration.

Settings are loaded once at startup, validated by Pydantic, and then read-only.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API keys
    mistral_api_key: str = Field(
        ...,
        description="API key for Mistral AI",
    )
    tavily_api_key: str | None = Field(
        None,
        description="API key for Tavily search (optional)",
    )

    # LLM settings
    llm_model: str = Field("mistral-large-latest")
    llm_temperature: float = Field(0.7, ge=0.0, le=2.0)
    llm_max_retries: int = Field(3, ge=0)

    # RAG settings
    embedding_model: str = Field("deepvk/USER-bge-m3")
    cross_encoder_model: str = Field("PitKoro/cross-encoder-ru-msmarco-passage")
    chunk_size: int = Field(500, gt=0)
    chunk_overlap: int = Field(100, ge=0)
    bm25_weight: float = Field(0.25, ge=0.0, le=1.0)

    # Paths
    data_dir: Path = Field(Path("data"))
    expert_csv: str = Field("expert_knowledge.csv")
    manager_csv: str = Field("manager_knowledge.csv")

    # Interview flow
    max_turns: int = Field(10, gt=0)
    min_turns_before_end: int = Field(8, gt=0)

    # Logging
    log_level: str = Field("INFO")

    @property
    def expert_csv_path(self) -> Path:
        """Full path to the expert knowledge CSV."""
        return self.data_dir / self.expert_csv

    @property
    def manager_csv_path(self) -> Path:
        """Full path to the manager knowledge CSV."""
        return self.data_dir / self.manager_csv


@lru_cache
def get_settings() -> Settings:
    """Return the application settings."""
    return Settings()  # type: ignore[call-arg]
