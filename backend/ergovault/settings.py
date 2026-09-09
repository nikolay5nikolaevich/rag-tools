"""Конфигурация. Нормативный источник — docs/CONTRACTS.md §9."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EV_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    profile: Literal["public", "pilot"] = "public"
    corpus_version: str = "v1"
    prompt_version: str = "p1"

    database_url: str = "postgresql://ergovault:ergovault@localhost:5433/ergovault"

    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    # cli = через подписку (Claude Code headless), ключ не нужен. api = Anthropic SDK.
    llm_backend: Literal["cli", "api"] = "cli"
    gen_model: str = "claude-opus-5"
    judge_model: str = "claude-sonnet-5"
    anthropic_api_key: str | None = None

    retrieval_mode: Literal["fts", "vector", "hybrid"] = "hybrid"
    top_k: int = 8
    rrf_k: int = 60
    weight_ru: float = 1.0
    weight_simple: float = 1.0

    @property
    def corpus_dir(self) -> Path:
        return REPO_ROOT / "corpus" / self.corpus_version

    @property
    def index_dir(self) -> Path:
        return self.corpus_dir / "index"

    @property
    def visibilities(self) -> tuple[str, ...]:
        """Какие источники видны при текущем профиле. Фильтр применяется в SQL."""
        return ("public",) if self.profile == "public" else ("public", "pilot")


@lru_cache
def get_settings() -> Settings:
    return Settings()
