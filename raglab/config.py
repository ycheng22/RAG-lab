"""Settings for the whole series: one place for paths, model names and defaults.

Values come from environment variables or the repo's `.env` file (see `.env.example`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_repo_root() -> Path:
    """Walk up from this file (then from the cwd) until we find pyproject.toml."""
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "raglab").is_dir():
                return candidate
    return Path.cwd().resolve()


REPO_ROOT: Path = _find_repo_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str = Field("", validation_alias=AliasChoices("OPENAI_API_KEY"))
    chat_model: str = Field("gpt-6-luna", validation_alias=AliasChoices("RAGLAB_CHAT_MODEL"))
    reasoning_effort: str = Field("none", validation_alias=AliasChoices("RAGLAB_REASONING_EFFORT"))
    embed_model: str = Field("text-embedding-3-small", validation_alias=AliasChoices("RAGLAB_EMBED_MODEL"))
    store: Literal["chroma", "pgvector"] = Field("chroma", validation_alias=AliasChoices("RAGLAB_STORE"))
    database_url: str | None = Field(None, validation_alias=AliasChoices("DATABASE_URL"))
    full_run: bool = Field(False, validation_alias=AliasChoices("RAGLAB_FULL_RUN"))
    service_api_key: str | None = Field(None, validation_alias=AliasChoices("RAGLAB_SERVICE_API_KEY"))
    admin_api_key: str | None = Field(None, validation_alias=AliasChoices("RAGLAB_ADMIN_API_KEY"))
    repo_url: str | None = Field(None, validation_alias=AliasChoices("RAGLAB_REPO_URL"))
    notebook_budget_usd: float = Field(0.05, validation_alias=AliasChoices("RAGLAB_NOTEBOOK_BUDGET_USD"))

    # ── paths (derived from the repo root) ──────────────────────────────────
    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / "data"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def cache_dir(self) -> Path:
        return REPO_ROOT / ".cache"

    @property
    def chroma_dir(self) -> Path:
        return self.cache_dir / "chroma"

    @property
    def results_dir(self) -> Path:
        return REPO_ROOT / "results"

    @property
    def diagrams_dir(self) -> Path:
        return REPO_ROOT / "diagrams"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


FACTS_FILE: Path = REPO_ROOT / "data" / "_build" / "facts.yaml"

HOTEL_FIELDS = (
    "hotel_name",
    "city",
    "country",
    "continent",
    "star_rating",
    "price_per_night",
    "has_pool",
    "has_spa",
)


@lru_cache(maxsize=1)
def hotel_metadata() -> dict[str, dict]:
    """Per-hotel metadata (name, city, continent, stars, price, pool, spa), keyed by hotel name.

    Loaded from data/_build/facts.yaml, the single source of truth for the fictional corpus.
    """
    facts = yaml.safe_load(FACTS_FILE.read_text(encoding="utf-8"))
    table: dict[str, dict] = {}
    for hotel in facts["hotels"]:
        row = {
            "hotel_name": hotel["name"],
            "city": hotel["city"],
            "country": hotel["country"],
            "continent": hotel["continent"],
            "star_rating": int(hotel["star_rating"]),
            "price_per_night": int(hotel["price_per_night"]),
            "has_pool": bool(hotel["has_pool"]),
            "has_spa": bool(hotel["has_spa"]),
        }
        table[hotel["name"]] = row
    return table
