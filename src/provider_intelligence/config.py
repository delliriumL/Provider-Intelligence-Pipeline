"""Configuration loading and path helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class AppSettings(BaseSettings):
    """Environment-backed settings for runtime overrides."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_mode: Literal["off", "auto", "force"] = Field(default="auto", alias="LLM_MODE")
    llm_api_base: str = Field(default="", alias="LLM_API_BASE")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_max_record_share: float = Field(default=0.08, alias="LLM_MAX_RECORD_SHARE")
    llm_min_risk_for_call: float = Field(default=0.70, alias="LLM_MIN_RISK_FOR_CALL")
    llm_min_conflict_for_call: float = Field(default=0.35, alias="LLM_MIN_CONFLICT_FOR_CALL")
    human_review_cost_per_record: float = Field(default=0.50, alias="HUMAN_REVIEW_COST_PER_RECORD")
    wrong_auto_update_cost: float = Field(default=10.0, alias="WRONG_AUTO_UPDATE_COST")
    missed_update_cost: float = Field(default=3.0, alias="MISSED_UPDATE_COST")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load merged configuration from YAML files and environment settings."""
    settings = AppSettings()
    config: dict[str, Any] = {
        "project_root": PROJECT_ROOT,
        "paths": {
            "data_dir": DATA_DIR,
            "sample_dir": DATA_DIR / "sample",
            "raw_dir": DATA_DIR / "raw",
            "processed_dir": DATA_DIR / "processed",
            "reference_dir": DATA_DIR / "reference",
            "outputs_dir": OUTPUTS_DIR,
            "config_dir": CONFIG_DIR,
        },
        "thresholds": _load_yaml(CONFIG_DIR / "thresholds.yaml"),
        "source_reliability": _load_yaml(CONFIG_DIR / "source_reliability.yaml"),
        "field_weights": _load_yaml(CONFIG_DIR / "field_weights.yaml"),
        "app": _load_yaml(CONFIG_DIR / "app_config.yaml"),
        "llm": _load_yaml(CONFIG_DIR / "llm_config.yaml"),
        "env": settings.model_dump(by_alias=False),
    }
    config["llm"]["mode"] = settings.llm_mode
    config["llm"]["api"] = {
        "base_url": settings.llm_api_base or config["llm"].get("api", {}).get("base_url", ""),
        "api_key": settings.llm_api_key or config["llm"].get("api", {}).get("api_key", ""),
        "model": settings.llm_model or config["llm"].get("api", {}).get("model", ""),
    }
    config["llm"]["gating"]["max_record_share"] = settings.llm_max_record_share
    config["llm"]["gating"]["min_risk_for_call"] = settings.llm_min_risk_for_call
    config["llm"]["gating"]["min_conflict_for_call"] = settings.llm_min_conflict_for_call
    config["costs"] = {
        "human_review_per_record": settings.human_review_cost_per_record,
        "wrong_auto_update": settings.wrong_auto_update_cost,
        "missed_update": settings.missed_update_cost,
    }
    return config


def get_path(key: str) -> Path:
    """Resolve a configured path by key."""
    config = load_config()
    return Path(config["paths"][key])


def ensure_outputs_dir() -> Path:
    """Create outputs directory if missing."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR
