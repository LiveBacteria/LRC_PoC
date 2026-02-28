"""Configuration loading utilities for LRC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ApiKeys:
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""


@dataclass(frozen=True)
class ModelDefaults:
    model_name: str = ""
    image_model_name: str = ""


@dataclass(frozen=True)
class AppConfig:
    defaults: ModelDefaults
    api_keys: ApiKeys
    source_path: Path


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded."""


def _coerce_str(value: Any) -> str:
    return str(value) if value is not None else ""


def load_config(base_dir: Path | None = None) -> AppConfig:
    """Load runtime config from models.yaml, falling back to models.template.yaml."""
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
    config_dir = root / "config"
    runtime_path = config_dir / "models.yaml"
    template_path = config_dir / "models.template.yaml"

    source_path = runtime_path if runtime_path.exists() else template_path
    if not source_path.exists():
        raise ConfigError(
            "Missing config/models.template.yaml. Add the tracked template config first."
        )

    with source_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    defaults_payload = payload.get("defaults") or {}
    api_keys_payload = payload.get("api_keys") or {}

    return AppConfig(
        defaults=ModelDefaults(
            model_name=_coerce_str(defaults_payload.get("model_name")),
            image_model_name=_coerce_str(defaults_payload.get("image_model_name")),
        ),
        api_keys=ApiKeys(
            google_api_key=_coerce_str(api_keys_payload.get("google_api_key")),
            openai_api_key=_coerce_str(api_keys_payload.get("openai_api_key")),
            anthropic_api_key=_coerce_str(api_keys_payload.get("anthropic_api_key")),
        ),
        source_path=source_path,
    )