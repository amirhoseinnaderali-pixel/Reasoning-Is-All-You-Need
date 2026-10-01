from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def get_api_keys(name: str) -> list[str]:
    plural = os.getenv(f"{name}_KEYS", "")
    singular = os.getenv(f"{name}_KEY", "")
    raw = plural or singular
    return [item.strip() for item in raw.split(",") if item.strip()]


def google_keys() -> list[str]:
    return get_api_keys("GOOGLE_API")


def ollama_keys() -> list[str]:
    return get_api_keys("OLLAMA_API")


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load experiment configs.")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data
