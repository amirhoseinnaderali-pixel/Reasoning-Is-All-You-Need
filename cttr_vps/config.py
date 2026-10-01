from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def load_environment(env_file: str | None = None) -> None:
    """Load an optional dotenv file without requiring it."""
    path = Path(env_file or ".env")
    if load_dotenv is not None and path.exists():
        load_dotenv(path)


def _split_keys(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def get_google_api_keys() -> list[str]:
    load_environment()
    return _split_keys(os.getenv("GOOGLE_API_KEYS")) or _split_keys(os.getenv("GOOGLE_API_KEY"))


def get_ollama_api_keys() -> list[str]:
    load_environment()
    return _split_keys(os.getenv("OLLAMA_API_KEYS")) or _split_keys(os.getenv("OLLAMA_API_KEY"))


def require_google_api_key(index: int = 0) -> str:
    keys = get_google_api_keys()
    if index >= len(keys):
        raise RuntimeError(
            "No Google API key is configured. Set GOOGLE_API_KEY or GOOGLE_API_KEYS in the environment."
        )
    return keys[index]


def require_ollama_api_key(index: int = 0) -> str:
    keys = get_ollama_api_keys()
    if index >= len(keys):
        raise RuntimeError(
            "No Ollama API key is configured. Set OLLAMA_API_KEY or OLLAMA_API_KEYS in the environment."
        )
    return keys[index]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at {path}, got {type(data).__name__}")
    return data
