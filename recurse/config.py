"""Configuration loading for Recurse.

Loads from ~/.recurse/config.yaml if present, otherwise uses defaults.
Supports ${ENV_VAR} expansion in string fields.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} placeholders in strings."""
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class ModelConfig(BaseModel):
    root: str = "qwen3.5:35b-a3b"
    sub: str = "qwen3.5:9b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"


class EngineConfig(BaseModel):
    max_iterations: int = 15
    max_output_truncation: int = 50000
    thinking_mode_root: bool = True
    thinking_mode_sub: bool = False


class SandboxConfig(BaseModel):
    mode: str = "subprocess"  # subprocess | docker
    timeout_seconds: int = 30

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("subprocess", "docker"):
            raise ValueError(f"sandbox.mode must be 'subprocess' or 'docker', got '{v}'")
        return v


class StorageConfig(BaseModel):
    path: str = "~/.recurse/threads"
    max_cache_size_mb: int = 500


class IngestConfig(BaseModel):
    default_exclude: list[str] = [
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "*.pyc",
        "*.lock",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "*.egg-info",
    ]
    max_file_size_kb: int = 500
    max_total_files: int = 5000


class RecurseConfig(BaseModel):
    models: ModelConfig = ModelConfig()
    engine: EngineConfig = EngineConfig()
    sandbox: SandboxConfig = SandboxConfig()
    storage: StorageConfig = StorageConfig()
    ingest: IngestConfig = IngestConfig()

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "RecurseConfig":
        """Load config from file, falling back to defaults."""
        if config_path is None:
            config_path = Path("~/.recurse/config.yaml").expanduser()
        else:
            config_path = Path(config_path).expanduser()

        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        raw = _expand_env(raw)
        return cls.model_validate(raw)

    @property
    def storage_path(self) -> Path:
        return Path(self.storage.path).expanduser()

    @property
    def cache_path(self) -> Path:
        return self.storage_path
