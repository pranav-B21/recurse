"""Configuration for Recurse: provider presets, user config file, overrides.

Provider is a config choice, not a code path — the engine constructs RLM
backend kwargs entirely from the resolved preset. API keys come from env vars
only and are never stored in the config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PRESETS: dict[str, dict] = {
    "groq": {
        "backend": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "root_model": "llama-3.3-70b-versatile",
        "sub_model": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY",
        "tpm_limit": 12000,  # free tier — engine guardrail uses this
        "max_output_chars": 4000,
    },
    "openai": {
        "backend": "openai",
        "base_url": None,
        "root_model": "gpt-5-mini",
        "sub_model": "gpt-5-nano",
        "api_key_env": "OPENAI_API_KEY",
        "tpm_limit": None,
        "max_output_chars": 20000,
    },
    "anthropic": {
        "backend": "anthropic",
        "base_url": None,
        "root_model": "claude-sonnet-4-6",
        "sub_model": "claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "tpm_limit": None,
        "max_output_chars": 20000,
    },
    "ollama": {
        # Local, free, no API key — Ollama exposes an OpenAI-compatible server
        # at localhost:11434/v1. Pull the model first, e.g. `ollama pull qwen3.5:35b-a3b`
        # (a fast MoE — only ~3B active params — that handles the RLM root protocol
        # well). Smaller local models flail more; the Mod #2 directive prompt helps.
        # Override root_model in ~/.recurse/config.yaml for whatever you have pulled.
        "backend": "openai",
        "base_url": "http://localhost:11434/v1",
        "root_model": "qwen3.5:35b-a3b",
        "sub_model": "same",  # single model — avoids the experimental other_backends path
        "api_key_env": None,  # local server ignores the key
        "tpm_limit": None,
        "max_output_chars": 8000,
    },
}

DEFAULT_CONFIG_PATH = Path.home() / ".recurse" / "config.yaml"

CONFIG_TEMPLATE = """\
# Recurse configuration. API keys are read from env vars only
# (GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY) — never stored here.

provider: groq              # groq (free, small) | openai (recommended) | anthropic | ollama (local, free)
# root_model: gpt-5-mini    # optional per-model overrides
# sub_model: same           # "same" forces single-model mode if other_backends misbehaves

max_iterations: 20
environment: local          # local | docker
verbose: true

storage_path: ~/.recurse/threads

ingest:
  exclude: [node_modules, .git, __pycache__, .venv, dist, build, vendor, "*.lock", "*.pyc"]
  max_file_size_kb: 500
  max_total_files: 5000
"""


@dataclass
class RecurseConfig:
    provider: str = "groq"  # current default; switch to "openai" once credits loaded
    root_model: str | None = None  # override preset
    sub_model: str | None = None  # override preset; "same" forces single-model mode
    max_iterations: int = 20
    environment: str = "local"  # local | docker
    verbose: bool = True
    storage_path: Path = field(default_factory=lambda: Path.home() / ".recurse" / "threads")
    ingest_exclude: list[str] = field(
        default_factory=lambda: [
            "node_modules", ".git", "__pycache__", ".venv", "dist", "build",
            "*.lock", "*.pyc", ".DS_Store", "vendor",
        ]
    )
    max_file_size_kb: int = 500
    max_total_files: int = 5000

    def resolved(self) -> dict:
        """Preset merged with overrides + API key from env.

        Raises a clear error naming the provider/env var when either is wrong,
        so the user never sees a cryptic auth failure from the backend.
        """
        if self.provider not in PRESETS:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Choose one of: {', '.join(sorted(PRESETS))}."
            )
        p = dict(PRESETS[self.provider])
        if self.root_model:
            p["root_model"] = self.root_model
        if self.sub_model:
            p["sub_model"] = self.sub_model

        key_env = p["api_key_env"]
        if key_env:
            api_key = os.environ.get(key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key for provider '{self.provider}': "
                    f"set the {key_env} environment variable. "
                    f"(export {key_env}=...)"
                )
            p["api_key"] = api_key
        else:
            # Local OpenAI-compatible server (e.g. Ollama) ignores the key but
            # the client still requires a non-empty string.
            p["api_key"] = "local"
        return p


def write_default_config(path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Write the commented default config file. Never overwrites an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> RecurseConfig:
    """~/.recurse/config.yaml over defaults; create file with commented defaults if absent."""
    write_default_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    cfg = RecurseConfig()
    for key in ("provider", "root_model", "sub_model", "max_iterations", "environment", "verbose"):
        if raw.get(key) is not None:
            setattr(cfg, key, raw[key])
    if raw.get("storage_path"):
        cfg.storage_path = Path(str(raw["storage_path"])).expanduser()

    ingest = raw.get("ingest") or {}
    if ingest.get("exclude") is not None:
        cfg.ingest_exclude = list(ingest["exclude"])
    if ingest.get("max_file_size_kb") is not None:
        cfg.max_file_size_kb = int(ingest["max_file_size_kb"])
    if ingest.get("max_total_files") is not None:
        cfg.max_total_files = int(ingest["max_total_files"])
    return cfg
