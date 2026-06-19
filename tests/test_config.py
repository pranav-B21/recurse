"""Tests for recurse.config: presets, yaml loading, override resolution."""

import pytest

from recurse.config import CONFIG_TEMPLATE, PRESETS, RecurseConfig, load_config


def test_load_config_creates_default_file(tmp_path):
    path = tmp_path / "config.yaml"
    cfg = load_config(path)
    assert path.exists()
    assert path.read_text() == CONFIG_TEMPLATE
    assert cfg.provider == "groq"
    assert cfg.max_iterations == 20
    assert "node_modules" in cfg.ingest_exclude


def test_load_config_applies_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "provider: openai\n"
        "sub_model: same\n"
        "max_iterations: 5\n"
        "verbose: false\n"
        "storage_path: ~/elsewhere\n"
        "ingest:\n"
        "  exclude: [foo]\n"
        "  max_file_size_kb: 10\n"
        "  max_total_files: 3\n"
    )
    cfg = load_config(path)
    assert cfg.provider == "openai"
    assert cfg.sub_model == "same"
    assert cfg.max_iterations == 5
    assert cfg.verbose is False
    assert cfg.storage_path.name == "elsewhere"
    assert cfg.ingest_exclude == ["foo"]
    assert cfg.max_file_size_kb == 10
    assert cfg.max_total_files == 3


def test_resolved_merges_preset_and_overrides(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    cfg = RecurseConfig(provider="groq", root_model="custom-root")
    p = cfg.resolved()
    assert p["api_key"] == "gsk_test"
    assert p["root_model"] == "custom-root"
    assert p["sub_model"] == PRESETS["groq"]["sub_model"]
    assert p["tpm_limit"] == 12000
    assert p["max_output_chars"] == 4000


def test_resolved_missing_key_names_env_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        RecurseConfig(provider="openai").resolved()


def test_resolved_unknown_provider():
    with pytest.raises(ValueError, match="nonexistent"):
        RecurseConfig(provider="nonexistent").resolved()


def test_resolved_local_provider_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = RecurseConfig(provider="ollama").resolved()
    assert p["api_key"] == "local"
    assert p["base_url"] == "http://localhost:11434/v1"
    assert p["tpm_limit"] is None
