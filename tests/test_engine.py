"""Tests for recurse.engine: guardrail, context resolution, RLM construction."""

from types import SimpleNamespace

import pytest

import recurse.engine as engine_mod
from recurse.config import RecurseConfig
from recurse.engine import DIRECTIVE_ADDENDUM, RecurseEngine


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    return RecurseConfig(provider="groq", storage_path=tmp_path / "threads", verbose=False)


@pytest.fixture
def engine(config):
    return RecurseEngine(config)


class _ExplodingRLM:
    """Stands in for RLM to prove no client is ever constructed."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("RLM was constructed — guardrail should fire first")


def test_guardrail_refuses_oversized_context_without_api_call(engine, monkeypatch):
    monkeypatch.setattr(engine_mod, "RLM", _ExplodingRLM)
    big = "x" * 200_000  # ≈50K tokens >> 60% of Groq's 12K TPM
    with pytest.raises(RuntimeError, match="provider 'groq'"):
        engine.query("find the needle", "inline:" + big)


def test_guardrail_allows_small_context(engine):
    engine._guardrail("small context")  # must not raise


def test_guardrail_disabled_when_no_tpm_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    eng = RecurseEngine(RecurseConfig(provider="openai", storage_path=tmp_path))
    eng._guardrail("y" * 200_000)  # no limit → no raise


def test_resolve_context_inline(engine):
    assert engine._resolve_context("inline:hello world", "t") == "hello world"


def test_resolve_context_bare_text_is_inline(engine):
    assert engine._resolve_context("just some text", "t") == "just some text"


def test_resolve_context_thread(engine):
    d = engine.store.base_path / "t1"
    d.mkdir(parents=True)
    (d / "context.txt").write_text("stored context")
    assert engine._resolve_context("thread:t1", "ignored") == "stored context"


def test_resolve_context_path_ingests_into_thread(engine, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hi')")
    ctx = engine._resolve_context(f"path:{src}", "t2")
    assert "=== FILE: main.py ===" in ctx
    assert engine.store.has_thread("t2")


def test_resolve_context_bare_dir_ingests(engine, tmp_path):
    src = tmp_path / "src2"
    src.mkdir()
    (src / "a.txt").write_text("content here")
    ctx = engine._resolve_context(str(src), "t3")
    assert "=== FILE: a.txt ===" in ctx


def test_store_operations_need_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    eng = RecurseEngine(RecurseConfig(provider="groq", storage_path=tmp_path))
    assert eng.store.list_threads() == []  # no key required until a query runs
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        eng.query("q", "inline:ctx")


class _RecordingRLM:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingRLM.instances.append(self)


def test_build_rlm_wires_preset(engine, monkeypatch):
    monkeypatch.setattr(engine_mod, "RLM", _RecordingRLM)
    _RecordingRLM.instances = []
    import os

    rlm = engine._build_rlm()
    kw = rlm.kwargs
    assert os.environ["RLM_MAX_OUTPUT_CHARS"] == "4000"  # groq preset, drives vendor Mod #1
    assert kw["backend"] == "openai"
    assert kw["backend_kwargs"]["base_url"] == "https://api.groq.com/openai/v1"
    assert kw["backend_kwargs"]["model_name"] == "llama-3.3-70b-versatile"
    assert kw["other_backend_kwargs"][0]["model_name"] == "llama-3.1-8b-instant"
    assert kw["custom_system_prompt"].endswith(DIRECTIVE_ADDENDUM)


def test_build_rlm_sub_model_same_skips_other_backends(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(engine_mod, "RLM", _RecordingRLM)
    cfg = RecurseConfig(provider="groq", sub_model="same", storage_path=tmp_path)
    rlm = RecurseEngine(cfg)._build_rlm()
    assert "other_backends" not in rlm.kwargs


def test_build_rlm_falls_back_when_other_backends_rejected(engine, monkeypatch):
    class _OldRLM(_RecordingRLM):
        def __init__(self, **kwargs):
            if "other_backends" in kwargs:
                raise TypeError("unexpected keyword argument 'other_backends'")
            super().__init__(**kwargs)

    monkeypatch.setattr(engine_mod, "RLM", _OldRLM)
    with pytest.warns(UserWarning, match="single model"):
        rlm = engine._build_rlm()
    assert "other_backends" not in rlm.kwargs


def test_query_persists_turn(engine, monkeypatch):
    fake_result = SimpleNamespace(
        response="the answer", execution_time=1.5, usage_summary="usage"
    )

    class _FakeRLM:
        def __init__(self, **kwargs):
            pass

        def completion(self, prompt, root_prompt):
            assert prompt == "some context"
            assert root_prompt == "the question?"
            return fake_result

    monkeypatch.setattr(engine_mod, "RLM", _FakeRLM)
    result = engine.query("the question?", "inline:some context", thread_id="t9")
    assert result is fake_result

    turns = engine.store.get_turns("t9")
    assert len(turns) == 1
    assert turns[0]["query"] == "the question?"
    assert turns[0]["answer"] == "the answer"
    assert turns[0]["meta"]["provider"] == "groq"
