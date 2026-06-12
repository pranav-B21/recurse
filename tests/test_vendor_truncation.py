"""Tests for vendor Mod #1: env-configurable REPL output truncation.

Verifies the clamp in vendor/rlm/rlm/utils/parsing.py — the mod that prevents
huge REPL outputs from being fed back into the root conversation untruncated
(the observed cause of Groq's 413 rate_limit_exceeded).
"""

from rlm.core.types import CodeBlock, REPLResult, RLMIteration
from rlm.utils.parsing import format_iteration


def _iteration_with_stdout(stdout: str) -> RLMIteration:
    result = REPLResult(stdout=stdout, stderr="", locals={})
    return RLMIteration(
        prompt="p",
        response="```repl\nprint(x)\n```",
        code_blocks=[CodeBlock(code="print(x)", result=result)],
    )


def _repl_output(messages: list[dict]) -> str:
    assert messages[1]["role"] == "user"
    return messages[1]["content"]


def test_env_var_clamps_output(monkeypatch):
    monkeypatch.setenv("RLM_MAX_OUTPUT_CHARS", "4000")
    messages = format_iteration(_iteration_with_stdout("A" * 40_000))
    output = _repl_output(messages)
    assert len(output) < 4500  # 4000 + header + truncation notice
    assert "truncated" in output
    assert "slice or filter" in output  # steering feedback


def test_default_preserves_upstream_limit(monkeypatch):
    monkeypatch.delenv("RLM_MAX_OUTPUT_CHARS", raising=False)
    short = format_iteration(_iteration_with_stdout("B" * 1000))
    assert "truncated" not in _repl_output(short)

    long = format_iteration(_iteration_with_stdout("B" * 30_000))
    assert "truncated" in _repl_output(long)  # upstream 20K default still applies


def test_explicit_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("RLM_MAX_OUTPUT_CHARS", "10")
    messages = format_iteration(_iteration_with_stdout("C" * 100), max_character_length=5000)
    assert "truncated" not in _repl_output(messages)


def test_invalid_env_value_falls_back(monkeypatch):
    monkeypatch.setenv("RLM_MAX_OUTPUT_CHARS", "not-a-number")
    messages = format_iteration(_iteration_with_stdout("D" * 1000))
    assert "truncated" not in _repl_output(messages)
