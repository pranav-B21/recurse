"""Tests for the RLM engine loop (uses mock Qwen client — no Ollama required)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from recurse.config import RecurseConfig
from recurse.engine.core import (
    RecurseEngine,
    RLMResult,
    _extract_code,
    _extract_final,
    _extract_final_var,
    _truncate,
)


# ── helper extraction tests ────────────────────────────────────────────────


def test_extract_final_simple():
    assert _extract_final("FINAL(42)") == "42"


def test_extract_final_multiline():
    text = "Some thinking...\nFINAL(The answer\nis 42)"
    assert _extract_final(text) == "The answer\nis 42"


def test_extract_final_none():
    assert _extract_final("No final here") is None


def test_extract_final_var():
    assert _extract_final_var("FINAL_VAR(result)") == "result"
    assert _extract_final_var("nothing") is None


def test_extract_code():
    text = "Here is code:\n```python\nprint('hello')\n```"
    assert _extract_code(text) == "print('hello')"


def test_extract_code_multiple_blocks():
    text = "```python\nx = 1\n```\nand\n```python\ny = 2\n```"
    code = _extract_code(text)
    assert "x = 1" in code
    assert "y = 2" in code


def test_extract_code_none():
    assert _extract_code("no code blocks here") is None


def test_truncate_short():
    assert _truncate("hello", 100) == "hello"


def test_truncate_long():
    text = "a" * 1000
    result = _truncate(text, 100)
    assert len(result) < 300  # truncated + marker
    assert "truncated" in result


# ── RecurseEngine tests ────────────────────────────────────────────────────


@pytest.fixture
def config():
    return RecurseConfig()


@pytest.fixture
def mock_engine(config, tmp_path):
    """Engine with mocked Qwen client and temp storage."""
    config.storage.path = str(tmp_path / "threads")
    engine = RecurseEngine(config)
    return engine


def run(coro):
    return asyncio.run(coro)


def test_engine_final_on_first_response(mock_engine):
    """Engine should return immediately when first response contains FINAL()."""
    mock_engine.qwen.root_completion = AsyncMock(return_value="FINAL(The answer is 42)")

    result = run(mock_engine.query(
        query="What is the answer?",
        context="The answer is 42.",
        thread_id="test",
    ))

    assert isinstance(result, RLMResult)
    assert "42" in result.answer
    assert result.iterations_used == 1
    mock_engine.qwen.root_completion.assert_called_once()


def test_engine_code_then_final(mock_engine):
    """Engine executes code on iteration 1, gets FINAL() on iteration 2."""
    responses = [
        "I'll search for it.\n```python\nprint(CONTEXT[:100])\n```",
        "FINAL(Found it: 42)",
    ]
    mock_engine.qwen.root_completion = AsyncMock(side_effect=responses)

    result = run(mock_engine.query(
        query="Find the number",
        context="The number is 42 in here.",
        thread_id="test",
    ))

    assert "42" in result.answer
    assert result.iterations_used == 2


def test_engine_final_var(mock_engine):
    """Engine resolves FINAL_VAR from sandbox globals."""
    async def code_then_final_var(system_prompt, messages):
        if len(messages) == 1:
            return "```python\nanswer = 'resolved from var'\n```"
        return "FINAL_VAR(answer)"

    mock_engine.qwen.root_completion = AsyncMock(side_effect=code_then_final_var)

    result = run(mock_engine.query(
        query="Get the answer",
        context="some context",
        thread_id="test",
    ))

    assert result.answer == "resolved from var"


def test_engine_max_iterations_triggers_force_final(mock_engine):
    """When max_iterations is reached, engine calls _force_final."""
    # Always return code, never FINAL
    mock_engine.qwen.root_completion = AsyncMock(
        return_value="Still thinking...\n```python\nprint('iteration')\n```"
    )
    # Override _force_final to return a known answer
    async def fake_force_final(conversation, system_prompt, trajectory):
        return RLMResult(
            answer="Forced final answer",
            iterations_used=0,
            sub_calls_made=0,
            tokens_used=0,
            cached_hits=0,
            trajectory_summary="",
        )
    mock_engine._force_final = fake_force_final

    result = run(mock_engine.query(
        query="Q",
        context="C",
        thread_id="test",
        max_iterations=3,
    ))

    assert result.answer == "Forced final answer"
    assert result.iterations_used == 3


def test_engine_sub_call_via_llm_query(mock_engine):
    """llm_query() registered in sandbox should call sub_completion."""
    call_log = []

    async def mock_sub(query, context):
        call_log.append(query)
        return "sub answer"

    mock_engine.qwen.sub_completion = AsyncMock(side_effect=mock_sub)

    responses = [
        "```python\nresult = llm_query('sub question', 'some context')\nprint(result)\n```",
        "FINAL(done)",
    ]
    mock_engine.qwen.root_completion = AsyncMock(side_effect=responses)

    result = run(mock_engine.query(
        query="Test sub-call",
        context="context data",
        thread_id="test",
    ))

    assert len(call_log) == 1
    assert call_log[0] == "sub question"


def test_engine_cache_hit(mock_engine):
    """Same sub-query+context should hit cache on second call."""
    sub_call_count = 0

    async def mock_sub(query, context):
        nonlocal sub_call_count
        sub_call_count += 1
        return "cached result"

    mock_engine.qwen.sub_completion = AsyncMock(side_effect=mock_sub)

    # Two iterations calling the same llm_query
    responses = [
        "```python\nr1 = llm_query('same q', 'same c')\nprint(r1)\n```",
        "```python\nr2 = llm_query('same q', 'same c')\nprint(r2)\n```",
        "FINAL(done)",
    ]
    mock_engine.qwen.root_completion = AsyncMock(side_effect=responses)

    result = run(mock_engine.query(
        query="Cache test",
        context="ctx",
        thread_id="test",
    ))

    # Second call should be a cache hit, not a real sub call
    assert sub_call_count == 1
    assert result.cached_hits == 1


def test_engine_status_tracking(mock_engine):
    """Status should reflect current state during execution."""
    mock_engine.qwen.root_completion = AsyncMock(return_value="FINAL(done)")

    run(mock_engine.query(
        query="Q",
        context="C",
        thread_id="status-test",
    ))

    status = mock_engine.get_status("status-test")
    assert status.state == "complete"


def test_engine_no_code_block_nudge(mock_engine):
    """When root LLM returns no code, engine adds a nudge message."""
    responses = [
        "I'm thinking about this...",  # no code
        "FINAL(42)",
    ]
    mock_engine.qwen.root_completion = AsyncMock(side_effect=responses)

    result = run(mock_engine.query(
        query="Q",
        context="C",
        thread_id="test",
    ))

    assert "42" in result.answer
    # Should have been called twice
    assert mock_engine.qwen.root_completion.call_count == 2
