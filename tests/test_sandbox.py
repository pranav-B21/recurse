"""Tests for the Sandbox execution environment."""

import asyncio
import pytest

from recurse.engine.sandbox import Sandbox


def test_sandbox_basic_execution():
    sb = Sandbox(mode="subprocess")
    result = asyncio.run(sb.execute("print('hello world')"))
    assert result == "hello world\n"


def test_sandbox_no_output():
    sb = Sandbox(mode="subprocess")
    result = asyncio.run(sb.execute("x = 1 + 1"))
    assert result == "(no output)"


def test_sandbox_error_handling():
    sb = Sandbox(mode="subprocess")
    result = asyncio.run(sb.execute("raise ValueError('test error')"))
    assert "ValueError" in result
    assert "test error" in result


def test_sandbox_globals_persist():
    sb = Sandbox(mode="subprocess")
    asyncio.run(sb.execute("MY_VAR = 42"))
    result = asyncio.run(sb.execute("print(MY_VAR)"))
    assert "42" in result


def test_sandbox_set_variable():
    sb = Sandbox(mode="subprocess")
    sb.set_variable("CONTEXT", "hello world context")
    sb.set_variable("CONTEXT_LENGTH", 19)
    result = asyncio.run(sb.execute("print(CONTEXT_LENGTH)"))
    assert "19" in result


def test_sandbox_register_function():
    sb = Sandbox(mode="subprocess")

    def my_fn(x):
        return x * 2

    sb.register_function("double", my_fn)
    result = asyncio.run(sb.execute("print(double(21))"))
    assert "42" in result


def test_sandbox_reset():
    sb = Sandbox(mode="subprocess")
    asyncio.run(sb.execute("STALE_VAR = 999"))
    sb.reset()
    result = asyncio.run(sb.execute("print(vars().get('STALE_VAR', 'gone'))"))
    assert "gone" in result


def test_sandbox_timeout():
    sb = Sandbox(mode="subprocess", timeout_seconds=1)
    result = asyncio.run(sb.execute("import time; time.sleep(5)"))
    assert "timed out" in result


def test_sandbox_multiline_code():
    sb = Sandbox(mode="subprocess")
    code = """
results = []
for i in range(5):
    results.append(i * i)
print(results)
"""
    result = asyncio.run(sb.execute(code))
    assert "[0, 1, 4, 9, 16]" in result


def test_sandbox_llm_query_integration():
    """Test that a registered llm_query function is callable from exec'd code."""
    sb = Sandbox(mode="subprocess")
    call_log = []

    def mock_llm_query(query, context=""):
        call_log.append((query, context))
        return "mock answer"

    sb.register_function("llm_query", mock_llm_query)
    result = asyncio.run(sb.execute("answer = llm_query('What is this?', 'some text')\nprint(answer)"))
    assert "mock answer" in result
    assert len(call_log) == 1
    assert call_log[0][0] == "What is this?"
