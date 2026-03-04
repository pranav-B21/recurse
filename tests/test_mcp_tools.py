"""MCP tool integration tests (no Ollama required — uses mocks)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from recurse.config import RecurseConfig
from recurse.engine.core import RecurseEngine, RLMResult
from recurse.tools.ingest import run_ingest
from recurse.tools.query import run_query
from recurse.tools.status import run_status
from recurse.tools.threads import run_threads


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def config(tmp_path):
    cfg = RecurseConfig()
    cfg.storage.path = str(tmp_path / "threads")
    return cfg


@pytest.fixture
def mock_engine(config):
    engine = RecurseEngine(config)
    engine.qwen.root_completion = AsyncMock(return_value="FINAL(mock answer)")
    engine.qwen.sub_completion = AsyncMock(return_value="sub mock")
    return engine


@pytest.fixture
def sample_project(tmp_path):
    p = tmp_path / "project"
    p.mkdir()
    (p / "main.py").write_text("x = 1\n")
    (p / "utils.py").write_text("def f(): pass\n")
    return p


# ── recurse_ingest ─────────────────────────────────────────────────────────


def test_ingest_returns_expected_keys(config, sample_project):
    result = run(run_ingest(config, str(sample_project), thread_id="ingest-test"))
    assert "files_ingested" in result
    assert "total_size_bytes" in result
    assert "total_tokens_estimate" in result
    assert "file_tree" in result
    assert result["files_ingested"] == 2


def test_ingest_nonexistent_path(config):
    with pytest.raises(FileNotFoundError):
        run(run_ingest(config, "/nonexistent/path/xyz"))


def test_ingest_with_include_patterns(config, sample_project):
    result = run(run_ingest(
        config, str(sample_project),
        thread_id="filtered",
        include_patterns=["main.py"],
    ))
    assert result["files_ingested"] == 1


# ── recurse_query ──────────────────────────────────────────────────────────


def test_query_inline_context(config, mock_engine):
    result = run(run_query(
        engine=mock_engine,
        config=config,
        query="What is X?",
        context_source="inline:X is 42",
        thread_id="q-test",
    ))
    assert "answer" in result
    assert result["answer"] == "mock answer"
    assert "iterations_used" in result
    assert "sub_calls_made" in result
    assert "tokens_used" in result
    assert "cached_hits" in result
    assert "trajectory_summary" in result


def test_query_thread_context(config, mock_engine, sample_project):
    # First ingest
    run(run_ingest(config, str(sample_project), thread_id="qt-thread"))
    # Then query via thread source
    result = run(run_query(
        engine=mock_engine,
        config=config,
        query="Summarize the code",
        context_source="thread:qt-thread",
        thread_id="qt-thread",
    ))
    assert result["answer"] == "mock answer"


def test_query_path_context(config, mock_engine, sample_project):
    result = run(run_query(
        engine=mock_engine,
        config=config,
        query="What does main.py do?",
        context_source=f"path:{sample_project}",
        thread_id="path-test",
    ))
    assert result["answer"] == "mock answer"


def test_query_invalid_context_source(config, mock_engine):
    with pytest.raises(ValueError, match="Unknown context_source format"):
        run(run_query(
            engine=mock_engine,
            config=config,
            query="Q",
            context_source="bad:source",
        ))


def test_query_saves_conversation(config, mock_engine):
    run(run_query(
        engine=mock_engine,
        config=config,
        query="Test query",
        context_source="inline:some context",
        thread_id="save-test",
    ))
    from recurse.store.context_store import ContextStore
    store = ContextStore(config.storage_path)
    convos = list(store._convos_dir("save-test").glob("*.json"))
    assert len(convos) == 1


# ── recurse_status ─────────────────────────────────────────────────────────


def test_status_idle_thread(config, mock_engine):
    result = run(run_status(mock_engine, thread_id="never-ran"))
    assert result["state"] == "idle"
    assert result["current_iteration"] == 0


def test_status_after_query(config, mock_engine):
    run(run_query(
        engine=mock_engine,
        config=config,
        query="Q",
        context_source="inline:C",
        thread_id="status-q",
    ))
    result = run(run_status(mock_engine, thread_id="status-q"))
    assert result["state"] == "complete"


# ── recurse_threads ────────────────────────────────────────────────────────


def test_threads_list_empty(config):
    result = run(run_threads(config, action="list"))
    assert result["count"] == 0
    assert result["threads"] == []


def test_threads_list_after_ingest(config, sample_project):
    run(run_ingest(config, str(sample_project), thread_id="list-test"))
    result = run(run_threads(config, action="list"))
    assert result["count"] == 1
    assert result["threads"][0]["thread_id"] == "list-test"


def test_threads_inspect(config, sample_project):
    run(run_ingest(config, str(sample_project), thread_id="inspect-me"))
    result = run(run_threads(config, action="inspect", thread_id="inspect-me"))
    assert result["thread_id"] == "inspect-me"
    assert result["files"] == 2


def test_threads_inspect_nonexistent(config):
    result = run(run_threads(config, action="inspect", thread_id="ghost"))
    assert "error" in result


def test_threads_delete(config, sample_project):
    run(run_ingest(config, str(sample_project), thread_id="delete-me"))
    result = run(run_threads(config, action="delete", thread_id="delete-me"))
    assert result["deleted"] is True
    # Verify it's gone
    list_result = run(run_threads(config, action="list"))
    assert list_result["count"] == 0


def test_threads_invalid_action(config):
    with pytest.raises(ValueError, match="Unknown action"):
        run(run_threads(config, action="explode"))
