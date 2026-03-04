"""Tests for ContextStore and ResultCache."""

import json
import tempfile
from pathlib import Path

import pytest

from recurse.store.cache import ResultCache
from recurse.store.context_store import ContextStore


# ── ContextStore tests ─────────────────────────────────────────────────────


@pytest.fixture
def tmp_store(tmp_path):
    return ContextStore(tmp_path)


@pytest.fixture
def sample_project(tmp_path):
    """Create a tiny sample project for ingestion tests."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hello')\n")
    (project / "utils.py").write_text("def helper():\n    return 42\n")
    src = project / "src"
    src.mkdir()
    (src / "core.py").write_text("# core module\nCONSTANT = 'test'\n")
    # Should be excluded
    (project / "__pycache__").mkdir()
    (project / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00" * 10)
    return project


def test_ingest_directory(tmp_store, sample_project):
    result = tmp_store.ingest_directory(
        path=sample_project,
        thread_id="test",
        exclude_patterns=["__pycache__", "*.pyc"],
    )
    assert result.files_ingested == 3  # main.py, utils.py, src/core.py
    assert result.total_size_bytes > 0
    assert result.thread_id == "test"
    assert "main.py" in result.file_tree or "main.py" in result.file_tree


def test_load_context_format(tmp_store, sample_project):
    tmp_store.ingest_directory(
        path=sample_project,
        thread_id="test",
        exclude_patterns=["__pycache__", "*.pyc"],
    )
    context = tmp_store.load_context("test")
    assert "=== FILE:" in context
    assert "print('hello')" in context
    assert "helper" in context


def test_get_file(tmp_store, sample_project):
    tmp_store.ingest_directory(
        path=sample_project,
        thread_id="test",
        exclude_patterns=["__pycache__", "*.pyc"],
    )
    content = tmp_store.get_file("test", "main.py")
    assert content is not None
    assert "print('hello')" in content


def test_get_file_not_found(tmp_store):
    result = tmp_store.get_file("nonexistent", "foo.py")
    assert result is None


def test_get_manifest(tmp_store, sample_project):
    tmp_store.ingest_directory(
        path=sample_project,
        thread_id="test",
        exclude_patterns=["__pycache__", "*.pyc"],
    )
    manifest = tmp_store.get_manifest("test")
    assert "files" in manifest
    assert len(manifest["files"]) == 3
    paths = [f["path"] for f in manifest["files"]]
    assert any("main.py" in p for p in paths)


def test_save_and_load_conversation(tmp_store):
    tmp_store.save_conversation(
        thread_id="test",
        query="What is 2+2?",
        answer="4",
        metadata={"iterations_used": 1},
    )
    convos_dir = tmp_store._convos_dir("test")
    files = list(convos_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["query"] == "What is 2+2?"
    assert data["answer"] == "4"


def test_list_threads(tmp_store, sample_project):
    tmp_store.ingest_directory(sample_project, "thread1", exclude_patterns=["__pycache__", "*.pyc"])
    tmp_store.ingest_directory(sample_project, "thread2", exclude_patterns=["__pycache__", "*.pyc"])
    threads = tmp_store.list_threads()
    thread_ids = [t["thread_id"] for t in threads]
    assert "thread1" in thread_ids
    assert "thread2" in thread_ids


def test_delete_thread(tmp_store, sample_project):
    tmp_store.ingest_directory(sample_project, "to_delete", exclude_patterns=["__pycache__", "*.pyc"])
    assert tmp_store.delete_thread("to_delete") is True
    assert tmp_store.get_manifest("to_delete") == {}


def test_encode_decode_path():
    assert ContextStore._encode_path("src/main.py") == "src__main.py"
    assert ContextStore._encode_path("a/b/c.py") == "a__b__c.py"


def test_max_file_size_respected(tmp_path):
    store = ContextStore(tmp_path / "store")
    project = tmp_path / "project"
    project.mkdir()
    (project / "small.py").write_text("x = 1")
    (project / "big.py").write_text("x" * 600 * 1024)  # 600KB
    result = store.ingest_directory(
        path=project,
        thread_id="test",
        max_file_size_kb=500,
    )
    assert result.files_ingested == 1  # only small.py


# ── ResultCache tests ──────────────────────────────────────────────────────


@pytest.fixture
def tmp_cache(tmp_path):
    return ResultCache(tmp_path)


def test_cache_key_deterministic():
    key1 = ResultCache.key("query", "context")
    key2 = ResultCache.key("query", "context")
    assert key1 == key2


def test_cache_key_different_inputs():
    key1 = ResultCache.key("query1", "context")
    key2 = ResultCache.key("query2", "context")
    assert key1 != key2


def test_cache_get_set(tmp_cache):
    key = ResultCache.key("test query", "test context")
    assert tmp_cache.get(key) is None  # miss

    tmp_cache.set(key, "cached answer")
    assert tmp_cache.get(key) == "cached answer"  # hit


def test_cache_clear(tmp_cache):
    key = ResultCache.key("q", "c")
    tmp_cache.set(key, "value")
    count = tmp_cache.clear()
    assert count == 1
    assert tmp_cache.get(key) is None
