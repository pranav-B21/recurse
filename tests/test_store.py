"""Tests for recurse.store: ingestion, exclusions, persistence, turns."""

import json

import pytest

from recurse.store import ContextStore

EXCLUDE = ["node_modules", ".git", "__pycache__", "*.lock", "*.pyc"]


@pytest.fixture
def sample_dir(tmp_path):
    src = tmp_path / "project"
    (src / "sub").mkdir(parents=True)
    (src / "node_modules").mkdir()
    (src / "a.py").write_text("def hello():\n    return 'world'\n")
    (src / "sub" / "b.md").write_text("# Notes\nthe needle is NEEDLE-42\n")
    (src / "binary.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    (src / "big.txt").write_text("x" * 2048)
    (src / "node_modules" / "dep.js").write_text("module.exports = 1")
    (src / "poetry.lock").write_text("[[package]]")
    return src


@pytest.fixture
def store(tmp_path):
    return ContextStore(tmp_path / "threads")


def _ingest(store, sample_dir, **overrides):
    kwargs = dict(
        exclude=EXCLUDE, max_file_size_kb=1, max_total_files=100, thread_id="t1"
    )
    kwargs.update(overrides)
    return store.ingest_directory(sample_dir, **kwargs)


def test_ingest_includes_text_files_with_headers(store, sample_dir):
    res = _ingest(store, sample_dir)
    ctx = store.load_context("t1")
    assert "=== FILE: a.py ===" in ctx
    assert "=== FILE: sub/b.md ===" in ctx
    assert "NEEDLE-42" in ctx
    assert res.files_count == 2
    assert res.token_estimate == len(ctx) // 4
    assert "a.py" in res.file_tree and "sub/" in res.file_tree


def test_ingest_skips_binary_oversized_and_excluded(store, sample_dir):
    res = _ingest(store, sample_dir)
    ctx = store.load_context("t1")
    assert "binary.bin" not in ctx
    assert "big.txt" not in ctx  # > 1 KB limit
    assert "node_modules" not in ctx
    assert "poetry.lock" not in ctx
    assert res.skipped_count == 2  # binary + oversized (excluded aren't counted)


def test_ingest_respects_max_total_files(store, sample_dir):
    res = _ingest(store, sample_dir, max_total_files=1)
    assert res.files_count == 1


def test_ingest_writes_manifest_with_hashes(store, sample_dir):
    _ingest(store, sample_dir)
    manifest = json.loads(
        (store.base_path / "t1" / "manifest.json").read_text()
    )
    paths = {f["path"] for f in manifest["files"]}
    assert paths == {"a.py", "sub/b.md"}
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])


def test_ingest_empty_dir_raises(store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No ingestible"):
        store.ingest_directory(
            empty, "t1", exclude=EXCLUDE, max_file_size_kb=500, max_total_files=10
        )


def test_ingest_missing_dir_raises(store, tmp_path):
    with pytest.raises(NotADirectoryError):
        store.ingest_directory(
            tmp_path / "nope", "t1", exclude=[], max_file_size_kb=500, max_total_files=10
        )


def test_append_turn_and_history(store, sample_dir):
    _ingest(store, sample_dir)
    store.append_turn("t1", "q1?", "a1", {"provider": "groq"})
    store.append_turn("t1", "q2?", "a2", {"provider": "groq"})

    ctx = store.load_context("t1")
    assert ctx.count("=== CONVERSATION TURN") == 2
    assert "Q: q2?" in ctx and "A: a2" in ctx

    turns = store.get_turns("t1")
    assert [t["query"] for t in turns] == ["q1?", "q2?"]
    assert store.get_turns("t1", limit=1)[0]["query"] == "q2?"
    assert store.get_turns("nope") == []


def test_thread_management(store, sample_dir):
    assert store.list_threads() == []
    assert not store.has_thread("t1")
    _ingest(store, sample_dir)
    store.append_turn("t1", "q", "a", {})

    assert store.has_thread("t1")
    threads = store.list_threads()
    assert len(threads) == 1
    assert threads[0]["thread_id"] == "t1"
    assert threads[0]["turns"] == 1

    store.delete_thread("t1")
    assert not store.has_thread("t1")
    with pytest.raises(FileNotFoundError):
        store.delete_thread("t1")


def test_load_context_missing_thread_raises(store):
    with pytest.raises(FileNotFoundError, match="nope"):
        store.load_context("nope")
