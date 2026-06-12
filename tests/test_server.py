"""Tests for the MCP server: tool registration and wiring (no live API)."""

from types import SimpleNamespace

import pytest

import recurse.server as server_mod
from recurse.server import recurse_ingest, recurse_query, recurse_threads


class _FakeStore:
    def __init__(self):
        self.deleted = []

    def list_threads(self):
        return [{"thread_id": "self", "context_chars": 1234, "turns": 2}]

    def get_turns(self, thread_id, limit=10):
        return [{"ts": "2026-06-11", "query": "q", "answer": "a", "meta": {}}]

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)


class _FakeEngine:
    def __init__(self):
        self.store = _FakeStore()

    def query(self, query, context_source, thread_id="default"):
        return SimpleNamespace(
            response="answer", execution_time=2.0, usage_summary="usage-info"
        )

    def ingest(self, path, thread_id="default"):
        return SimpleNamespace(
            files_count=3, token_estimate=1000, skipped_count=1, file_tree="a.py"
        )


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(server_mod, "_engine", engine)
    return engine


def test_tools_registered():
    import asyncio

    tools = {t.name for t in asyncio.run(server_mod.mcp.list_tools())}
    assert tools == {"recurse_query", "recurse_ingest", "recurse_threads"}


def test_recurse_query_formats_answer_with_footer():
    out = recurse_query("q?", "inline:ctx", "t1")
    assert out.startswith("answer")
    assert "2.0s" in out and "thread: t1" in out


def test_recurse_ingest_reports_counts():
    out = recurse_ingest("/some/dir", "t1")
    assert "3 files" in out and "1,000 tokens" in out and "a.py" in out


def test_recurse_threads_actions(fake_engine):
    assert "self: 1,234 chars, 2 turns" in recurse_threads("list")
    assert '"query": "q"' in recurse_threads("history", "self")
    assert "required" in recurse_threads("history")
    assert "Deleted" in recurse_threads("delete", "old")
    assert fake_engine.store.deleted == ["old"]
    assert "unknown action" in recurse_threads("bogus")
