"""Recurse MCP server (stdio, FastMCP).

Register with Claude Code (use the project venv's python):
    claude mcp add recurse --transport stdio -- \\
        /path/to/recurse/recurse/.venv/bin/python -m recurse.server
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from recurse.config import load_config
from recurse.engine import RecurseEngine

mcp = FastMCP("recurse")

_engine: RecurseEngine | None = None


def get_engine() -> RecurseEngine:
    """Engine is built lazily so the server can start (and list tools) even
    before an API key is configured; key errors surface per-call instead."""
    global _engine
    if _engine is None:
        config = load_config()
        config.verbose = False  # rich console output would corrupt the stdio transport
        _engine = RecurseEngine(config)
    return _engine


@mcp.tool()
def recurse_query(query: str, context_source: str, thread_id: str = "default") -> str:
    """Answer a question over a large codebase/document set using recursive reasoning
    with persistent per-thread memory. context_source: "path:/abs/dir" | "thread:<id>" | "inline:<text>"."""
    r = get_engine().query(query, context_source, thread_id)
    usage = r.usage_summary
    try:
        usage_str = f"{usage.total_input_tokens:,} in / {usage.total_output_tokens:,} out tokens"
    except AttributeError:
        usage_str = str(usage)
    return f"{r.response}\n\n---\n{r.execution_time:.1f}s | {usage_str} | thread: {thread_id}"


@mcp.tool()
def recurse_ingest(path: str, thread_id: str = "default") -> str:
    """Index a directory into a persistent thread for future queries."""
    res = get_engine().ingest(path, thread_id)
    return (
        f"Ingested {res.files_count} files (~{res.token_estimate:,} tokens, "
        f"{res.skipped_count} skipped) into '{thread_id}'.\n{res.file_tree}"
    )


@mcp.tool()
def recurse_threads(action: str = "list", thread_id: str | None = None) -> str:
    """Manage threads: action = list | history | delete."""
    store = get_engine().store
    if action == "list":
        threads = store.list_threads()
        if not threads:
            return "No threads yet."
        return "\n".join(
            f"{t['thread_id']}: {t['context_chars']:,} chars, {t['turns']} turns"
            for t in threads
        )
    if action == "history":
        if not thread_id:
            return "Error: thread_id is required for action='history'."
        turns = store.get_turns(thread_id)
        if not turns:
            return f"No history for thread '{thread_id}'."
        return json.dumps(
            [{"ts": t["ts"], "query": t["query"], "answer": t["answer"]} for t in turns],
            indent=2,
        )
    if action == "delete":
        if not thread_id:
            return "Error: thread_id is required for action='delete'."
        store.delete_thread(thread_id)
        return f"Deleted thread '{thread_id}'."
    return f"Error: unknown action '{action}' (expected list | history | delete)."


if __name__ == "__main__":
    mcp.run()  # stdio
