"""recurse_threads tool implementation."""

from __future__ import annotations

from recurse.config import RecurseConfig
from recurse.store.context_store import ContextStore


async def run_threads(
    config: RecurseConfig,
    action: str = "list",
    thread_id: str | None = None,
) -> dict:
    """List, inspect, or delete threads."""
    store = ContextStore(config.storage_path)

    if action == "list":
        threads = store.list_threads()
        return {"threads": threads, "count": len(threads)}

    if action == "inspect":
        if not thread_id:
            raise ValueError("thread_id required for action='inspect'")
        manifest = store.get_manifest(thread_id)
        if not manifest:
            return {"error": f"Thread '{thread_id}' not found"}
        convos_dir = store._convos_dir(thread_id)
        last_query = None
        if convos_dir.exists():
            convo_files = sorted(convos_dir.glob("*.json"), reverse=True)
            if convo_files:
                import json
                data = json.loads(convo_files[0].read_text(encoding="utf-8"))
                last_query = data.get("query", "")[:200]
        return {
            "thread_id": thread_id,
            "files": len(manifest.get("files", [])),
            "total_size_bytes": manifest.get("total_size", 0),
            "source": manifest.get("source", ""),
            "ingested_at": manifest.get("ingested_at", 0),
            "last_query": last_query,
        }

    if action == "delete":
        if not thread_id:
            raise ValueError("thread_id required for action='delete'")
        deleted = store.delete_thread(thread_id)
        return {"deleted": deleted, "thread_id": thread_id}

    raise ValueError(f"Unknown action: {action!r}. Use 'list', 'inspect', or 'delete'.")
