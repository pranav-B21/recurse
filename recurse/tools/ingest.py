"""recurse_ingest tool implementation."""

from __future__ import annotations

from pathlib import Path

from recurse.config import RecurseConfig
from recurse.store.context_store import ContextStore


async def run_ingest(
    config: RecurseConfig,
    path: str,
    thread_id: str = "default",
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict:
    """Ingest a directory or file into a thread's context store."""
    store = ContextStore(config.storage_path)
    ingest_cfg = config.ingest

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    effective_exclude = list(ingest_cfg.default_exclude)
    if exclude_patterns:
        effective_exclude.extend(exclude_patterns)

    result = store.ingest_directory(
        path=resolved,
        thread_id=thread_id,
        include_patterns=include_patterns,
        exclude_patterns=effective_exclude,
        max_file_size_kb=ingest_cfg.max_file_size_kb,
        max_total_files=ingest_cfg.max_total_files,
    )

    return {
        "thread_id": result.thread_id,
        "files_ingested": result.files_ingested,
        "total_size_bytes": result.total_size_bytes,
        "total_tokens_estimate": result.total_tokens_estimate,
        "file_tree": result.file_tree,
    }
