"""recurse_query tool implementation."""

from __future__ import annotations

from pathlib import Path

from recurse.config import RecurseConfig
from recurse.engine.core import RecurseEngine, RLMResult
from recurse.store.context_store import ContextStore


async def run_query(
    engine: RecurseEngine,
    config: RecurseConfig,
    query: str,
    context_source: str,
    thread_id: str = "default",
    max_iterations: int = 15,
    token_budget: int | None = None,
) -> dict:
    """
    Resolve context from context_source, then run the RLM loop.

    context_source formats:
      "thread:{thread_id}"  — use stored context
      "path:/abs/path"      — ingest and use directory/file
      "inline:..."          — context provided directly
    """
    context = await _resolve_context(engine, config, context_source, thread_id)
    result: RLMResult = await engine.query(
        query=query,
        context=context,
        thread_id=thread_id,
        max_iterations=max_iterations,
        token_budget=token_budget,
    )

    # Persist the conversation
    engine.store.save_conversation(
        thread_id=thread_id,
        query=query,
        answer=result.answer,
        metadata={
            "iterations_used": result.iterations_used,
            "sub_calls_made": result.sub_calls_made,
            "tokens_used": result.tokens_used,
            "cached_hits": result.cached_hits,
        },
    )

    return {
        "answer": result.answer,
        "iterations_used": result.iterations_used,
        "sub_calls_made": result.sub_calls_made,
        "tokens_used": result.tokens_used,
        "cached_hits": result.cached_hits,
        "trajectory_summary": result.trajectory_summary,
    }


async def _resolve_context(
    engine: RecurseEngine,
    config: RecurseConfig,
    context_source: str,
    thread_id: str,
) -> str:
    if context_source.startswith("inline:"):
        return context_source[len("inline:"):]

    if context_source.startswith("thread:"):
        tid = context_source[len("thread:"):]
        return engine.store.load_context(tid)

    if context_source.startswith("path:"):
        raw_path = context_source[len("path:"):]
        path = Path(raw_path).expanduser().resolve()
        ingest_cfg = config.ingest
        if path.is_dir():
            engine.store.ingest_directory(
                path=path,
                thread_id=thread_id,
                exclude_patterns=ingest_cfg.default_exclude,
                max_file_size_kb=ingest_cfg.max_file_size_kb,
                max_total_files=ingest_cfg.max_total_files,
            )
        elif path.is_file():
            # Single file: write directly to store
            store = ContextStore(config.storage_path)
            content = path.read_text(encoding="utf-8", errors="replace")
            (store._files_dir(thread_id)).mkdir(parents=True, exist_ok=True)
            encoded = store._encode_path(path.name)
            (store._files_dir(thread_id) / encoded).write_text(content, encoding="utf-8")
        return engine.store.load_context(thread_id)

    raise ValueError(
        f"Unknown context_source format: {context_source!r}. "
        "Use 'thread:ID', 'path:/abs/path', or 'inline:...'"
    )
