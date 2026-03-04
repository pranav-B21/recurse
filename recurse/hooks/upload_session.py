"""Stop hook: ingest project into a Recurse thread and save session summary.

Claude Code calls this at session end via the Stop hook. It:
  1. Ingests the cwd into a persistent thread (pure file I/O, no LLM calls)
  2. Extracts the last ~10 messages from the session transcript
  3. Saves them as a conversation record in the thread
  4. Writes a .recurse-thread marker file so the next session can restore context

Usage (invoked by Claude Code Stop hook):
  python -m recurse.hooks.upload_session
  (reads JSON payload from stdin)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    try:
        _run()
    except Exception:
        pass  # Never block session end


def _run() -> None:
    payload: dict = {}
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        pass

    cwd = payload.get("cwd", os.getcwd())
    transcript_path = payload.get("transcript_path", "")

    thread_id = _get_thread_id(cwd)
    _write_thread_file(cwd, thread_id)

    # Import here so failures don't affect the hook silently at import time
    from recurse.config import RecurseConfig
    from recurse.store.context_store import ContextStore

    config = RecurseConfig.load()
    store = ContextStore(config.storage_path)
    ingest_cfg = config.ingest

    result = store.ingest_directory(
        path=cwd,
        thread_id=thread_id,
        exclude_patterns=ingest_cfg.default_exclude,
        max_file_size_kb=ingest_cfg.max_file_size_kb,
        max_total_files=ingest_cfg.max_total_files,
    )

    summary = _extract_summary(transcript_path)
    if summary:
        store.save_conversation(
            thread_id,
            "[session end]",
            summary,
            {"files_ingested": result.files_ingested},
        )


def _get_thread_id(cwd: str) -> str:
    """Use git root basename for consistency across subdirectories."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if r.returncode == 0:
            return Path(r.stdout.strip()).name
    except Exception:
        pass
    return Path(cwd).name


def _write_thread_file(cwd: str, thread_id: str) -> None:
    (Path(cwd) / ".recurse-thread").write_text(thread_id)


def _extract_summary(transcript_path: str) -> str:
    """Read last 10 user+assistant messages from transcript JSONL."""
    if not transcript_path or not Path(transcript_path).exists():
        return ""
    messages: list[str] = []
    for line in Path(transcript_path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
            if entry.get("type") not in ("user", "assistant"):
                continue
            role = entry["type"]
            content = entry.get("message", {}).get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            if content.strip():
                messages.append(f"[{role}] {content.strip()[:500]}")
        except Exception:
            continue
    tail = messages[-10:]
    return "\n".join(tail)[:3000] if tail else ""


if __name__ == "__main__":
    main()
