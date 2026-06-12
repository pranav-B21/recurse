"""Local-disk thread persistence for Recurse.

~/.recurse/threads/{thread_id}/
    context.txt          # concatenated files + appended conversation turns (what the RLM sees)
    manifest.json        # ingested file list: path, size, sha256
    turns/{iso_ts}.json  # structured Q&A records

This recreates Monolith's cross-session memory on plain local disk.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FILE_HEADER = "=== FILE: {path} ==="
TURN_HEADER = "=== CONVERSATION TURN {ts} ==="


@dataclass
class IngestResult:
    thread_id: str
    files_count: int
    skipped_count: int
    total_chars: int
    token_estimate: int
    file_tree: str


def _now_iso() -> str:
    # Filesystem-safe ISO timestamp (':' breaks some tooling on macOS/Windows)
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")


def _is_excluded(rel_path: Path, exclude: list[str]) -> bool:
    """A file is excluded when any path component or its name matches a pattern."""
    for pattern in exclude:
        if any(fnmatch.fnmatch(part, pattern) for part in rel_path.parts):
            return True
    return False


def _build_file_tree(paths: list[str]) -> str:
    """Compact indented tree of the ingested relative paths."""
    lines: list[str] = []
    seen_dirs: set[tuple[str, ...]] = set()
    for p in sorted(paths):
        parts = Path(p).parts
        for depth in range(len(parts) - 1):
            dir_key = parts[: depth + 1]
            if dir_key not in seen_dirs:
                seen_dirs.add(dir_key)
                lines.append("  " * depth + parts[depth] + "/")
        lines.append("  " * (len(parts) - 1) + parts[-1])
    return "\n".join(lines)


class ContextStore:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)

    def _thread_dir(self, thread_id: str) -> Path:
        return self.base_path / thread_id

    def has_thread(self, thread_id: str) -> bool:
        return (self._thread_dir(thread_id) / "context.txt").exists()

    def ingest_directory(
        self,
        path: Path,
        thread_id: str,
        exclude: list[str],
        max_file_size_kb: int,
        max_total_files: int,
    ) -> IngestResult:
        """Walk `path`, concatenate readable files into the thread's context.txt.

        Skips excluded, binary, and oversized files; caps the total file count.
        Re-ingesting a thread replaces context.txt (prior turns stay in turns/).
        """
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        thread_dir = self._thread_dir(thread_id)
        thread_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict] = []
        sections: list[str] = []
        skipped = 0

        candidates = sorted(p for p in root.rglob("*") if p.is_file())
        for file_path in candidates:
            rel = file_path.relative_to(root)
            if _is_excluded(rel, exclude):
                continue
            if len(manifest) >= max_total_files:
                skipped += 1
                continue
            raw = file_path.read_bytes()
            if len(raw) > max_file_size_kb * 1024:
                skipped += 1
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                skipped += 1  # binary file
                continue

            sections.append(f"{FILE_HEADER.format(path=rel)}\n{text}\n")
            manifest.append(
                {
                    "path": str(rel),
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )

        if not manifest:
            raise ValueError(
                f"No ingestible text files found in {root} "
                f"(after exclusions: {exclude})."
            )

        context = "\n".join(sections)
        (thread_dir / "context.txt").write_text(context, encoding="utf-8")
        (thread_dir / "manifest.json").write_text(
            json.dumps(
                {"root": str(root), "ingested_at": _now_iso(), "files": manifest},
                indent=2,
            ),
            encoding="utf-8",
        )

        paths = [f["path"] for f in manifest]
        return IngestResult(
            thread_id=thread_id,
            files_count=len(manifest),
            skipped_count=skipped,
            total_chars=len(context),
            token_estimate=len(context) // 4,
            file_tree=_build_file_tree(paths),
        )

    def load_context(self, thread_id: str) -> str:
        context_file = self._thread_dir(thread_id) / "context.txt"
        if not context_file.exists():
            raise FileNotFoundError(
                f"Thread '{thread_id}' has no context. "
                f"Ingest a directory first (recurse ingest DIR --thread {thread_id})."
            )
        return context_file.read_text(encoding="utf-8")

    def append_turn(self, thread_id: str, query: str, answer: str, meta: dict) -> None:
        thread_dir = self._thread_dir(thread_id)
        turns_dir = thread_dir / "turns"
        turns_dir.mkdir(parents=True, exist_ok=True)

        ts = _now_iso()
        (turns_dir / f"{ts}.json").write_text(
            json.dumps({"ts": ts, "query": query, "answer": answer, "meta": meta}, indent=2),
            encoding="utf-8",
        )
        with open(thread_dir / "context.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{TURN_HEADER.format(ts=ts)}\nQ: {query}\nA: {answer}\n")

    def get_turns(self, thread_id: str, limit: int = 10) -> list[dict]:
        turns_dir = self._thread_dir(thread_id) / "turns"
        if not turns_dir.is_dir():
            return []
        files = sorted(turns_dir.glob("*.json"))[-limit:]
        return [json.loads(f.read_text(encoding="utf-8")) for f in files]

    def list_threads(self) -> list[dict]:
        if not self.base_path.is_dir():
            return []
        threads = []
        for d in sorted(self.base_path.iterdir()):
            context_file = d / "context.txt"
            if not (d.is_dir() and context_file.exists()):
                continue
            threads.append(
                {
                    "thread_id": d.name,
                    "context_chars": context_file.stat().st_size,
                    "turns": len(list((d / "turns").glob("*.json"))) if (d / "turns").is_dir() else 0,
                }
            )
        return threads

    def delete_thread(self, thread_id: str) -> None:
        thread_dir = self._thread_dir(thread_id)
        if not thread_dir.is_dir():
            raise FileNotFoundError(f"No such thread: '{thread_id}'")
        # Only ever delete inside our own storage root
        thread_dir = thread_dir.resolve()
        if self.base_path.resolve() not in thread_dir.parents:
            raise ValueError(f"Refusing to delete outside storage path: {thread_dir}")
        import shutil

        shutil.rmtree(thread_dir)
