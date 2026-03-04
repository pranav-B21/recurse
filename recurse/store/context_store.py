"""ContextStore — persistent structured storage for thread contexts.

Storage layout per thread:
  {storage_path}/{thread_id}/
    manifest.json         — file list with path, hash, size
    files/
      src__main.py        — file contents (path separators → __)
    conversations/
      {timestamp}.json    — past Q&A pairs
    cache/
      {hash}.json         — cached sub-call results
"""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterator


@dataclasses.dataclass
class IngestResult:
    files_ingested: int
    total_size_bytes: int
    total_tokens_estimate: int
    file_tree: str
    thread_id: str


class ContextStore:
    def __init__(self, storage_path: Path | str) -> None:
        self.root = Path(storage_path).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _thread_dir(self, thread_id: str) -> Path:
        return self.root / thread_id

    def _files_dir(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "files"

    def _convos_dir(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "conversations"

    def _cache_dir(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "cache"

    def _manifest_path(self, thread_id: str) -> Path:
        return self._thread_dir(thread_id) / "manifest.json"

    @staticmethod
    def _encode_path(file_path: str) -> str:
        """Convert file path to flat filename: src/main.py → src__main.py"""
        return file_path.replace(os.sep, "__").replace("/", "__").lstrip("_")

    @staticmethod
    def _decode_path(encoded: str) -> str:
        return encoded.replace("__", "/")

    def ingest_directory(
        self,
        path: str | Path,
        thread_id: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size_kb: int = 500,
        max_total_files: int = 5000,
    ) -> IngestResult:
        """Walk directory, store each file, build manifest."""
        source = Path(path).expanduser().resolve()
        files_dir = self._files_dir(thread_id)
        files_dir.mkdir(parents=True, exist_ok=True)
        self._convos_dir(thread_id).mkdir(parents=True, exist_ok=True)
        self._cache_dir(thread_id).mkdir(parents=True, exist_ok=True)

        manifest_entries: list[dict] = []
        total_size = 0
        file_tree_lines: list[str] = []

        for rel_path in _walk_files(source, include_patterns, exclude_patterns, max_file_size_kb):
            if len(manifest_entries) >= max_total_files:
                break

            abs_path = source / rel_path
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            size = abs_path.stat().st_size
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            encoded_name = self._encode_path(str(rel_path))
            dest = files_dir / encoded_name
            dest.write_text(content, encoding="utf-8")

            manifest_entries.append({
                "path": str(rel_path),
                "encoded": encoded_name,
                "hash": content_hash,
                "size": size,
            })
            total_size += size
            file_tree_lines.append(str(rel_path))

        manifest = {
            "thread_id": thread_id,
            "source": str(source),
            "ingested_at": time.time(),
            "files": manifest_entries,
            "total_size": total_size,
        }
        self._manifest_path(thread_id).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # Estimate tokens (~4 chars per token)
        tokens_estimate = total_size // 4

        file_tree = _build_file_tree(file_tree_lines)
        return IngestResult(
            files_ingested=len(manifest_entries),
            total_size_bytes=total_size,
            total_tokens_estimate=tokens_estimate,
            file_tree=file_tree,
            thread_id=thread_id,
        )

    def load_context(self, thread_id: str) -> str:
        """Load all files as a single concatenated string with file path headers."""
        files_dir = self._files_dir(thread_id)
        if not files_dir.exists():
            return ""

        manifest = self.get_manifest(thread_id)
        parts: list[str] = []

        # Load in manifest order to be deterministic
        for entry in manifest.get("files", []):
            encoded = entry.get("encoded", "")
            rel_path = entry.get("path", encoded)
            file_path = files_dir / encoded
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8", errors="replace")
            parts.append(f"=== FILE: {rel_path} ===\n{content}")

        return "\n\n".join(parts)

    def get_file(self, thread_id: str, file_path: str) -> str | None:
        """Load a single file's contents."""
        encoded = self._encode_path(file_path)
        dest = self._files_dir(thread_id) / encoded
        if not dest.exists():
            return None
        return dest.read_text(encoding="utf-8", errors="replace")

    def get_manifest(self, thread_id: str) -> dict:
        """Return file list with metadata."""
        path = self._manifest_path(thread_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_conversation(
        self,
        thread_id: str,
        query: str,
        answer: str,
        metadata: dict,
    ) -> None:
        """Persist a Q&A pair."""
        convos_dir = self._convos_dir(thread_id)
        convos_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        payload = {"query": query, "answer": answer, "timestamp": ts, **metadata}
        (convos_dir / f"{ts}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def list_threads(self) -> list[dict]:
        """Return metadata for all stored threads."""
        result = []
        for thread_dir in sorted(self.root.iterdir()):
            if not thread_dir.is_dir():
                continue
            thread_id = thread_dir.name
            manifest = self.get_manifest(thread_id)
            result.append({
                "thread_id": thread_id,
                "files": len(manifest.get("files", [])),
                "total_size": manifest.get("total_size", 0),
                "source": manifest.get("source", ""),
                "ingested_at": manifest.get("ingested_at", 0),
            })
        return result

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread directory. Returns True if it existed."""
        thread_dir = self._thread_dir(thread_id)
        if thread_dir.exists():
            shutil.rmtree(thread_dir)
            return True
        return False


# ── helpers ──────────────────────────────────────────────────────────────────

def _walk_files(
    root: Path,
    include: list[str] | None,
    exclude: list[str] | None,
    max_size_kb: int,
) -> Iterator[Path]:
    """Yield relative file paths under root, applying filters."""
    max_bytes = max_size_kb * 1024
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)

        # Prune excluded directories in-place so os.walk skips them
        if exclude:
            dirnames[:] = [
                d for d in dirnames
                if not any(fnmatch.fnmatch(d, pat) for pat in exclude)
            ]

        for filename in filenames:
            rel_file = rel_dir / filename if str(rel_dir) != "." else Path(filename)

            if exclude:
                skip = any(
                    fnmatch.fnmatch(filename, pat) or fnmatch.fnmatch(str(rel_file), pat)
                    for pat in exclude
                )
                if skip:
                    continue

            if include:
                match = any(fnmatch.fnmatch(filename, pat) for pat in include)
                if not match:
                    continue

            full = root / rel_file
            try:
                if full.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue

            yield rel_file


def _build_file_tree(paths: list[str]) -> str:
    """Build a compact ASCII tree from a flat list of relative paths."""
    if not paths:
        return "(empty)"
    # Simple representation: just join sorted paths
    return "\n".join(sorted(paths))
