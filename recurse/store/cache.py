"""Sub-call result cache.

Cache key = sha256(query)[:16] + "_" + sha256(context)[:16]
Cache value = sub-LLM response string, stored as JSON file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResultCache:
    def __init__(self, storage_path: Path | str) -> None:
        self.root = Path(storage_path).expanduser()

    def _cache_dir(self, thread_id: str = "default") -> Path:
        return self.root / thread_id / "cache"

    @staticmethod
    def key(query: str, context: str) -> str:
        q_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        c_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
        return f"{q_hash}_{c_hash}"

    def get(self, key: str, thread_id: str = "default") -> str | None:
        path = self._cache_dir(thread_id) / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("value")
        except Exception:
            return None

    def set(self, key: str, value: str, thread_id: str = "default") -> None:
        cache_dir = self._cache_dir(thread_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}.json"
        path.write_text(json.dumps({"key": key, "value": value}), encoding="utf-8")

    def clear(self, thread_id: str = "default") -> int:
        """Delete all cache entries for a thread. Returns count deleted."""
        cache_dir = self._cache_dir(thread_id)
        if not cache_dir.exists():
            return 0
        count = 0
        for f in cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
