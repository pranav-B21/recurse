"""RecurseEngine: resolve context, build the vendored RLM, run, persist.

Everything is synchronous — rlm.completion() blocks and FastMCP tolerates
sync tools. No asyncio in v1.
"""

from __future__ import annotations

import os
from pathlib import Path

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

from recurse.config import RecurseConfig
from recurse.store import ContextStore, IngestResult

# RECURSE Mod #2 (see vendor/rlm/RECURSE-MODS.md): directive addendum appended
# to the stock system prompt and passed via RLM's custom_system_prompt kwarg.
# Weak open models flail at the RLM protocol without these; harmless for
# strong models. Must not contain unescaped '{'/'}' — the combined prompt is
# later .format()-ed with custom_tools_section.
DIRECTIVE_ADDENDUM = "\n\n".join(
    [
        "Additional operating instructions from Recurse:",
        (
            '- The context may be a concatenation of source files formatted as '
            '`=== FILE: relative/path ===` blocks, possibly followed by '
            '`=== CONVERSATION TURN ... ===` blocks holding prior Q&A. Use regex over these '
            'markers to list files and jump to the relevant ones, e.g. '
            're.findall(r"=== FILE: (.+) ===", context).'
        ),
        (
            "- NEVER print more than 2,000 characters of the context at once. Print slices "
            "and filtered matches, not whole chunks — REPL output beyond the configured "
            "limit is truncated anyway, so re-printing large objects only wastes turns."
        ),
        (
            "- Before chunking blindly, regex/keyword-search the context for the literal "
            "terms of the task (identifiers, function names, quoted phrases from the query) "
            "and inspect a small window around each match."
        ),
        (
            '- As soon as you are confident in the answer, submit it immediately: set '
            'answer["content"] to the answer and answer["ready"] = True in a repl block. '
            "Do not keep exploring after you have what you need."
        ),
    ]
)


class RecurseEngine:
    def __init__(self, config: RecurseConfig):
        self.config = config
        self._p: dict | None = None
        self.store = ContextStore(config.storage_path)

    @property
    def p(self) -> dict:
        """Provider preset + overrides + API key, resolved lazily so that
        store-only operations (threads, history) work without an API key."""
        if self._p is None:
            self._p = self.config.resolved()
        return self._p

    def _build_rlm(self) -> RLM:
        # Drives vendor Mod #1 — per-provider clamp on REPL output fed back to the root LM
        os.environ["RLM_MAX_OUTPUT_CHARS"] = str(self.p["max_output_chars"])

        bk = {"api_key": self.p["api_key"], "model_name": self.p["root_model"]}
        if self.p["base_url"]:
            bk["base_url"] = self.p["base_url"]

        kwargs = dict(
            backend=self.p["backend"],
            backend_kwargs=bk,
            environment=self.config.environment,
            max_iterations=self.config.max_iterations,
            custom_system_prompt=RLM_SYSTEM_PROMPT + "\n\n" + DIRECTIVE_ADDENDUM,
            logger=RLMLogger(log_dir=str(self.config.storage_path / "_logs")),
            verbose=self.config.verbose,
        )
        if self.p["sub_model"] not in (None, "same", self.p["root_model"]):
            sbk = dict(bk, model_name=self.p["sub_model"])
            kwargs.update(other_backends=[self.p["backend"]], other_backend_kwargs=[sbk])
        try:
            return RLM(**kwargs)
        except TypeError:
            # other_backends is experimental — fall back to a single model
            if "other_backends" not in kwargs:
                raise
            kwargs.pop("other_backends", None)
            kwargs.pop("other_backend_kwargs", None)
            import warnings

            warnings.warn(
                "other_backends rejected by the vendored RLM; "
                "falling back to a single model for root + sub calls.",
                stacklevel=2,
            )
            return RLM(**kwargs)

    def _guardrail(self, context: str) -> None:
        """Refuse before spending an API call when the context can't fit the
        provider's per-request token budget (converts a cryptic 413 into an
        actionable message)."""
        est = len(context) // 4
        tpm = self.p.get("tpm_limit")
        if tpm and est > 0.6 * tpm:
            raise RuntimeError(
                f"Context ≈{est:,} tokens exceeds the safe budget for provider "
                f"'{self.config.provider}' (TPM limit {tpm:,}). Shrink the context "
                f"or switch provider (e.g. provider: openai in ~/.recurse/config.yaml)."
            )

    def query(self, query: str, context_source: str, thread_id: str = "default"):
        context = self._resolve_context(context_source, thread_id)
        self._guardrail(context)
        result = self._build_rlm().completion(prompt=context, root_prompt=query)
        self.store.append_turn(
            thread_id,
            query,
            result.response,
            {
                "execution_time": result.execution_time,
                "usage": str(result.usage_summary),
                "provider": self.config.provider,
            },
        )
        return result

    def ingest(self, path: str | Path, thread_id: str = "default") -> IngestResult:
        return self.store.ingest_directory(
            Path(path),
            thread_id,
            exclude=self.config.ingest_exclude,
            max_file_size_kb=self.config.max_file_size_kb,
            max_total_files=self.config.max_total_files,
        )

    def _resolve_context(self, source: str, thread_id: str) -> str:
        """Resolve a context_source string to the actual context text.

        "thread:<id>"  → load that thread's stored context
        "path:<dir>"   → ingest the directory into `thread_id`, then load
        "inline:<text>"→ the text itself
        bare string    → existing path → ingest+load; otherwise treated as inline
        """
        if source.startswith("thread:"):
            return self.store.load_context(source[len("thread:"):])
        if source.startswith("path:"):
            self.ingest(source[len("path:"):], thread_id)
            return self.store.load_context(thread_id)
        if source.startswith("inline:"):
            return source[len("inline:"):]
        if Path(source).expanduser().is_dir():
            self.ingest(source, thread_id)
            return self.store.load_context(thread_id)
        return source
