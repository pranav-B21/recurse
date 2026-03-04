"""RecurseEngine — the RLM loop implementation.

The core idea:
  1. Context is stored as a Python variable in the Sandbox (never fed directly into LLM)
  2. Root LLM writes Python code to inspect/decompose the context
  3. Root LLM can call llm_query() from within the sandbox to delegate to sub-LLM
  4. Loop until FINAL() / FINAL_VAR() detected, or max_iterations
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import re
import time
from typing import Any

from recurse.config import RecurseConfig
from recurse.engine.prompts import ROOT_SYSTEM_PROMPT
from recurse.engine.qwen import QwenClient
from recurse.engine.sandbox import Sandbox
from recurse.store.cache import ResultCache
from recurse.store.context_store import ContextStore


@dataclasses.dataclass
class RLMResult:
    answer: str
    iterations_used: int
    sub_calls_made: int
    tokens_used: int
    cached_hits: int
    trajectory_summary: str


@dataclasses.dataclass
class ThreadStatus:
    state: str = "idle"  # idle | decomposing | analyzing | aggregating | complete
    current_iteration: int = 0
    max_iterations: int = 15
    sub_calls_completed: int = 0
    elapsed_seconds: float = 0.0
    partial_findings: list[str] = dataclasses.field(default_factory=list)
    start_time: float = dataclasses.field(default_factory=time.time)


class RecurseEngine:
    def __init__(self, config: RecurseConfig) -> None:
        self.config = config
        self.qwen = QwenClient(config.models)
        self.sandbox = Sandbox(
            mode=config.sandbox.mode,
            timeout_seconds=config.sandbox.timeout_seconds,
        )
        self.store = ContextStore(config.storage_path)
        self.cache = ResultCache(config.cache_path)
        self._status: dict[str, ThreadStatus] = {}

    def get_status(self, thread_id: str) -> ThreadStatus:
        return self._status.get(thread_id, ThreadStatus())

    async def query(
        self,
        query: str,
        context: str,
        thread_id: str = "default",
        max_iterations: int | None = None,
        token_budget: int | None = None,
    ) -> RLMResult:
        if max_iterations is None:
            max_iterations = self.config.engine.max_iterations

        # Reset sandbox state for this query
        self.sandbox.reset()
        self.sandbox.set_variable("CONTEXT", context)
        self.sandbox.set_variable("CONTEXT_LENGTH", len(context))

        # Track sub-call stats
        sub_calls_made = 0
        cached_hits = 0

        # Register sync-safe wrappers for llm_query and batch_llm_query
        def make_sync_wrapper(async_fn: Any):
            def sync_fn(*args: Any, **kwargs: Any) -> Any:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(asyncio.run, async_fn(*args, **kwargs)).result()
            return sync_fn

        async def _llm_query(sub_query: str, sub_context: str = "") -> str:
            nonlocal sub_calls_made, cached_hits
            cache_key = self.cache.key(sub_query, sub_context)
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached_hits += 1
                return cached
            sub_calls_made += 1
            status = self._status.get(thread_id)
            if status:
                status.state = "analyzing"
                status.sub_calls_completed += 1
            result = await self.qwen.sub_completion(sub_query, sub_context)
            self.cache.set(cache_key, result)
            return result

        async def _batch_llm_query(items: list[tuple[str, str]]) -> list[str]:
            return await asyncio.gather(*[_llm_query(q, c) for q, c in items])

        self.sandbox.register_function("llm_query", make_sync_wrapper(_llm_query))
        self.sandbox.register_function("batch_llm_query", make_sync_wrapper(_batch_llm_query))

        # Initialize status
        status = ThreadStatus(max_iterations=max_iterations)
        status.state = "decomposing"
        self._status[thread_id] = status

        system_prompt = ROOT_SYSTEM_PROMPT.format(context_length=len(context))
        conversation: list[dict] = [
            {
                "role": "user",
                "content": f"Query: {query}\n\nContext length: {len(context)} characters.",
            }
        ]
        trajectory: list[dict] = []
        max_trunc = self.config.engine.max_output_truncation

        for i in range(max_iterations):
            status.current_iteration = i + 1
            status.elapsed_seconds = time.time() - status.start_time

            response = await self.qwen.root_completion(system_prompt, conversation)

            # Check for termination signals first
            final = _extract_final(response)
            if final is not None:
                status.state = "complete"
                return RLMResult(
                    answer=final,
                    iterations_used=i + 1,
                    sub_calls_made=sub_calls_made,
                    tokens_used=self.qwen.tokens_used,
                    cached_hits=cached_hits,
                    trajectory_summary=_summarize_trajectory(trajectory),
                )

            # Check for FINAL_VAR — resolve variable from sandbox globals
            final_var = _extract_final_var(response)
            if final_var is not None:
                answer = str(self.sandbox.globals.get(final_var, f"(variable '{final_var}' not found)"))
                status.state = "complete"
                return RLMResult(
                    answer=answer,
                    iterations_used=i + 1,
                    sub_calls_made=sub_calls_made,
                    tokens_used=self.qwen.tokens_used,
                    cached_hits=cached_hits,
                    trajectory_summary=_summarize_trajectory(trajectory),
                )

            # Extract and execute code
            code = _extract_code(response)
            if code:
                status.state = "analyzing"
                output = await self.sandbox.execute(code)
                output = _truncate(output, max_trunc)
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": f"[Execution Output]\n{output}"})
                trajectory.append({"iteration": i + 1, "code": code[:300], "output": output[:300]})
                # Capture partial findings if output looks informative
                if output and output != "(no output)" and not output.startswith("Error:"):
                    status.partial_findings.append(output[:200])
            else:
                # No code block — model is thinking; nudge it forward
                conversation.append({"role": "assistant", "content": response})
                conversation.append({
                    "role": "user",
                    "content": (
                        "Continue. Write Python code in a ```python block to analyze the context, "
                        "or output FINAL(answer) when ready."
                    ),
                })

        # Max iterations reached — force a final answer
        status.state = "aggregating"
        result = await self._force_final(conversation, system_prompt, trajectory)
        result.sub_calls_made = sub_calls_made
        result.tokens_used = self.qwen.tokens_used
        result.cached_hits = cached_hits
        result.iterations_used = max_iterations
        status.state = "complete"
        return result

    async def _force_final(
        self,
        conversation: list[dict],
        system_prompt: str,
        trajectory: list[dict],
    ) -> RLMResult:
        """Ask the root LLM for a final answer after hitting max_iterations."""
        force_conversation = conversation + [
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of iterations. "
                    "Based on everything you have found so far, please output your best answer "
                    "using FINAL(your answer here). Do not write more code."
                ),
            }
        ]
        response = await self.qwen.root_completion(system_prompt, force_conversation)
        final = _extract_final(response)
        if final is None:
            # Try to salvage any text after the last iteration
            final = response.strip() or "Unable to determine answer within iteration budget."

        return RLMResult(
            answer=final,
            iterations_used=0,  # caller overwrites
            sub_calls_made=0,
            tokens_used=0,
            cached_hits=0,
            trajectory_summary=_summarize_trajectory(trajectory),
        )


# ── helpers ──────────────────────────────────────────────────────────────────

_FINAL_RE = re.compile(r"FINAL\(([\s\S]*?)\)", re.MULTILINE)
_FINAL_VAR_RE = re.compile(r"FINAL_VAR\((\w+)\)")
_CODE_BLOCK_RE = re.compile(r"```python\n([\s\S]*?)```", re.MULTILINE)


def _extract_final(text: str) -> str | None:
    m = _FINAL_RE.search(text)
    if not m:
        return None
    return m.group(1).strip() or None


def _extract_final_var(text: str) -> str | None:
    m = _FINAL_VAR_RE.search(text)
    return m.group(1) if m else None


def _extract_code(text: str) -> str | None:
    blocks = _CODE_BLOCK_RE.findall(text)
    return "\n\n".join(blocks).strip() if blocks else None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n... [truncated {len(text) - max_chars} chars] ...\n" + text[-half:]


def _summarize_trajectory(trajectory: list[dict]) -> str:
    if not trajectory:
        return "No code executed."
    lines = [f"Iteration {t['iteration']}: {t['output'][:100]}" for t in trajectory]
    return " | ".join(lines)
