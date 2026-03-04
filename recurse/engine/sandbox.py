"""Sandboxed Python REPL for executing root LLM code.

Modes:
  subprocess (default) — exec() with persistent globals, timeout via threading
  docker               — run code in an isolated python:3.12-slim container
"""

from __future__ import annotations

import asyncio
import io
import contextlib
import threading
from typing import Any


class Sandbox:
    def __init__(self, mode: str = "subprocess", timeout_seconds: int = 30) -> None:
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        # Globals persist across iterations for this sandbox instance
        self.globals: dict[str, Any] = {
            "__builtins__": __builtins__,
        }

    def reset(self) -> None:
        """Reset globals between top-level queries (keeps builtins)."""
        self.globals = {"__builtins__": __builtins__}

    def set_variable(self, name: str, value: Any) -> None:
        self.globals[name] = value

    def register_function(self, name: str, fn: Any) -> None:
        self.globals[name] = fn

    async def execute(self, code: str) -> str:
        """Execute Python code and capture stdout + return value."""
        if self.mode == "docker":
            return await self._execute_docker(code)
        return await self._execute_subprocess(code)

    async def _execute_subprocess(self, code: str) -> str:
        """Execute in-process with a timeout, capturing stdout."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_code_sync, code)

    def _run_code_sync(self, code: str) -> str:
        stdout_capture = io.StringIO()
        local_vars: dict[str, Any] = {}
        result = {"output": "", "error": None}

        def target() -> None:
            try:
                with contextlib.redirect_stdout(stdout_capture):
                    exec(code, self.globals, local_vars)  # noqa: S102
                result["output"] = stdout_capture.getvalue()
                # Persist local vars into globals for subsequent iterations
                self.globals.update(local_vars)
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=self.timeout_seconds)

        if t.is_alive():
            return f"Error: execution timed out after {self.timeout_seconds}s"

        if result["error"]:
            return f"Error: {result['error']}"

        output = result["output"]
        return output if output else "(no output)"

    async def _execute_docker(self, code: str) -> str:
        """Execute code in an isolated Docker container."""
        try:
            import docker  # type: ignore[import]
        except ImportError:
            return "Error: docker package not installed. Run: pip install recurse-rlm[docker]"

        client = docker.from_env()
        # Serialize globals that are JSON-serializable for injection
        # (CONTEXT as env var, others dropped — Docker mode is stateless per call)
        context_val = self.globals.get("CONTEXT", "")
        context_len = self.globals.get("CONTEXT_LENGTH", len(context_val))

        preamble = (
            f"CONTEXT = {repr(context_val)}\n"
            f"CONTEXT_LENGTH = {context_len}\n"
        )
        full_code = preamble + code

        try:
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.containers.run(
                    "python:3.12-slim",
                    ["python", "-c", full_code],
                    remove=True,
                    stdout=True,
                    stderr=True,
                    mem_limit="512m",
                    network_disabled=True,
                    timeout=self.timeout_seconds,
                ),
            )
            return output.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error (docker): {type(e).__name__}: {e}"
