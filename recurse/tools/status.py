"""recurse_status tool implementation."""

from __future__ import annotations

import time

from recurse.engine.core import RecurseEngine


async def run_status(engine: RecurseEngine, thread_id: str = "default") -> dict:
    """Report what the RLM is currently doing for a thread."""
    status = engine.get_status(thread_id)
    elapsed = time.time() - status.start_time if status.state != "idle" else 0.0
    return {
        "state": status.state,
        "current_iteration": status.current_iteration,
        "max_iterations": status.max_iterations,
        "sub_calls_completed": status.sub_calls_completed,
        "elapsed_seconds": round(elapsed, 1),
        "partial_findings": status.partial_findings[-10:],  # last 10 findings
    }
