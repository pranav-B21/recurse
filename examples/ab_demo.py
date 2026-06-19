"""A/B demo: do the Recurse vendor mods actually change behavior on Groq?

Same model (llama-3.3-70b-versatile via Groq), same task, two conditions:

  baseline : stock RLM system prompt + truncation effectively OFF
             (RLM_MAX_OUTPUT_CHARS very high) — the fork with mods disabled
  recurse  : directive system prompt (Mod #2) + 4000-char output clamp (Mod #1)

Part 1 is a deterministic, zero-token proof of Mod #1 (the truncation clamp).
Part 2 is a live A/B on Groq that reports, per condition: found the needle?,
root iterations used, total tokens, and the largest single REPL-output message
fed back into the root conversation (the quantity Mod #1 clamps — and what blew
Groq's per-request token cap before the fix).

Run:  recurse/.venv/bin/python examples/ab_demo.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load GROQ_API_KEY from the repo-root .env (spaces around '=' make it
# non-shell-sourceable, but python-dotenv handles it).
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import rlm.core.rlm as rlm_core
from rlm import RLM
from rlm.utils.parsing import format_iteration as _orig_format_iteration
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

from recurse.engine import DIRECTIVE_ADDENDUM

# A COUNTING task in the regime where weak models flail: N markers scattered
# through a large context. Reading a slice and eyeballing undercounts; the
# Mod #2 directive ("regex the markers, don't print big chunks") points
# straight at len(re.findall(...)) == the exact answer. This makes Mod #2 a
# correctness discriminator, not just a style preference.
N_MARKERS = 7
CORRECT = str(N_MARKERS)
SERIALS = [f"WIDGET-SERIAL-{s}" for s in ("7F3A9", "2B81C", "9D4E0", "5A6F2", "C0193", "E7B45", "1F8D6")]
_filler = "logistics note: pallet routed through depot, status nominal, no action required. " * 90


def _build_counting_context() -> str:
    parts = []
    for serial in SERIALS:
        parts.append(_filler)
        parts.append(f"\n>>> INVENTORY RECORD — {serial} — received and shelved.\n")
    parts.append(_filler)
    return "".join(parts)


CONTEXT = _build_counting_context()
QUERY = (
    "How many distinct WIDGET-SERIAL inventory records appear in the document? "
    "Answer with just the number."
)

MODEL = "llama-3.3-70b-versatile"
MAX_ITER = 8


# --- instrumentation: capture the largest REPL-output message fed back -------
_max_fed_back = {"chars": 0}


def _instrumented_format_iteration(*args, **kwargs):
    messages = _orig_format_iteration(*args, **kwargs)
    for m in messages:
        if m.get("role") == "user":
            _max_fed_back["chars"] = max(_max_fed_back["chars"], len(m["content"]))
    return messages


_iters = {"n": 0}
_orig_turn = rlm_core.RLM._completion_turn


def _counting_turn(self, *a, **k):
    _iters["n"] += 1
    return _orig_turn(self, *a, **k)


def part1_deterministic_truncation() -> None:
    """Zero-token proof that Mod #1 clamps a big REPL output."""
    from rlm.core.types import CodeBlock, REPLResult, RLMIteration

    big = "X" * 40_000  # the size that blew Groq's cap in the original incident
    iteration = RLMIteration(
        prompt="p",
        response="```repl\nprint(context)\n```",
        code_blocks=[CodeBlock(code="print(context)", result=REPLResult(stdout=big, stderr="", locals={}))],
    )

    def fed_back_size(limit_env: str | None) -> int:
        if limit_env is None:
            os.environ.pop("RLM_MAX_OUTPUT_CHARS", None)
        else:
            os.environ["RLM_MAX_OUTPUT_CHARS"] = limit_env
        msgs = _orig_format_iteration(iteration)
        return len(next(m["content"] for m in msgs if m["role"] == "user"))

    off = fed_back_size("200000")  # mod effectively disabled
    on = fed_back_size("4000")  # Groq preset

    print("=" * 70)
    print("PART 1 — Mod #1 (REPL output truncation), deterministic, 0 tokens")
    print("=" * 70)
    print(f"  A 40,000-char REPL print is fed back to the root model as:")
    print(f"    mods OFF (limit 200000): {off:>6,} chars")
    print(f"    mods ON  (limit   4000): {on:>6,} chars   <- clamped")
    print(f"  Shrinkage: {off - on:,} chars kept out of the next request.")
    print(f"  At ~4 chars/token that is ~{(off - on)//4:,} tokens not spent —")
    print(f"  the exact overflow that caused Groq's 413 in the original run.\n")


def run_condition(label: str, directive: bool, trunc_limit: str) -> dict:
    os.environ["RLM_MAX_OUTPUT_CHARS"] = trunc_limit
    _max_fed_back["chars"] = 0
    _iters["n"] = 0

    system_prompt = RLM_SYSTEM_PROMPT + ("\n\n" + DIRECTIVE_ADDENDUM if directive else "")

    machine = RLM(
        backend="openai",
        backend_kwargs={
            "api_key": os.environ["GROQ_API_KEY"],
            "base_url": "https://api.groq.com/openai/v1",
            "model_name": MODEL,
        },
        environment="local",
        max_iterations=MAX_ITER,
        custom_system_prompt=system_prompt,
        verbose=False,
    )
    try:
        result = machine.completion(prompt=CONTEXT, root_prompt=QUERY)
        answer = (result.response or "").strip().replace("\n", " ")
        return {
            "label": label,
            "outcome": "correct" if CORRECT in answer else "WRONG",
            "iterations": _iters["n"],
            "tokens": result.usage_summary.total_input_tokens
            + result.usage_summary.total_output_tokens,
            "max_fed_back": _max_fed_back["chars"],
            "answer": answer[:90],
        }
    except Exception as e:  # e.g. Groq 413 token-cap when a big output is fed back uncliped
        return {
            "label": label,
            "outcome": f"FAILED ({type(e).__name__})",
            "iterations": _iters["n"],
            "tokens": 0,
            "max_fed_back": _max_fed_back["chars"],
            "answer": str(e)[:90],
        }


def part3_accumulation_stress() -> None:
    """Reproduce the documented failure: output accumulating across iterations
    until a request crosses Groq's 12K-token cap.

    The query tempts the model to print large verbatim slices each turn. With
    Mod #1 OFF those slices are fed back whole and pile up in history; with
    Mod #1 ON each is clamped to 4000 chars so history stays bounded. Same
    model, same task — the only difference is the clamp.
    """
    big_ctx = (
        "logistics ledger. " + ("entry: pallet moved, seal intact, audit ok. " * 60) + "\n"
    ) * 90  # ~250K chars
    query = (
        "Work through the ledger in chunks. On each turn, print the next ~15000-character "
        "slice of `context` verbatim so we have an audit trail, then continue. After you "
        "have surveyed it, state how many times the word 'pallet' appears."
    )

    print("=" * 70)
    print("PART 3 — accumulation stress test (the documented 413 failure mode)")
    print("=" * 70)
    print(f"  context: {len(big_ctx):,} chars | query tempts ~15K-char verbatim prints/turn")
    print(f"  Groq per-request cap: 12,000 tokens (~48,000 chars)\n")

    rlm_core.format_iteration = _instrumented_format_iteration
    rlm_core.RLM._completion_turn = _counting_turn

    for label, directive, limit in [
        ("baseline (Mod #1 OFF)", False, "200000"),
        ("recurse  (Mod #1 ON) ", True, "4000"),
    ]:
        r = run_condition_on(label, directive, limit, big_ctx, query)
        print(
            f"  {label:<24}{r['outcome']:<26}"
            f"iters={r['iterations']}  tokens={r['tokens']:,}  "
            f"max fed-back={r['max_fed_back']:,} chars"
        )
    print()


def run_condition_on(label, directive, trunc_limit, context, query) -> dict:
    os.environ["RLM_MAX_OUTPUT_CHARS"] = trunc_limit
    _max_fed_back["chars"] = 0
    _iters["n"] = 0
    system_prompt = RLM_SYSTEM_PROMPT + ("\n\n" + DIRECTIVE_ADDENDUM if directive else "")
    machine = RLM(
        backend="openai",
        backend_kwargs={
            "api_key": os.environ["GROQ_API_KEY"],
            "base_url": "https://api.groq.com/openai/v1",
            "model_name": MODEL,
        },
        environment="local",
        max_iterations=MAX_ITER,
        custom_system_prompt=system_prompt,
        verbose=False,
    )
    try:
        result = machine.completion(prompt=context, root_prompt=query)
        return {
            "outcome": "completed",
            "iterations": _iters["n"],
            "tokens": result.usage_summary.total_input_tokens
            + result.usage_summary.total_output_tokens,
            "max_fed_back": _max_fed_back["chars"],
        }
    except Exception as e:
        msg = str(e).lower()
        kind = "FAILED: 413 token-cap" if ("rate_limit" in msg or "413" in msg) else f"FAILED: {type(e).__name__}"
        return {
            "outcome": kind,
            "iterations": _iters["n"],
            "tokens": 0,
            "max_fed_back": _max_fed_back["chars"],
        }


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not found (checked env and ../.env).")

    part1_deterministic_truncation()

    # Live A/B: instrument the loop to measure fed-back sizes + iterations.
    rlm_core.format_iteration = _instrumented_format_iteration
    rlm_core.RLM._completion_turn = _counting_turn
    print("=" * 70)
    print(f"PART 2 — live A/B on Groq ({MODEL}), same task both runs")
    print("=" * 70)
    print(f"  context: {len(CONTEXT):,} chars | {N_MARKERS} markers scattered through it")
    print(f"  query  : {QUERY!r}")
    print(f"  correct answer: {CORRECT} | max_iterations: {MAX_ITER}\n")

    rows = [
        run_condition("baseline (mods OFF)", directive=False, trunc_limit="200000"),
        run_condition("recurse  (mods ON) ", directive=True, trunc_limit="4000"),
    ]

    hdr = f"  {'condition':<22}{'outcome':<10}{'iters':<7}{'tokens':<10}{'max fed-back':<14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r['label']:<22}"
            f"{r['outcome']:<10}"
            f"{r['iterations']:<7}"
            f"{r['tokens']:<10,}"
            f"{r['max_fed_back']:>6,} chars"
        )
    print()
    for r in rows:
        print(f"  {r['label']} -> {r['answer']}")
    print()

    part3_accumulation_stress()


if __name__ == "__main__":
    main()
