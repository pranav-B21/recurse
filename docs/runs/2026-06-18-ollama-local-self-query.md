# Run log — fully local RLM query on Recurse's own source

**Date:** 2026-06-18
**Machine:** Apple M4 Pro, 36 GB unified memory (Metal)
**Model:** `qwen3.5:35b-a3b` via Ollama (local, OpenAI-compatible endpoint) — **$0, no API, offline**
**Provider preset:** `ollama` (`base_url: http://localhost:11434/v1`, single model for root + sub)

## Command

```bash
recurse ingest ./recurse --thread self          # 6 files, ~7,214 tokens
recurse "what does engine.py do and how does it pick a provider?" \
    --thread self --provider ollama
```

The full Recurse package was loaded into a Python REPL as the variable `context`
(28,858 chars ≈ 7,214 tokens). The model never received the code directly — it
wrote and ran Python to navigate it, recursively, until it submitted a final
answer.

---

## Final answer (verbatim from terminal)

```
╭─ ★ Final Answer ───────────────────────────────────────────────────────────╮
│  Based on my analysis of the codebase:                                       │
│                                                                              │
│  **What engine.py does:**                                                    │
│  1. Defines RecurseEngine class: the main orchestrator for recursive LLM     │
│     reasoning over large codebases/documents.                                │
│  2. Core responsibilities:                                                   │
│     - Resolves context from sources (file path, thread, or inline text)      │
│     - Builds a vendored RLM instance with provider-specific configuration    │
│     - Guards against token budget violations before making API calls         │
│     - Supports persistent per-thread memory/storage via ContextStore         │
│  3. Key methods: __init__, _build_rlm, _guardrail, query, ingest, p property │
│                                                                              │
│  **How it picks a provider:**                                                │
│  1. Config file: provider from ~/.recurse/config.yaml (default "groq")       │
│     - Options: groq, openai, anthropic                                       │
│  2. Resolved via config.resolved(): looks up PRESETS, merges overrides,      │
│     fetches API key from the matching env var                                │
│  3. CLI override: --provider flag                                            │
│  4. Server mode: engine built lazily on first request                        │
│  5. Backend kwargs passed to RLM from the resolved preset                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Run summary (verbatim from terminal)

```
══════════════════════════════════════════════════════════════════════════════
                          Iterations       16
                          Total Time       263.81s
                          Input Tokens     102,521
                          Output Tokens    5,344
══════════════════════════════════════════════════════════════════════════════
263.8s | 102,521 in / 5,344 out tokens | thread: self
```

---

## Accuracy

The answer is **substantively correct** — verified against the real source. It
correctly synthesized four files (`engine.py`, `config.py`, `cli.py`,
`server.py`). Two minor blemishes:

1. **Hallucinated line range:** it cited `_build_rlm()` at "line 210-245";
   engine.py is ~160 lines and `_build_rlm` is ~line 67.
2. **Listed 3 providers, not 4:** it omitted `ollama`. This is correct *for the
   context it was given* — the thread was ingested before the `ollama` preset was
   added, so the snapshot genuinely had only groq/openai/anthropic. The thread
   context is a point-in-time snapshot; re-ingest after code changes.

---

## Iteration-by-iteration (what happened, and what was wasted)

| # | Action | Outcome |
|---|---|---|
| 1 | `re.findall("=== FILE: (.+) ===", context)` to list files | Partly polluted — picked up `relative/path`, `(.+)`, `{path}` (see "self-reference trap") |
| 2 | `context.find("=== FILE: engine.py ===")` | ✅ found engine.py at byte 11837 |
| 3 | Slice engine.py via `find("=== FILE: ")` on the substring | ❌ **wasted** — `next_marker=0`, extracted 0 chars |
| 4 | Regex extract with broken lookahead `(?=== FILE:` | ⚠️ got only **1036 chars** (truncated at engine.py's *own* quoted marker) |
| 5 | Keyword search `str.find(kw, case_insensitive=False)` | ❌ **wasted** — `TypeError: str.find() takes no keyword arguments` |
| 6 | Re-search using `full_engine` | ❌ **wasted** — `NameError: full_engine not defined` (state dropped after iter 5's exception) |
| 7 | Re-extract + keyword search | ⚠️ worked, but on the 1036-char fragment |
| 8 | Print fragment again + find `rlm.completion` | ⚠️ **redundant** — same truncated fragment |
| 9 | List files + extract **config.py** | ✅ got full config.py (the real provider logic) |
| 10–11 | Read **cli.py** in full | ✅ productive |
| 12 | (LLM response) | ❌ **wasted** — empty response, 36.6s |
| 13 | Re-examine engine "from rlm" onward | ⚠️ **redundant** — still the truncated fragment |
| 14 | `context[engine_pos+len(header):]` — slice to end | ✅ **finally** got real engine.py (+ server.py + store.py) |
| 15 | (LLM response) | ❌ **wasted** — empty response |
| 16 | Compose `answer["content"]`, set `ready=True` | ✅ correct final answer |

### Repetition / waste tally

- **Total iterations:** 16
- **Clearly wasted** (errors, empty responses, redundant re-reads): **7** —
  iters 3, 5, 6, 8, 12, 13, 15 → **~44%**
- **Hard Python exceptions:** 2 (TypeError, NameError)
- **Empty LLM responses:** 2 (iters 12, 15)
- **Productive:** ~9 iterations
- A frontier model (e.g. gpt-5-mini) typically resolves this class of question
  in **3–4** tight iterations.

### The "self-reference trap" (why it flailed)

Recurse splits files with `=== FILE: path ===` headers. But Recurse's *own*
source contains that string as **string literals** (`engine.py`'s
`DIRECTIVE_ADDENDUM`, `store.py`'s `FILE_HEADER`). When you point Recurse at
itself, the delimiter appears both as a real separator *and* inside the code, so
the model's naive regex truncated `engine.py` at its own quoted marker (1036 of
~6700 chars) and briefly concluded it was "just imports and constants." This is
specific to analyzing Recurse-on-Recurse; a normal target codebase has no such
collision.

---

## Timing

- **Total wall-clock:** 263.81 s (≈ 4 min 24 s)
- **Average per iteration:** 263.81 / 16 ≈ **16.5 s**
- Slowest turns were the two empty responses + the final compose (32–37 s each).
- This is local inference on a laptop GPU; a hosted API answers far faster, but
  this run cost **$0** and ran fully offline.

---

## Token economics — the honest version

There are **two different budgets**, and they point in opposite directions. Be
explicit about which one you mean.

### Budget A — total tokens across all models

> **RLM used ~14× *more* total tokens than a single stuffed call.** Expected, and
> the wrong metric for the MCP use case (see Budget B).

| Approach | Tokens | Calls |
|---|---|---|
| Naive single-shot (paste the whole 7,214-token codebase into one prompt) | ≈ **7,850** (7,214 context + question + ~600 answer) | 1 |
| **RLM (this run)** | **107,865** (102,521 in + 5,344 out) | 16 |
| **Ratio** | **≈ 13.7× more** | — |

### Budget B — *metered* tokens (the one that costs money) — RLM wins big

In the real product, Claude Code (paid Anthropic tokens) calls Recurse over MCP.
The 16-iteration, 107K-token grind runs on the **local Ollama model at $0**.
Claude only spends tokens to issue the tool call and read the final answer.

| Path | Claude's **billed** tokens | Where the 107,865 tokens went |
|---|---|---|
| Claude reads the repo itself | ≈ **7,800** (≈7,214 codebase + ~600 answer) | all on Claude (paid) |
| **Via Recurse MCP → local Ollama** | ≈ **700** (tool call + ~600-token answer) | 107,865 on local Ollama (**$0**) |
| **Reduction** | **≈ 90% fewer paid tokens (~11×)** | absorbed by free local compute |

And this is the *worst case* for the ratio — a tiny 7K repo. As the codebase
grows the metered saving widens without bound: Claude cannot ingest a 1M- or
10M-token repo in one window at all, yet Recurse still returns a ~700-token
answer. The expensive model offloads the heavy reading to free compute.

**Two honest caveats:** (1) at small scale the local model is correct but
sloppier (hallucinated line numbers) than Claude reading directly — the offload
clearly wins only once the repo is too big for Claude to read cheaply; (2) the
saving exists *because the offloaded model is free/local* — route the sub-work to
another paid API and you've just moved the bill.

- **Average input tokens per LLM call:** 102,521 / 16 ≈ **6,408** — notably this
  is *less than the full context* (7,214), confirming the root model never sees
  the whole codebase in any single request. But summed across 16 iterations
  (each re-sending accumulated history), the total dwarfs a single stuffed call.

### Where RLM actually pays off

The decomposition is overhead when the context fits the window, and a win only
when it **doesn't**:

1. **Enablement at scale.** A 1M- or 10M-token codebase can't be pasted into any
   model's prompt at all. RLM keeps each root request bounded (~6K tokens here)
   regardless of total context size — the paper shows no degradation past 10M
   tokens. Naive stuffing isn't "more expensive" there; it's *impossible*.
2. **Cost via model tiering** (not used in this run — `sub_model: same`). The
   smart/expensive root stays lean (~6K tokens/call) while a cheap sub-model
   does the bulk chunk-reading. That's where per-dollar savings appear — a future
   A/B (`root=gpt-5-mini`, `sub=gpt-5-nano`) would show it.
3. **Per-request safety.** Bounded requests are why Recurse's truncation mod
   exists and why a huge context never trips a provider's per-request token cap.

**Bottom line for messaging:** the headline here is *"a free, offline, 35B local
model answered a multi-file architecture question correctly by writing and
running its own navigation code"* — **not** "saved tokens." The token story is a
scale story, and this 7K-token example is below the crossover point.

---

## Suggested LinkedIn framing (honest)

> I asked a model running entirely on my laptop — no API, no cloud, $0 — to
> explain how my codebase selects an LLM provider. It never saw the code
> directly. Instead it loaded the repo into a Python sandbox and wrote its own
> code to grep, slice, and read its way to a correct, multi-file answer. This is
> a Recursive Language Model: the context lives in a REPL, the model navigates it
> programmatically, and the window never overflows — the same trick that scales
> to 10M+ tokens. 16 iterations, ~4 minutes, fully offline.
>
> The economic angle: when this runs as an MCP tool, my paid agent (Claude)
> doesn't read the codebase at all — it offloads the entire read to the free
> local model and pays only for the final answer. ~90% fewer metered tokens on a
> tiny repo, and unbounded savings at scale where the agent couldn't fit the repo
> in its window anyway.
>
> Honest caveat: *total* tokens across all models go up ~14×; the win is that
> those tokens are free local compute instead of metered API calls.
```
