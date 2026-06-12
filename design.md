# DESIGN.md — Recurse

> Source of truth for this project. Claude Code: read this file top-to-bottom before writing any code. Also read the "Gotchas" section twice.

---

## 1. What This Is

Recurse gives AI coding agents the ability to reason over codebases and document sets of effectively unlimited size, using Recursive Language Models (RLMs). It runs as a local process and calls a hosted model API for inference — no GPU required.

It ships in two forms sharing one engine:

- **CLI:** `recurse "explain the auth flow" --path ./my-project`
- **MCP server:** `claude mcp add recurse` → Claude Code delegates huge-context questions to it, with memory that persists across sessions.

This recreates the user-facing behavior of [Monolith](https://github.com/WingchunSiu/Monolith) (TreeHacks 2026 winner: RLM-as-a-service MCP tool with persistent memory) — but as a plain local process instead of Modal + Cloudflare infrastructure, and built on a **vendored fork** of the official RLM library that we can read and modify.

---

## 2. Current State — What Is ALREADY Done (do not redo)

The environment is set up and verified. Claude Code should build on this, not recreate it.

**Repo root:** `~/Documents/GitHub/recurse/`

```
recurse/                        # repo root
├── CLAUDE.md                   # points Claude Code here
├── DESIGN.md                   # this file
├── README.md
├── pyproject.toml
├── recurse/                    # Python package dir (target for our code)
│   └── .venv/                  # the venv lives here (quirk — harmless, gitignored)
├── vendor/
│   └── rlm/                    # FORK of alexzhang13/rlm — editable-installed, MODIFIABLE
├── examples/
└── tests/
```

**Verified facts:**

1. `vendor/rlm` is editable-installed into the venv. Confirmed:
   ```bash
   python -c "import rlm; print(rlm.__file__)"
   # → .../recurse/vendor/rlm/rlm/__init__.py   ✓
   ```
   Edits to `vendor/rlm/rlm/*.py` take effect immediately — no reinstall.

2. **The RLM loop works end-to-end on this machine.** A smoke test via Groq (free tier, `llama-3.3-70b-versatile`) ran 14 iterations: the root LM peeked at the context, chunked it, grepped, and executed code in the REPL. The mechanism is proven.

3. **Two failure modes were observed and must inform the build** (see §6 and §7):
   - Groq free tier has a **12,000 tokens-per-minute hard cap per request**. The run died with `413 rate_limit_exceeded` when accumulated conversation history + a huge REPL output pushed one request to 12,226 tokens.
   - The root cause of the blowup: the model printed an entire **40,000-character chunk** into REPL output, which the library fed back into the next root call **untruncated (or insufficiently truncated)**. Output truncation is mandatory work.
   - `llama-3.3-70b-versatile` **flails at the RLM protocol** — it never found the needle, fixated on irrelevant searches, and ignored the "don't print the whole context" instruction. Open models need more directive prompts; `gpt-5-mini` is the reliable choice.

4. The user currently has **no OpenAI API credits** (ChatGPT Pro ≠ API credits). Groq (free) works for small tests now; OpenAI `gpt-5-mini` is the target once ~$10 of credits are loaded. The design is provider-agnostic via config so this is a config swap, not a code change.

---

## 3. How RLMs Work (mechanism reference)

Sources: [paper](https://arxiv.org/abs/2512.24601) · [blog](https://alexzhang13.github.io/blog/2025/rlm/) · [repo](https://github.com/alexzhang13/rlm)

An RLM wraps a language model so it can handle context far larger than its window. `rlm.completion(prompt=context, root_prompt=query)` is a drop-in for a normal LM call. Internally:

1. The huge **context** is loaded into a Python REPL as a variable — **never** sent to the model directly.
2. The **root LM** (depth 0) sees only the query + metadata (context exists, its size).
3. The root LM writes Python into the REPL to navigate the context. Emergent strategies: **peek** (`context[:2000]`), **grep** (regex/keyword filters), **partition + map** (chunk it, send chunks to a sub-LM), **summarize** (fold results).
4. The REPL exposes `llm_query(query, context)` and `batch_llm_query(...)` — the recursive sub-calls (depth 1, the current max).
5. The loop ends when the root LM emits `FINAL(answer)` or `FINAL_VAR(varname)`, or `max_iterations` is hit.

Headline result: RLM(GPT-5-mini) beat plain GPT-5 by ~34 points on the 132k-token OOLONG split at similar cost, with no degradation past 10M tokens. The root LM's real job is **writing good navigation code** — its own context never clogs up.

---

## 4. The Vendored RLM Library

We build ON the fork in `vendor/rlm`, never reimplement the loop. Verified public API (exact signature used in the working smoke test):

```python
from rlm import RLM
from rlm.logger import RLMLogger

rlm = RLM(
    backend="openai",                      # also: "anthropic", "openrouter", "portkey", "litellm", "vllm"
    backend_kwargs={
        "api_key": "...",
        "model_name": "gpt-5-mini",
        # "base_url": "https://api.groq.com/openai/v1",   # optional — any OpenAI-compatible host
    },
    environment="local",                   # REPL runs in-process; also "docker", "modal"
    max_iterations=30,                     # library default observed: 30; max_depth default: 1
    logger=RLMLogger(log_dir="./logs"),    # optional JSONL trajectories (works with their visualizer)
    verbose=True,                          # rich console output of iterations
    # other_backends=["openai"],                          # EXPERIMENTAL: separate sub-LM
    # other_backend_kwargs=[{...}],                       # must match other_backends order
)

result = rlm.completion(prompt=huge_context, root_prompt="the question")
result.response          # final answer string
result.execution_time    # seconds
result.usage_summary     # aggregated token usage
```

**Internal files (paths verified from live tracebacks on this machine):**

| File | What's there | Why we care |
|---|---|---|
| `vendor/rlm/rlm/core/rlm.py` | `completion()` (~line 402) — the main loop; `_completion_turn()` (~line 655) — one iteration: LM call → parse → execute | Where iteration history accumulates; where REPL output is appended back to the conversation → **truncation mod lives here or adjacent** |
| `vendor/rlm/rlm/core/lm_handler.py` | Routes completions to the right client (`get_client(model).completion(prompt)`) | Where root vs sub model routing happens |
| `vendor/rlm/rlm/clients/openai.py` | OpenAI-compatible client (~line 111: `chat.completions.create`), honors `base_url`, has `_normalize_sampling_args` | Confirms Groq/any OpenAI-compatible host works via `base_url` |

**What the library does NOT provide — this is the entire Recurse build:**

| Gap | Recurse adds |
|---|---|
| No way to turn a folder into context | `ingest` — walk a directory, concatenate files with `=== FILE:` headers |
| No persistence between runs | Thread store on local disk (`~/.recurse/threads/`) |
| No conversation memory | Append each Q&A turn to the thread context (the Monolith feature) |
| No CLI | `recurse "query" --path ./dir` |
| No agent integration | MCP server exposing tools to Claude Code |
| Insufficient REPL-output truncation | **Vendor mod #1** — hard clamp, configurable (§7) |
| Single model for root + sub by default | Root/sub split via `other_backends` (experimental — verify, with single-model fallback) |

---

## 5. Architecture

```
   Claude Code                    Terminal
        │ MCP (stdio)                │ args
        ▼                           ▼
  recurse/server.py           recurse/cli.py
        └────────────┬──────────────┘
                     ▼
            recurse/engine.py  (RecurseEngine)
              • resolve context (thread:/path:/inline:)
              • build RLM() from config
              • guardrails (size estimate vs provider limits)
              • rlm.completion(prompt=context, root_prompt=query)
              • persist the Q&A turn
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
   vendor/rlm  (fork)       recurse/store.py
   loop + REPL + sub-calls  ~/.recurse/threads/{id}/
         │                    context.txt, turns/, manifest.json
         ▼
   Hosted API (Groq free / OpenAI / Anthropic)
```

Real logic = five files: `config.py`, `engine.py`, `store.py`, `cli.py`, `server.py`. Everything else is the vendored library.

---

## 6. Provider & Model Strategy

Provider is a config choice, not a code path. The engine constructs `backend_kwargs` from a preset.

| Provider | Models (root / sub) | Cost | Verdict from live testing |
|---|---|---|---|
| **Groq** (current) | `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` | Free tier | Loop works. **12K TPM cap per request** → only viable for small contexts (≤ ~30K chars) and short runs. Model flails at protocol — needs directive prompts. Use for dev smoke tests only. |
| **OpenAI** (target) | `gpt-5-mini` / `gpt-5-nano` | ~cents/query | The paper's benchmarked config; follows the protocol reliably; generous limits. **Requires loading API credits (~$10)** — ChatGPT Pro does not include API access. Switch here for real work. |
| **Anthropic** | `claude-sonnet-4-6` / `claude-haiku-4-5` | low | Native `backend="anthropic"`. Needs console.anthropic.com credits (separate from Claude.ai subscription). |

**Engine guardrail (required):** before running, estimate context tokens (`len(context) // 4`). If the active provider preset declares a `tpm_limit` and the estimate exceeds ~60% of it, refuse with a clear error telling the user to shrink the context or switch provider. This converts the cryptic 413 we hit into an actionable message.

**Root/sub split:** attempt `other_backends` per the API above. If it errors or behaves oddly in this library version (it's experimental), fall back to a single model for both and log a warning. Engine takes a `sub_model: same` option to force the fallback.

---

## 7. Vendor Modifications (the point of the fork)

Keep mods minimal, surgical, and documented. Each mod gets a `# RECURSE-MOD:` comment at the site.

### Mod #1 — REPL output truncation (REQUIRED, Phase 2)

**Problem (observed live):** the model printed a 40K-char chunk; the library fed it back into the next root call; accumulated history blew Groq's per-request token cap (413).

**Fix:** in `vendor/rlm/rlm/core/rlm.py` (locate where code-execution output is appended to the message history inside `_completion_turn` / the loop in `completion`), clamp the output before appending:

```python
# RECURSE-MOD: clamp REPL output fed back to the root LM
MAX_OUTPUT_CHARS = int(os.getenv("RLM_MAX_OUTPUT_CHARS", "4000"))
if len(output) > MAX_OUTPUT_CHARS:
    output = (
        output[:MAX_OUTPUT_CHARS]
        + f"\n...[truncated {len(output) - MAX_OUTPUT_CHARS} chars — slice or filter instead of printing large objects]"
    )
```

Env-var driven so the engine can set it per provider (4000 for Groq, 20000 for OpenAI). The truncation notice doubles as feedback that steers the model away from re-printing.

If the library already truncates somewhere, **lower its limit and make it env-configurable** rather than adding a second mechanism. Search the codebase for existing truncation first.

### Mod #2 — directive system prompt for weak models + codebase contexts (Phase 2)

Find the root system prompt (search `vendor/rlm/rlm/` for the prompt template — likely under `core/` or a `prompts` module). Add, behind an env flag or by appending via whatever extension hook exists (`custom_system_prompt` kwarg if present — check the RLM constructor):

- The context may be formatted as `=== FILE: path ===` blocks; grep those markers to navigate.
- NEVER print more than 2,000 chars of the context at once; slice and filter.
- Prefer regex search for the literal task keywords before chunking blindly.
- When confident, emit `FINAL(answer)` immediately — do not keep exploring.

Prefer a constructor kwarg over editing the template if the library supports one; edit the vendored template only if not.

### Future mods (roadmap, NOT v1): 

`llm_query` interception for caching and prompt-injection screening — both live at the same choke point where `llm_query` is defined/dispatched (likely `core/` or the environment module). Locate and note the file path during Phase 2, implement in roadmap phase.

---

## 8. Module Specs

### `recurse/config.py`

```python
from dataclasses import dataclass, field
from pathlib import Path
import os, yaml

PRESETS = {
    "groq": {
        "backend": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "root_model": "llama-3.3-70b-versatile",
        "sub_model": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY",
        "tpm_limit": 12000,            # free tier — engine guardrail uses this
        "max_output_chars": 4000,
    },
    "openai": {
        "backend": "openai",
        "base_url": None,
        "root_model": "gpt-5-mini",
        "sub_model": "gpt-5-nano",
        "api_key_env": "OPENAI_API_KEY",
        "tpm_limit": None,
        "max_output_chars": 20000,
    },
    "anthropic": {
        "backend": "anthropic",
        "base_url": None,
        "root_model": "claude-sonnet-4-6",
        "sub_model": "claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "tpm_limit": None,
        "max_output_chars": 20000,
    },
}

@dataclass
class RecurseConfig:
    provider: str = "groq"             # current default; switch to "openai" once credits loaded
    root_model: str | None = None      # override preset
    sub_model: str | None = None       # override preset; "same" forces single-model mode
    max_iterations: int = 20
    environment: str = "local"         # local | docker
    verbose: bool = True
    storage_path: Path = Path.home() / ".recurse" / "threads"
    ingest_exclude: list[str] = field(default_factory=lambda: [
        "node_modules", ".git", "__pycache__", ".venv", "dist", "build",
        "*.lock", "*.pyc", ".DS_Store", "vendor",
    ])
    max_file_size_kb: int = 500
    max_total_files: int = 5000

    def resolved(self) -> dict:
        """Preset merged with overrides + api key from env. Raise a clear error if key missing."""
        ...

def load_config() -> RecurseConfig:
    """~/.recurse/config.yaml over defaults; create file with commented defaults if absent."""
    ...
```

### `recurse/store.py`

```python
class ContextStore:
    """
    ~/.recurse/threads/{thread_id}/
        context.txt       # concatenated files + appended conversation turns (what the RLM sees)
        manifest.json     # ingested file list: path, size, sha256
        turns/{iso_ts}.json  # structured Q&A records
    """
    def __init__(self, base_path: Path): ...

    def ingest_directory(self, path: Path, thread_id: str, exclude, max_file_size_kb, max_total_files) -> "IngestResult":
        # Walk dir; skip excluded/binary/oversized; write context.txt as:
        #   === FILE: relative/path.py ===
        #   <contents>
        # Build manifest.json. Return counts + token estimate (chars//4) + compact file tree string.
        ...

    def load_context(self, thread_id: str) -> str: ...
    def append_turn(self, thread_id: str, query: str, answer: str, meta: dict):
        # 1) turns/{ts}.json  2) append to context.txt:
        #    === CONVERSATION TURN {ts} ===
        #    Q: ...
        #    A: ...
        ...
    def get_turns(self, thread_id: str, limit: int = 10) -> list[dict]: ...
    def list_threads(self) -> list[dict]: ...
    def delete_thread(self, thread_id: str): ...
    def has_thread(self, thread_id: str) -> bool: ...
```

Binary detection: try `bytes.decode("utf-8")`; on failure, skip the file and count it in `skipped`.

### `recurse/engine.py`

```python
import os
from rlm import RLM
from rlm.logger import RLMLogger
from recurse.store import ContextStore

class RecurseEngine:
    def __init__(self, config):
        self.config = config
        self.p = config.resolved()                       # provider preset + overrides + api key
        self.store = ContextStore(config.storage_path)

    def _build_rlm(self) -> RLM:
        os.environ["RLM_MAX_OUTPUT_CHARS"] = str(self.p["max_output_chars"])   # drives vendor Mod #1
        bk = {"api_key": self.p["api_key"], "model_name": self.p["root_model"]}
        if self.p["base_url"]:
            bk["base_url"] = self.p["base_url"]

        kwargs = dict(
            backend=self.p["backend"], backend_kwargs=bk,
            environment=self.config.environment,
            max_iterations=self.config.max_iterations,
            logger=RLMLogger(log_dir=str(self.config.storage_path / "_logs")),
            verbose=self.config.verbose,
        )
        if self.p["sub_model"] not in (None, "same", self.p["root_model"]):
            sbk = dict(bk, model_name=self.p["sub_model"])
            kwargs.update(other_backends=[self.p["backend"]], other_backend_kwargs=[sbk])
        try:
            return RLM(**kwargs)
        except TypeError:
            # other_backends experimental — fall back to single model
            kwargs.pop("other_backends", None); kwargs.pop("other_backend_kwargs", None)
            return RLM(**kwargs)

    def _guardrail(self, context: str):
        est = len(context) // 4
        tpm = self.p.get("tpm_limit")
        if tpm and est > 0.6 * tpm:
            raise RuntimeError(
                f"Context ≈{est} tokens exceeds safe budget for provider '{self.config.provider}' "
                f"(TPM limit {tpm}). Shrink the context or switch provider (e.g. provider: openai)."
            )

    def query(self, query: str, context_source: str, thread_id: str = "default"):
        context = self._resolve_context(context_source, thread_id)
        self._guardrail(context)
        result = self._build_rlm().completion(prompt=context, root_prompt=query)
        self.store.append_turn(thread_id, query, result.response, {
            "execution_time": result.execution_time,
            "usage": str(result.usage_summary),
            "provider": self.config.provider,
        })
        return result

    def ingest(self, path: str, thread_id: str = "default"):
        return self.store.ingest_directory(
            Path(path), thread_id,
            exclude=self.config.ingest_exclude,
            max_file_size_kb=self.config.max_file_size_kb,
            max_total_files=self.config.max_total_files,
        )

    def _resolve_context(self, source: str, thread_id: str) -> str:
        # "thread:<id>" → load; "path:<dir>" → ingest then load; "inline:<text>" → text;
        # bare string that is an existing path → ingest+load; else treat as inline.
        ...
```

Everything is **synchronous** — `rlm.completion()` blocks; FastMCP tolerates sync tools. No asyncio in v1.

### `recurse/cli.py`

Commands (argparse, `rich` for output):

```
recurse "<query>" [--path DIR] [--thread NAME] [--provider groq|openai|anthropic]
recurse ingest DIR [--thread NAME]
recurse threads [delete NAME]
recurse history [--thread NAME] [--limit N]
recurse init          # write ~/.recurse/config.yaml with commented defaults; print API-key setup steps per provider
```

`--provider` overrides config for one run (handy while on Groq). Query flow: spinner via `rich.console.Console().status(...)`, print `result.response`, then a dim footer with time + usage. Entry point: `recurse = "recurse.cli:main"` in pyproject.

### `recurse/server.py` (MCP — use FastMCP)

```python
from mcp.server.fastmcp import FastMCP
from recurse.config import load_config
from recurse.engine import RecurseEngine

mcp = FastMCP("recurse")
engine = RecurseEngine(load_config())

@mcp.tool()
def recurse_query(query: str, context_source: str, thread_id: str = "default") -> str:
    """Answer a question over a large codebase/document set using recursive reasoning
    with persistent per-thread memory. context_source: "path:/abs/dir" | "thread:<id>" | "inline:<text>"."""
    r = engine.query(query, context_source, thread_id)
    return f"{r.response}\n\n---\n{r.execution_time:.1f}s | {r.usage_summary}"

@mcp.tool()
def recurse_ingest(path: str, thread_id: str = "default") -> str:
    """Index a directory into a persistent thread for future queries."""
    res = engine.ingest(path, thread_id)
    return f"Ingested {res.files_count} files (~{res.token_estimate:,} tokens) into '{thread_id}'.\n{res.file_tree}"

@mcp.tool()
def recurse_threads(action: str = "list", thread_id: str | None = None) -> str:
    """Manage threads: action = list | history | delete."""
    ...

if __name__ == "__main__":
    mcp.run()   # stdio
```

Register: `claude mcp add recurse --transport stdio -- python -m recurse.server` (run with the project venv's python — use the absolute venv python path in the command if needed).

---

## 9. Build Phases (each ends in a runnable verification — do not skip)

### Phase 0 — clean smoke pass (½ hr) — partially done
The loop already runs; get one **clean needle success** on Groq with a small context and a directive prompt:
```bash
export GROQ_API_KEY=gsk_...
python - <<'EOF'
from rlm import RLM
import os
rlm = RLM(backend='openai',
    backend_kwargs={'api_key': os.environ['GROQ_API_KEY'],
                    'base_url': 'https://api.groq.com/openai/v1',
                    'model_name': 'llama-3.3-70b-versatile'},
    environment='local', verbose=True)
ctx = ('apple banana cherry ' * 600) + '\nThe secret code is PURPLE-ELEPHANT-42.\n' + ('mango grape kiwi ' * 600)
r = rlm.completion(prompt=ctx, root_prompt='Find the secret code (format WORD-WORD-NUMBER). Grep the context for the word "secret". Emit FINAL(code) as soon as you find it.')
print('ANSWER:', r.response)
EOF
```
**Accept:** response contains `PURPLE-ELEPHANT-42`. (If the model still flails, that's signal for Mod #2's prompt work, not a blocker — proceed.)

### Phase 1 — config + engine
Build `config.py` + `engine.py` (specs above), including guardrail + preset system + `other_backends` fallback.
**Accept:** a 5-line script: `RecurseEngine(load_config()).query("find the secret code…", "inline:" + ctx)` returns the needle. Guardrail check: feed a 200K-char context on provider=groq → clean RuntimeError with the helpful message, **no API call made**.

### Phase 2 — vendor mods
Implement Mod #1 (truncation, env-driven) and Mod #2 (directive prompt). Add `vendor/rlm/RECURSE-MODS.md` listing each mod: file, line area, rationale, env vars.
**Accept:** rerun a needle test where the root LM is induced to print a big chunk (e.g., 60K-char context on OpenAI later, or assert by unit-testing the truncation function directly); confirm output in the conversation history is clamped to the configured limit. Grep `vendor/rlm` for `RECURSE-MOD` returns both sites.

### Phase 3 — store + ingest
Build `store.py`.
**Accept:**
```python
store.ingest_directory(Path("./recurse"), "self", ...)        # ingest our own package
ctx = store.load_context("self")
assert "=== FILE: " in ctx and "engine.py" in ctx
```
Then end-to-end: `engine.query("what does engine.py do?", "thread:self")` on Groq (our package is small enough) returns a grounded answer.

### Phase 4 — CLI
Build `cli.py` + pyproject entry point (`pip install -e .` for the recurse package itself).
**Accept:** fresh terminal → `recurse init`, then `recurse "what does this project do?" --path ./recurse --thread self` prints an answer; `recurse threads` lists `self`; `recurse history --thread self` shows the turn.

### Phase 5 — persistent memory (the Monolith feature)
`append_turn` already wires it; verify the loop closes.
**Accept:** ask Q1, then a follow-up Q2 that depends on Q1's answer in the same thread; A2 reflects A1. `~/.recurse/threads/self/turns/` has two files; `context.txt` ends with two `=== CONVERSATION TURN` blocks.

### Phase 6 — MCP server
Build `server.py`; register with Claude Code.
**Accept:** in Claude Code: "use recurse to ingest <dir> and then ask it how X works" → it calls `recurse_ingest` then `recurse_query` and relays the answer. This is Monolith's user-facing behavior, achieved locally.

### Phase 7 — polish & ship v0.1
README (setup per provider incl. "ChatGPT Pro ≠ API credits" note), error handling (missing API key → name the env var; empty dir; Ollama-style helpful messages), `.gitignore` (`.venv/`, `__pycache__/`, `logs/`, `*.jsonl`), demo GIF, tag v0.1.0.

---

## 10. Configuration File

`~/.recurse/config.yaml` (created by `recurse init`):

```yaml
provider: groq              # groq (free, small contexts) | openai (recommended) | anthropic
# root_model: gpt-5-mini    # optional per-model overrides
# sub_model: same           # "same" forces single-model mode if other_backends misbehaves

max_iterations: 20
environment: local          # local | docker
verbose: true

storage_path: ~/.recurse/threads

ingest:
  exclude: [node_modules, .git, __pycache__, .venv, dist, build, vendor, "*.lock", "*.pyc"]
  max_file_size_kb: 500
  max_total_files: 5000
```

API keys come from env vars only (`GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) — never stored in the file.

---

## 11. Repo Hygiene & Packaging

- **Vendor git:** `vendor/rlm` was cloned from upstream and has its own `.git`. Pick ONE: **(recommended)** `rm -rf vendor/rlm/.git`, commit the code directly into this repo, and create `vendor/rlm/VENDORED.md` recording upstream URL + the commit hash vendored from (for manual future syncs). Alternative: make it a proper git submodule of a GitHub fork — only if the user wants easy upstream merges; skip otherwise.
- **License:** upstream is MIT — keep `vendor/rlm/LICENSE` intact; Recurse itself is MIT.
- **pyproject (recurse):** do **NOT** list `rlms` as a dependency — a fresh `pip install` would pull the unmodified PyPI package and shadow the fork. Document the two-step dev install in README instead:
  ```bash
  pip install -e vendor/rlm && pip install -e .
  ```
  Deps for recurse itself: `mcp>=1.0.0`, `pyyaml>=6.0`, `rich>=13.0.0`. Entry point: `recurse = "recurse.cli:main"`. `requires-python = ">=3.12"`.
- **CLAUDE.md:** ensure it contains: "Read DESIGN.md in full before any task. The RLM library is vendored at vendor/rlm (editable-installed, modifiable — mods are tagged RECURSE-MOD)."
- **.gitignore:** `.venv/`, `recurse/.venv/`, `__pycache__/`, `logs/`, `*.jsonl`, `.DS_Store`.

---

## 12. Gotchas (Claude Code: read twice)

1. **Activate the right venv:** it lives at `recurse/.venv` *inside the inner package folder* (path: `<repo>/recurse/.venv`). Use its python explicitly when in doubt.
2. **Never `pip install rlms` from PyPI** — it shadows the fork. If the import path ever stops pointing at `vendor/rlm`, run `pip uninstall rlms -y && pip install -e vendor/rlm` and re-verify with `python -c "import rlm; print(rlm.__file__)"`.
3. **Groq 413 ≠ bug:** `rate_limit_exceeded` on tokens means the request (context + accumulated history + output echo) exceeded 12K tokens. The guardrail + truncation mod exist precisely for this. Keep Groq test contexts ≤ ~30K chars.
4. **Insufficient_quota on OpenAI ≠ rate limit:** it means $0 credit balance. ChatGPT Pro does not grant API credits.
5. **Weak models flail:** on Groq, make `root_prompt`s directive (tell it to grep for specific keywords, tell it to emit FINAL immediately). Don't tune the architecture around llama's confusion — `gpt-5-mini` resolves it.
6. **`other_backends` is experimental:** always keep the single-model fallback path working.
7. **`exec()` runs LM-generated code in-process** (`environment="local"`). Fine for trusted personal use; `environment="docker"` is the one-line upgrade for untrusted contexts. Never run `local` against contexts you don't trust.
8. **Costs:** one RLM query = many LM calls (root per iteration + sub per chunk). Keep `max_iterations` modest (20) and watch `usage_summary`.

---

## 13. Roadmap (after v0.1 — all additive, none require rearchitecting)

1. **Sub-call cache** — intercept at the `llm_query` choke point in the vendored source; key `sha256(query + chunk)`; store under thread `cache/`. Highest value-per-effort: repeated queries on stable codebases get drastically cheaper.
2. **Prompt-injection screener** — same choke point: screen each decomposed chunk for injected instructions before it reaches the sub-LM. Small decomposed pieces = cleaner detection surface than one monolithic prompt. Novel research angle and standalone-product potential.
3. **Modal Sandbox environment** — the user's explicit learning milestone: swap the local `exec()` REPL for `modal.Sandbox` isolation (`environment="modal"` exists upstream; study it, then extend). Teaches real serverless isolation; mirrors Monolith's `ModalSandboxSubRLM`.
4. **Batch sub-calls** — push the prompt to use `batch_llm_query` for independent chunks (parallelism).
5. **Recursion depth > 1** — upstream caps at 1; the paper notes deeper recursion unlocks harder tasks.
6. **One-shot multi-file edits** — extend from answering questions to proposing/applying consistent changes across a codebase (the user's "sprint team / one-shot platform" idea).
7. **Cloud + desktop** — optional Modal-hosted backend (the monetization seam: free local, paid cloud) and a drag-a-folder desktop app. Last, not first.

---

## 14. Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Vendored fork vs pip dependency | Fork in `vendor/rlm`, editable-installed | User explicitly wants to learn/modify internals; mods (truncation, prompts, later cache/screener/depth) need source access. Verified working on this machine. |
| Provider model | Config presets: Groq free now → OpenAI `gpt-5-mini` for real work | No API credits yet; Groq proves the pipeline free; gpt-5-mini is the paper's benchmarked, protocol-reliable config. Swap = one yaml line. |
| Truncation as vendor Mod #1 | Hard clamp, env-configurable | Directly caused the observed 413 failure; also protects cost on any provider. |
| Guardrail pre-check | Estimate tokens vs preset TPM | Converts cryptic provider errors into actionable messages before spending a call. |
| Sync everywhere | No asyncio in v1 | `rlm.completion()` is blocking; FastMCP accepts sync tools; simplicity wins. |
| Local-disk persistence | `~/.recurse/threads/` | Recreates Monolith's cross-session memory without Modal Volumes. |
| CLI first, MCP second | Shared engine | CLI testable in isolation; MCP rides Claude Code. Mirrors the phase order. |
| `environment="local"` default | docker as config option | Personal trusted use; flagged in Gotchas #7. |
| MIT license | matches upstream | Maximum adoption; upstream LICENSE preserved in vendor/. |