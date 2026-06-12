# Implement design.md (vendored-RLM pivot)

The new design.md replaces the old custom Qwen/Ollama implementation
(`recurse/engine/`, `recurse/store/`, `recurse/tools/`, `recurse/hooks/`)
with five flat modules built on the vendored fork at `vendor/rlm`.
Old code is recoverable from git (commit b4032ae).

## Plan

- [x] Survey repo: venv OK (recurse/.venv, py3.13), `import rlm` → vendor fork,
      GROQ_API_KEY in .env, vendor/rlm has embedded .git @ 156fd72.
- [x] Vendor hygiene: `vendor/rlm/VENDORED.md` (upstream URL + commit),
      embedded `.git` removed.
- [x] Vendor Mod #1: existing truncation in `rlm/utils/parsing.py` made
      env-configurable via `RLM_MAX_OUTPUT_CHARS` (fallback 20000 = upstream),
      steering truncation notice. Tagged `RECURSE-MOD`.
- [x] Vendor Mod #2: directive prompt via `custom_system_prompt` constructor
      kwarg (design's preferred hook). `DIRECTIVE_ADDENDUM` in
      `recurse/engine.py`; marker comment in vendored `prompts.py` so
      `grep RECURSE-MOD` finds both sites. Adapted to the real protocol
      (`answer["ready"] = True`, not the paper's `FINAL()`).
- [x] `vendor/rlm/RECURSE-MODS.md` + roadmap choke point
      (`environments/local_repl.py` `_llm_query` ~line 258).
- [x] New `recurse/{config,store,engine,cli,server}.py`; old modules removed.
- [x] `pyproject.toml`, `.gitignore`, `CLAUDE.md` note, `README.md`, `examples/`.
- [x] Tests: 35 passing (config, store, engine incl. guardrail-without-API-call,
      vendor truncation, MCP tool wiring).
- [x] `pip install -e .`; CLI entry point works.
- [x] Live Groq: inline needle (PURPLE-ELEPHANT-42 in 2.6s), ingest self →
      grounded answer (56s), memory loop (2 turns persisted, A2 recalls A1),
      `recurse threads`/`history`, MCP stdio handshake + tools/list.
- [x] MCP re-registered with absolute venv python (local scope, connected).

## Review

- Engine resolves provider preset lazily so store-only commands
  (`threads`, `history`) need no API key.
- CLI catches exceptions at the boundary (message, not traceback) after a
  Groq daily-cap 429 dumped a stack trace.
- Groq free tier also has a ~100K tokens/DAY cap (beyond the 12K TPM the
  guardrail handles) — exhausted during testing; CLI query retried after the
  window reset.
- CLI verified end-to-end: re-ingesting ./recurse trips the guardrail with the
  clean message (package grew to ≈7.2K est tokens, threshold 7,200 — expected
  on Groq); query on ./examples returned a grounded answer (39.2s, footer OK).
- OUTSTANDING (user action): global ~/.claude/settings.json has Stop +
  SessionEnd hooks running `python -m recurse.hooks.upload_session` (module
  deleted with the old design) — they fail on every session; permission
  classifier blocked me from editing that file.
- Old tracked `__pycache__/*.pyc` files show as deleted in git status;
  .gitignore now excludes them going forward.
