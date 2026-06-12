# RECURSE-MODS — local modifications to the vendored rlm fork

Each mod is tagged with a `# RECURSE-MOD:` comment at the site.
Find them all with: `grep -rn "RECURSE-MOD" vendor/rlm/rlm/`

## Mod #1 — REPL output truncation, env-configurable

- **File:** `rlm/utils/parsing.py` (`format_iteration`, top of file)
- **What:** Upstream already truncated per-block REPL output at a hardcoded
  20,000 chars. The limit is now read from the `RLM_MAX_OUTPUT_CHARS` env var
  (fallback: 20000, preserving upstream behavior when unset), and the
  truncation notice tells the model to slice/filter instead of re-printing
  large objects.
- **Why:** Observed live failure: the root LM printed a 40K-char chunk, the
  output was fed back into the next root call, and accumulated history blew
  Groq's 12,000 tokens-per-minute per-request cap (HTTP 413). The Recurse
  engine sets `RLM_MAX_OUTPUT_CHARS` per provider preset (4000 for Groq,
  20000 for OpenAI/Anthropic).
- **Env vars:** `RLM_MAX_OUTPUT_CHARS` — max chars of a single REPL block's
  output fed back to the root LM.

## Mod #2 — directive system prompt (via constructor hook, not a template edit)

- **Files:** `rlm/utils/prompts.py` (marker comment only) +
  `recurse/engine.py` (`DIRECTIVE_ADDENDUM`)
- **What:** The library supports a `custom_system_prompt` kwarg on `RLM(...)`,
  so per the design ("prefer a constructor kwarg over editing the template"),
  Recurse appends a directive addendum to the stock `RLM_SYSTEM_PROMPT` and
  passes the combined prompt in. The addendum tells the model: the context may
  be `=== FILE: path ===` blocks (grep those markers), never print >2,000
  chars of context at once, regex-search literal task keywords before chunking
  blindly, and submit via `answer["ready"] = True` as soon as it is confident.
- **Why:** Weak open models (observed: `llama-3.3-70b-versatile`) flail at the
  RLM protocol — they fixate on irrelevant searches and re-print huge chunks.
  Directive instructions measurably steer them; harmless for strong models.
- **Note:** This library version finalizes via the `answer` dict
  (`answer["ready"] = True`), not the paper's `FINAL(...)` syntax — the
  addendum matches the real protocol.

## Roadmap mods (located, NOT implemented — see design.md §13)

- **`llm_query` choke point** for the sub-call cache and prompt-injection
  screener: `rlm/environments/local_repl.py` — `_llm_query` (~line 258) and
  `_llm_query_batched` (~line 282); both registered into REPL globals at
  ~line 224.
