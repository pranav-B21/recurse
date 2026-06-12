# Recurse

Recurse gives AI coding agents the ability to reason over codebases and
document sets of effectively unlimited size, using
[Recursive Language Models](https://arxiv.org/abs/2512.24601) (RLMs). It runs
as a local process and calls a hosted model API for inference — no GPU
required.

It ships in two forms sharing one engine:

- **CLI:** `recurse "explain the auth flow" --path ./my-project`
- **MCP server:** Claude Code delegates huge-context questions to it, with
  memory that persists across sessions.

## How it works

An RLM wraps a language model so it can handle context far larger than its
window:

1. The huge context is loaded into a Python REPL as a variable — **never**
   sent to the model directly.
2. The root LM sees only the query + metadata, and writes Python to navigate
   the context: peek, grep, partition + map over sub-LM calls, summarize.
3. The loop ends when the model submits its final answer.

Recurse builds on a vendored fork of the official
[rlm library](https://github.com/alexzhang13/rlm) (`vendor/rlm`, mods tagged
`RECURSE-MOD`) and adds what the library doesn't have: directory ingestion,
persistent threads with conversation memory, provider presets with guardrails,
a CLI, and an MCP server.

## Install (dev)

```bash
python -m venv recurse/.venv && source recurse/.venv/bin/activate
pip install -e vendor/rlm    # the FORK — never `pip install rlms` from PyPI
pip install -e .
recurse init                 # writes ~/.recurse/config.yaml
```

## Providers

API keys come from env vars only — never stored in the config file.

| Provider | Models (root / sub) | Notes |
|---|---|---|
| `groq` (default) | `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` | Free tier, but a **12K tokens-per-minute cap per request** — small contexts (≤ ~30K chars) only. Dev smoke tests. `export GROQ_API_KEY=...` |
| `openai` (recommended) | `gpt-5-mini` / `gpt-5-nano` | The paper's benchmarked config; cents per query. `export OPENAI_API_KEY=...` ⚠️ **ChatGPT Pro ≠ API credits** — load ~$10 at platform.openai.com/billing first. |
| `anthropic` | `claude-sonnet-4-6` / `claude-haiku-4-5` | Needs console.anthropic.com credits (separate from a Claude.ai subscription). `export ANTHROPIC_API_KEY=...` |

Switch providers in `~/.recurse/config.yaml` (`provider: openai`) or per run
with `--provider`.

## CLI

```bash
recurse "what does the engine do?" --path ./recurse --thread self
recurse "and how does it persist turns?" --thread self     # remembers the previous turn
recurse ingest ./my-project --thread proj
recurse threads                  # list; `recurse threads delete NAME`
recurse history --thread proj
recurse init
```

## MCP server (Claude Code)

```bash
claude mcp add recurse --transport stdio -- \
    /ABS/PATH/TO/recurse/recurse/.venv/bin/python -m recurse.server
```

Tools: `recurse_query(query, context_source, thread_id)` with
`context_source` of `path:/abs/dir` | `thread:<id>` | `inline:<text>`,
`recurse_ingest(path, thread_id)`, `recurse_threads(action, thread_id)`.

## Notes & gotchas

- One RLM query = many LM calls (root per iteration + sub per chunk). Keep
  `max_iterations` modest and watch the usage footer.
- `environment: local` runs LM-generated code in-process — fine for trusted
  personal use. Use `environment: docker` for untrusted contexts.
- Groq HTTP 413 `rate_limit_exceeded` means the per-request 12K-token cap was
  hit; Recurse's guardrail refuses oversized contexts before spending a call.
- OpenAI `insufficient_quota` means a $0 credit balance, not a rate limit.

## License

MIT. The vendored `rlm` library is MIT (upstream license preserved at
`vendor/rlm/LICENSE`).
