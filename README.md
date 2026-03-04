# Recurse — RLM-Powered Unlimited Context MCP Server

Give Claude Code unlimited context using local Qwen 3.5 models. When a codebase or document set is too large for Claude's context window, delegate to Recurse.

```bash
claude mcp add recurse --transport stdio -- python -m recurse.server
```

## How It Works

Based on [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601):

1. Context is stored as a Python variable — **never fed into the LLM directly**
2. A root LLM (Qwen 3.5 35B-A3B) writes Python code to inspect and decompose the context
3. The root LLM calls `llm_query()` to delegate focused analysis to a sub-LLM (Qwen 3.5 9B)
4. Loop continues until `FINAL(answer)` is output or max iterations reached

The root LLM never sees the full context. It navigates it programmatically. This is how a 3B-active-parameter model can reason over 10M+ tokens.

## Quick Start

```bash
# 1. Install Ollama and pull models
ollama pull qwen3.5:35b-a3b
ollama pull qwen3.5:9b

# 2. Install Recurse
pip install -e .

# 3. Register with Claude Code
claude mcp add recurse --transport stdio -- python -m recurse.server
```

See [examples/setup_claude_code.md](examples/setup_claude_code.md) for full setup instructions.

## Tools

| Tool | Description |
|------|-------------|
| `recurse_query` | Answer questions over large context via RLM loop |
| `recurse_ingest` | Pre-index a codebase or document set |
| `recurse_status` | Check progress of a running query |
| `recurse_threads` | List, inspect, or delete stored threads |

## Example Usage

```
# In Claude Code:
"Ingest my project at /Users/me/myapp"
→ calls recurse_ingest(path="/Users/me/myapp", thread_id="myapp")

"Explain how authentication works in myapp"
→ calls recurse_query(query="Explain auth...", context_source="thread:myapp")
```

## Models

| Role | Model | Active Params | Notes |
|------|-------|---------------|-------|
| Root (orchestrator) | `qwen3.5:35b-a3b` | 3B | MoE — fast, excellent at code gen |
| Sub (analyst) | `qwen3.5:9b` | 9B | Beats 30B-class on comprehension |
| Power mode | `qwen3.5:27b` | 27B | Ties GPT-5 mini on SWE-bench |

All models run locally via Ollama. Also supports Qwen API (DashScope) and OpenRouter.

## Configuration

Copy `examples/config.example.yaml` to `~/.recurse/config.yaml`:

```yaml
models:
  root: qwen3.5:35b-a3b
  sub: qwen3.5:9b
  base_url: http://localhost:11434/v1

engine:
  max_iterations: 15

sandbox:
  mode: subprocess  # or docker
  timeout_seconds: 30
```

## Architecture

```
Claude Code → stdio → recurse/server.py (MCP)
                           ↓
                    recurse/engine/core.py (RLM loop)
                     ├── qwen.py (Qwen 3.5 via Ollama)
                     ├── sandbox.py (Python exec REPL)
                     └── prompts.py (system prompts)
                           ↓
                    recurse/store/
                     ├── context_store.py (file storage)
                     └── cache.py (sub-call caching)
```

## Testing

```bash
# Unit tests (no Ollama required)
pytest tests/ -v

# Integration smoke test (needs Ollama)
python -c "
import asyncio
from recurse.config import RecurseConfig
from recurse.engine.core import RecurseEngine

config = RecurseConfig()
engine = RecurseEngine(config)
result = asyncio.run(engine.query(
    query='What is the secret number?',
    context='... lots of text ... The secret number is 42 ... more text ...',
    thread_id='test'
))
print(result.answer)
assert '42' in result.answer
"
```
# Usage
To make it permanent for all projects, add this to your ~/.claude/CLAUDE.md:

## Large Analysis
When analyzing codebases with many files, use recurse_ingest first, then recurse_query instead of reading files individually.
This preserves context window for reasoning, not file storage.

What Recurse helps with:

  ┌─────────────────┬─────────────────┬───────────────────────────────────┐
  │    Scenario     │ Without Recurse │           With Recurse            │
  ├─────────────────┼─────────────────┼───────────────────────────────────┤
  │ "Explain auth   │ Claude reads    │ Claude calls recurse_query → Qwen │
  │ in my 50k-line  │ files → burns   │  reads everything locally →       │
  │ codebase"       │ your context    │ Claude only receives the answer   │
  │                 │ fast            │ (~500 tokens)                     │
  ├─────────────────┼─────────────────┼───────────────────────────────────┤
  │ Your Claude     │ Fills up with   │ Stays clean                       │
  │ Code context    │ file contents   │                                   │
  ├─────────────────┼─────────────────┼───────────────────────────────────┤
  │ Anthropic API   │ Large, used for │ Small (just the answer)           │
  │ tokens          │  analysis       │                                   │
  ├─────────────────┼─────────────────┼───────────────────────────────────┤
  │ Local compute   │ None            │ Qwen models run on your machine   │
  └─────────────────┴─────────────────┴───────────────────────────────────┘

  

## License

MIT
