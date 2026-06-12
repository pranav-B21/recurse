# Using Recurse from Claude Code (MCP)

## 1. Install

```bash
cd /path/to/recurse
python -m venv recurse/.venv && source recurse/.venv/bin/activate
pip install -e vendor/rlm    # the vendored FORK — never `pip install rlms`
pip install -e .
recurse init
export GROQ_API_KEY=gsk_...  # or OPENAI_API_KEY / ANTHROPIC_API_KEY
```

## 2. Register the MCP server

Use the project venv's python explicitly (absolute path):

```bash
claude mcp add recurse --transport stdio -- \
    /ABS/PATH/TO/recurse/recurse/.venv/bin/python -m recurse.server
```

## 3. Use it

In a Claude Code session:

> use recurse to ingest /path/to/big-project and then ask it how the auth flow works

Claude Code will call `recurse_ingest` then `recurse_query` and relay the
answer. Each thread keeps memory across sessions — follow-up questions on the
same `thread_id` see prior Q&A turns.

## Tools

| Tool | Args | Purpose |
|---|---|---|
| `recurse_query` | `query`, `context_source`, `thread_id` | Answer over `path:/abs/dir` \| `thread:<id>` \| `inline:<text>` |
| `recurse_ingest` | `path`, `thread_id` | Index a directory into a persistent thread |
| `recurse_threads` | `action` (list/history/delete), `thread_id` | Manage threads |
