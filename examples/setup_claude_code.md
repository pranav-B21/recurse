# Setting Up Recurse with Claude Code

This guide walks you through installing and configuring the Recurse MCP server so Claude Code can reason over large codebases and documents using local Qwen 3.5 models.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Claude Code (`claude` CLI)

## Step 1: Install Ollama Models

```bash
# Root LLM: orchestrates decomposition (only 3B active params, very fast)
ollama pull qwen3.5:35b-a3b

# Sub LLM: handles focused chunk analysis
ollama pull qwen3.5:9b
```

Verify they're available:

```bash
ollama list
```

## Step 2: Install Recurse

```bash
# Clone the repo
git clone https://github.com/yourname/recurse.git
cd recurse

# Install (editable mode for development)
pip install -e .

# Or install with Docker sandbox support
pip install -e ".[docker]"
```

## Step 3: Configure (Optional)

Copy the example config:

```bash
mkdir -p ~/.recurse
cp examples/config.example.yaml ~/.recurse/config.yaml
```

Edit `~/.recurse/config.yaml` to customize models, sandbox mode, or storage path.

## Step 4: Register with Claude Code

```bash
claude mcp add recurse --transport stdio -- python -m recurse.server
```

Verify registration:

```bash
claude mcp list
```

You should see `recurse` listed.

## Step 5: Use from Claude Code

In any Claude Code conversation, you now have access to four tools:

### `recurse_ingest` — Index a codebase

```
Please ingest my project at /Users/me/myproject into thread "myproject"
```

Claude Code will call:
```json
{
  "tool": "recurse_ingest",
  "path": "/Users/me/myproject",
  "thread_id": "myproject"
}
```

### `recurse_query` — Ask questions

```
Using the RLM, explain how authentication works in myproject
```

Claude Code will call:
```json
{
  "tool": "recurse_query",
  "query": "Explain how authentication works",
  "context_source": "thread:myproject",
  "thread_id": "myproject"
}
```

### `recurse_status` — Check progress

For long-running queries:
```
What's the status of the myproject query?
```

### `recurse_threads` — Manage threads

```
List all my recurse threads
```

## Troubleshooting

### "Connection refused" errors
Make sure Ollama is running:
```bash
ollama serve
```

### Models not found
Check available models:
```bash
ollama list
```
Pull any missing models per Step 1.

### Slow performance
- Switch root model to `qwen3.5:9b` for faster inference (less accurate decomposition)
- Reduce `max_iterations` in config
- Enable Docker mode only if security isolation is needed

### Testing without Ollama

Run the unit tests (they use mocks):
```bash
pytest tests/ -v
```

## Alternative Backends

### Qwen API (DashScope)

```yaml
models:
  root: qwen3.5-plus
  sub: qwen3.5-flash
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: ${DASHSCOPE_API_KEY}
```

### OpenRouter

```yaml
models:
  root: qwen/qwen3.5-35b-a3b
  sub: qwen/qwen3.5-9b
  base_url: https://openrouter.ai/api/v1
  api_key: ${OPENROUTER_API_KEY}
```
