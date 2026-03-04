# DESIGN.md — Recurse: RLM-Powered Unlimited Context MCP Server

> This document is the source of truth for building this project. Claude Code should reference it throughout development.

---

## 1. What This Is

An MCP server that gives Claude Code unlimited context by using Recursive Language Models (RLMs) powered by Qwen 3.5. When Claude Code encounters a codebase, document set, or conversation history too large for its context window, it delegates to this tool. The RLM decomposes the input, recursively analyzes it using local Qwen 3.5 models, and returns a synthesized answer.

**One sentence:** `claude mcp add recurse` and Claude Code can now reason over million-token codebases using local Qwen 3.5 models.

---

## 2. How RLMs Work (Reference for Implementation)

Based on the paper: [arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)
Reference implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm) (MIT licensed)
Inspiration: [github.com/WingchunSiu/Monolith](https://github.com/WingchunSiu/Monolith)

The core idea:

1. User provides a **query** and a **context** (could be millions of tokens)
2. The context is **never fed into the LLM directly** — it's stored as a Python variable in a sandboxed REPL
3. A **root LLM** (Qwen 3.5) receives only the query + metadata about the context (length, type)
4. The root LLM writes Python code in the REPL to peek at, grep through, split, and analyze the context
5. Inside the REPL, the root LLM can call `llm_query(sub_query, sub_context)` to delegate focused analysis to a **sub-LLM** (smaller/cheaper Qwen 3.5)
6. After N iterations, the root LLM outputs `FINAL(answer)` or `FINAL_VAR(variable_name)`

The root LLM never sees the full context. It writes code to navigate it. Sub-LLMs see small focused chunks. This is why a 27B model can effectively reason over 10M+ tokens.

---

## 3. Qwen 3.5 Model Strategy

All Qwen 3.5 models: Apache 2.0, 256K native context, 201 languages, thinking mode (`<think>...</think>`), native tool calling, natively multimodal (text + image + video).

### Model Roles

| Role | Model | Active Params | Why |
|------|-------|---------------|-----|
| **Root LLM** (orchestrator) | `qwen3.5:35b-a3b` | 3B | MoE — only 3B active params despite 35B total. Runs on 8GB VRAM. Surpasses previous-gen 235B. Fast inference, smart enough to write good decomposition code. |
| **Sub-LLM** (focused analysis) | `qwen3.5:9b` | 9B | Beats previous-gen 30B. Strong at focused reading comprehension. Cheap and fast for hundreds of sub-calls per query. |
| **Power mode** (optional) | `qwen3.5:27b` | 27B | Dense model, all 27B active. Ties GPT-5 mini on SWE-bench. Use when root needs maximum reasoning quality. Needs 22GB+ RAM. |
| **Edge/fast mode** (optional) | `qwen3.5:4b` | 4B | For extremely resource-constrained environments or as a ultra-cheap sub-LLM. |

### Default Configuration

```yaml
models:
  root: qwen3.5:35b-a3b    # orchestrates decomposition
  sub: qwen3.5:9b           # handles focused sub-queries
```

### Why This Pairing Works

The RLM architecture means the root LLM's job is to **write good Python code** that decomposes context — it doesn't need to understand the full content itself. The 35B-A3B is extremely good at code generation (it surpasses the previous 235B on coding benchmarks) while only activating 3B parameters per token, making it fast. The sub-LLM needs to **read and comprehend** focused chunks of text. The 9B model beats models 3x its size on comprehension tasks and runs fast enough to handle dozens of sub-calls per query.

### Serving

All models served locally via **Ollama** using the OpenAI-compatible API at `http://localhost:11434/v1`. The `alexzhang13/rlm` library already supports OpenAI-compatible endpoints, so this requires minimal integration work.

```bash
# User setup (one time)
ollama pull qwen3.5:35b-a3b
ollama pull qwen3.5:9b
```

For users with more powerful hardware or who want API access, also support:
- **vLLM** — for GPU-optimized serving (`http://localhost:8000/v1`)
- **Qwen API** (DashScope) — Alibaba's hosted API, OpenAI-compatible
- **OpenRouter** — route to any Qwen 3.5 variant via API

The backend is configured in `~/.recurse/config.yaml` and the engine auto-detects which models are available.

### Thinking Mode

Qwen 3.5 defaults to thinking mode (`<think>...</think>` before response). For the **root LLM**, enable thinking — it helps the model plan its decomposition strategy. For the **sub-LLM**, disable thinking (`/no_think` or set in API) to save tokens and speed up sub-calls. The sub-LLM should just answer the focused question directly.

```python
# Root LLM call — thinking enabled (default)
root_response = client.chat.completions.create(
    model="qwen3.5:35b-a3b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.6,
    top_p=0.95,
    extra_body={"top_k": 20}
)

# Sub-LLM call — thinking disabled for speed
sub_response = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=[
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": sub_prompt}
    ],
    temperature=0.3  # lower temp for focused analysis
)
```

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Claude Code                        │
│               (or any MCP client)                     │
└──────────────────┬───────────────────────────────────┘
                   │ MCP (stdio)
                   ▼
┌──────────────────────────────────────────────────────┐
│                  recurse/server.py                     │
│                  MCP Server                            │
│                                                       │
│  Tools:                                               │
│    recurse_query    — ask questions over large context │
│    recurse_ingest   — index a codebase or doc set     │
│    recurse_status   — check progress of running query │
│    recurse_threads  — list/inspect persistent threads │
│                                                       │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│               recurse/engine/core.py                  │
│               RLM Engine                              │
│                                                       │
│  1. Load context from ContextStore                    │
│  2. Send query + context metadata to root LLM         │
│  3. Root LLM writes Python → execute in Sandbox       │
│  4. Sandbox calls llm_query() → routed to sub-LLM    │
│  5. Loop until FINAL() or max_iterations              │
│  6. Cache result, return answer                       │
│                                                       │
│  ┌────────────────────────────────────────────────┐   │
│  │          recurse/engine/sandbox.py             │   │
│  │          Sandboxed Python REPL                 │   │
│  │                                                │   │
│  │  Globals:                                      │   │
│  │    CONTEXT: str  — the full context as string  │   │
│  │    llm_query(query, context) → str             │   │
│  │    batch_llm_query(queries) → list[str]        │   │
│  │                                                │   │
│  │  Execution: Docker container (default)         │   │
│  │             or subprocess with restricted exec │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
│  ┌────────────────────────────────────────────────┐   │
│  │          recurse/engine/qwen.py                │   │
│  │          Qwen 3.5 Client                       │   │
│  │                                                │   │
│  │  Connects to Ollama / vLLM / DashScope API     │   │
│  │  Handles thinking mode toggle                  │   │
│  │  Manages root vs sub model routing             │   │
│  │  Token counting and budget tracking            │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
└──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│             recurse/store/                             │
│             Context Store                             │
│                                                       │
│  ~/.recurse/threads/{thread_id}/                      │
│    manifest.json       — file list + content hashes   │
│    files/              — individual file contents      │
│    conversations/      — past Q&A pairs               │
│    cache/              — cached sub-call results       │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## 5. MCP Tool Specifications

### 5.1 `recurse_query`

Primary tool. Claude Code calls this to reason over large context.

```python
@server.tool()
async def recurse_query(
    query: str,
    # Context source — one of:
    #   "thread:{thread_id}" — use stored context from a thread
    #   "path:/absolute/path" — read from filesystem (will auto-ingest)
    #   "inline:..." — context provided directly (for smaller inputs)
    context_source: str,
    thread_id: str = "default",
    max_iterations: int = 15,
    token_budget: int | None = None,  # max tokens to spend, None = unlimited
) -> dict:
    """
    Returns:
        answer: str — the final answer
        iterations_used: int
        sub_calls_made: int
        tokens_used: int
        cached_hits: int
        trajectory_summary: str — brief description of what the RLM did
    """
```

**Example Claude Code usage:**
```
User: "Explain how authentication works in this codebase"
Claude Code: [calls recurse_query(query="Explain how authentication works...", context_source="path:/users/me/project")]
```

### 5.2 `recurse_ingest`

Pre-indexes a codebase or document set into a thread. Makes subsequent queries faster.

```python
@server.tool()
async def recurse_ingest(
    path: str,             # directory or file to ingest
    thread_id: str = "default",
    include_patterns: list[str] | None = None,  # e.g. ["*.py", "*.ts"]
    exclude_patterns: list[str] | None = None,  # e.g. ["node_modules", ".git"]
) -> dict:
    """
    Returns:
        files_ingested: int
        total_size_bytes: int
        total_tokens_estimate: int
        file_tree: str — compact representation of the codebase structure
    """
```

### 5.3 `recurse_status`

For long-running queries — reports what the RLM is currently doing.

```python
@server.tool()
async def recurse_status(
    thread_id: str = "default",
) -> dict:
    """
    Returns:
        state: str — "idle" | "decomposing" | "analyzing" | "aggregating" | "complete"
        current_iteration: int
        max_iterations: int
        sub_calls_completed: int
        elapsed_seconds: float
        partial_findings: list[str]  # what the RLM has found so far
    """
```

### 5.4 `recurse_threads`

List and inspect persistent threads.

```python
@server.tool()
async def recurse_threads(
    action: str = "list",  # "list" | "inspect" | "delete"
    thread_id: str | None = None,
) -> dict:
    """
    list: returns all thread IDs with metadata
    inspect: returns details of a specific thread (file count, size, last query)
    delete: removes a thread and its stored context
    """
```

---

## 6. Core Engine Implementation

### 6.1 RLM Loop (`recurse/engine/core.py`)

This is the heart of the system. Reference `alexzhang13/rlm` for the loop structure.

```python
class RecurseEngine:
    def __init__(self, config: RecurseConfig):
        self.qwen = QwenClient(config.models)
        self.sandbox = Sandbox(config.sandbox_mode)
        self.store = ContextStore(config.storage_path)
        self.cache = ResultCache(config.cache_path)

    async def query(self, query: str, context: str, thread_id: str,
                    max_iterations: int = 15, token_budget: int | None = None) -> RLMResult:
        # 1. Load context into sandbox
        self.sandbox.set_variable("CONTEXT", context)
        self.sandbox.set_variable("CONTEXT_LENGTH", len(context))

        # 2. Register llm_query function in sandbox
        self.sandbox.register_function("llm_query", self._make_sub_query_fn(thread_id))
        self.sandbox.register_function("batch_llm_query", self._make_batch_query_fn(thread_id))

        # 3. Build system prompt for root LLM
        system_prompt = self._build_system_prompt(context_length=len(context))

        # 4. RLM loop
        conversation = [{"role": "user", "content": f"Query: {query}\n\nContext length: {len(context)} characters."}]
        trajectory = []

        for i in range(max_iterations):
            # Get root LLM response (with thinking enabled)
            response = await self.qwen.root_completion(system_prompt, conversation)

            # Check for FINAL() or FINAL_VAR()
            final = self._extract_final(response)
            if final is not None:
                return RLMResult(answer=final, iterations=i+1, trajectory=trajectory)

            # Extract code blocks and execute in sandbox
            code = self._extract_code(response)
            if code:
                output = await self.sandbox.execute(code)
                # Truncate output if too long
                output = self._truncate(output, max_chars=50000)
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": f"[Execution Output]\n{output}"})
                trajectory.append({"iteration": i, "code": code, "output": output[:500]})
            else:
                # No code — root LLM is thinking/planning, continue
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": "Continue. Write code to analyze the context, or output FINAL(answer) when ready."})

        # Max iterations hit — force final answer
        return await self._force_final(conversation, trajectory)

    def _make_sub_query_fn(self, thread_id: str):
        """Creates the llm_query() function available in the sandbox."""
        async def llm_query(query: str, context: str = "") -> str:
            # Check cache first
            cache_key = self.cache.key(query, context)
            cached = self.cache.get(cache_key)
            if cached:
                return cached

            # Call sub-LLM (thinking disabled for speed)
            result = await self.qwen.sub_completion(query, context)
            self.cache.set(cache_key, result)
            return result
        return llm_query
```

### 6.2 System Prompt for Root LLM (`recurse/engine/prompts.py`)

The root LLM needs a specific prompt that teaches it the RLM pattern. Based on the original paper's prompt structure:

```python
ROOT_SYSTEM_PROMPT = """You are an AI assistant with access to a Python REPL environment.
A potentially very large context has been loaded into the variable `CONTEXT` (a string).
The context has {context_length} characters.

Your task: answer the user's query by examining the context programmatically.

You have these tools in the REPL:
- `CONTEXT` — the full context string (DO NOT try to print it all)
- `CONTEXT_LENGTH` — length in characters
- `llm_query(query: str, context: str) -> str` — ask a sub-LLM to analyze a snippet
- `batch_llm_query(items: list[tuple[str, str]]) -> list[str]` — parallel sub-LLM calls

Strategy:
1. RECON: Check CONTEXT_LENGTH. Peek at CONTEXT[:2000] and CONTEXT[-2000:] to understand format.
2. DECOMPOSE: Write Python to split context into meaningful chunks (by file, section, paragraph).
3. FILTER: Use regex, keywords, or string operations to find relevant chunks.
4. ANALYZE: Call llm_query() on each relevant chunk with a focused sub-question.
5. SYNTHESIZE: Combine sub-answers into a final response.

Rules:
- Write code in ```python blocks. You'll see execution output.
- NEVER print the entire CONTEXT. Always slice or filter first.
- Use llm_query() for any reasoning over text. Don't try to reason about content yourself.
- Use batch_llm_query() when you have multiple independent sub-questions.
- When done, output FINAL(your answer here) or FINAL_VAR(variable_name) to return.
- If you can't find the answer, say so in FINAL() — don't hallucinate.
"""
```

### 6.3 Qwen Client (`recurse/engine/qwen.py`)

Wraps the OpenAI-compatible API to talk to Qwen 3.5 models.

```python
from openai import AsyncOpenAI

class QwenClient:
    def __init__(self, model_config: ModelConfig):
        self.root_model = model_config.root     # e.g. "qwen3.5:35b-a3b"
        self.sub_model = model_config.sub       # e.g. "qwen3.5:9b"
        self.client = AsyncOpenAI(
            base_url=model_config.base_url,     # e.g. "http://localhost:11434/v1"
            api_key=model_config.api_key or "ollama"  # Ollama doesn't need a real key
        )

    async def root_completion(self, system_prompt: str, messages: list[dict]) -> str:
        """Root LLM call. Thinking mode ON. Used for decomposition/orchestration."""
        response = await self.client.chat.completions.create(
            model=self.root_model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=16384,
            temperature=0.6,
            top_p=0.95,
        )
        return response.choices[0].message.content

    async def sub_completion(self, query: str, context: str) -> str:
        """Sub-LLM call. Thinking mode OFF. Used for focused chunk analysis."""
        messages = [
            {"role": "system", "content": "/no_think\nYou are a precise analyst. Answer the question based only on the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        response = await self.client.chat.completions.create(
            model=self.sub_model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
        )
        return response.choices[0].message.content
```

### 6.4 Sandbox (`recurse/engine/sandbox.py`)

Executes the root LLM's Python code safely.

Two modes:
- **Docker** (default, safe): Runs code in a `python:3.12-slim` container. Slower but isolated.
- **Subprocess** (fast, less safe): Runs code in a subprocess with restricted globals. For trusted contexts only.

```python
class Sandbox:
    def __init__(self, mode: str = "subprocess"):
        self.mode = mode
        self.globals = {}
        self.registered_functions = {}

    def set_variable(self, name: str, value):
        self.globals[name] = value

    def register_function(self, name: str, fn):
        self.registered_functions[name] = fn
        self.globals[name] = fn

    async def execute(self, code: str) -> str:
        """Execute Python code and capture stdout + return value."""
        if self.mode == "docker":
            return await self._execute_docker(code)
        else:
            return await self._execute_subprocess(code)

    async def _execute_subprocess(self, code: str) -> str:
        """Execute in a restricted subprocess."""
        import io, contextlib
        stdout = io.StringIO()
        local_vars = {}
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, self.globals.copy(), local_vars)
            output = stdout.getvalue()
            # Also capture any assigned variables for future iterations
            self.globals.update(local_vars)
            return output if output else "(no output)"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
```

### 6.5 Context Store (`recurse/store/context_store.py`)

Persistent, structured storage for thread context.

```python
class ContextStore:
    """
    Storage layout:
    ~/.recurse/threads/{thread_id}/
        manifest.json       — {"files": [{"path": ..., "hash": ..., "size": ...}], "total_tokens": ...}
        files/
            src__main.py    — file contents (path separators replaced with __)
            src__utils.py
        conversations/
            {timestamp}.json — {"query": ..., "answer": ..., "tokens": ...}
        cache/
            {hash}.json     — cached sub-call results
    """

    def ingest_directory(self, path: str, thread_id: str,
                         include: list[str] | None = None,
                         exclude: list[str] | None = None) -> IngestResult:
        """Walk directory, store each file, build manifest."""

    def load_context(self, thread_id: str) -> str:
        """Load all files as a single concatenated string with file path headers."""
        # Format:
        # === FILE: src/main.py ===
        # <contents>
        # === FILE: src/utils.py ===
        # <contents>

    def get_file(self, thread_id: str, file_path: str) -> str | None:
        """Load a single file's contents."""

    def get_manifest(self, thread_id: str) -> dict:
        """Return file list with metadata."""

    def save_conversation(self, thread_id: str, query: str, answer: str, metadata: dict):
        """Persist a Q&A pair."""
```

### 6.6 Result Cache (`recurse/store/cache.py`)

Avoids redundant sub-LLM calls across queries.

```python
class ResultCache:
    """
    Cache key = hash(query + content_hash)
    Cache value = sub-LLM response

    If the same question is asked about the same content, return cached result.
    """

    def key(self, query: str, context: str) -> str:
        content_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{query_hash}_{content_hash}"

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str): ...
    def clear(self, thread_id: str): ...
```

---

## 7. File Structure

```
recurse/
├── DESIGN.md                    # this file
├── README.md
├── LICENSE                      # MIT
├── pyproject.toml
├── recurse/
│   ├── __init__.py
│   ├── server.py                # MCP server entry point
│   ├── config.py                # config loading from ~/.recurse/config.yaml
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── core.py              # RecurseEngine — the RLM loop
│   │   ├── qwen.py              # QwenClient — Qwen 3.5 API wrapper
│   │   ├── sandbox.py           # Sandbox — safe Python execution
│   │   └── prompts.py           # system prompts for root and sub LLMs
│   ├── store/
│   │   ├── __init__.py
│   │   ├── context_store.py     # structured file/thread storage
│   │   └── cache.py             # sub-call result caching
│   └── tools/
│       ├── __init__.py
│       ├── query.py             # recurse_query implementation
│       ├── ingest.py            # recurse_ingest implementation
│       ├── status.py            # recurse_status implementation
│       └── threads.py           # recurse_threads implementation
├── tests/
│   ├── test_engine.py
│   ├── test_sandbox.py
│   ├── test_store.py
│   ├── test_mcp_tools.py
│   └── fixtures/
│       ├── sample_codebase/     # small test codebase
│       └── sample_context.txt   # long text for NIAH testing
└── examples/
    ├── setup_claude_code.md     # step-by-step install guide
    └── config.example.yaml
```

---

## 8. Configuration

```yaml
# ~/.recurse/config.yaml

models:
  root: qwen3.5:35b-a3b
  sub: qwen3.5:9b
  base_url: http://localhost:11434/v1   # Ollama default
  api_key: ollama                        # Ollama doesn't validate keys

  # Alternative: Qwen API (DashScope)
  # root: qwen3.5-plus
  # sub: qwen3.5-flash
  # base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  # api_key: ${DASHSCOPE_API_KEY}

  # Alternative: OpenRouter
  # root: qwen/qwen3.5-35b-a3b
  # sub: qwen/qwen3.5-9b
  # base_url: https://openrouter.ai/api/v1
  # api_key: ${OPENROUTER_API_KEY}

engine:
  max_iterations: 15
  max_output_truncation: 50000       # chars — truncate sandbox output beyond this
  thinking_mode_root: true           # enable <think> for root LLM
  thinking_mode_sub: false           # disable <think> for sub-LLM

sandbox:
  mode: subprocess                   # subprocess | docker
  timeout_seconds: 30                # per code execution

storage:
  path: ~/.recurse/threads
  max_cache_size_mb: 500

ingest:
  default_exclude:
    - node_modules
    - .git
    - __pycache__
    - .venv
    - "*.pyc"
    - "*.lock"
    - dist
    - build
  max_file_size_kb: 500              # skip files larger than this
  max_total_files: 5000
```

---

## 9. Implementation Order

Build in this exact order. Each phase is independently testable.

### Phase 1: Qwen Client + Basic RLM Loop

**Files:** `recurse/engine/qwen.py`, `recurse/engine/sandbox.py`, `recurse/engine/prompts.py`, `recurse/engine/core.py`, `recurse/config.py`

**Goal:** A Python script that runs an RLM query against a local Qwen 3.5 model.

**Test:**
```python
engine = RecurseEngine(config)
result = await engine.query(
    query="What is the secret number?",
    context="... 500K chars of random text with 'The secret number is 42' hidden inside ...",
    thread_id="test"
)
assert "42" in result.answer
```

**Depends on:** User has Ollama running with `qwen3.5:35b-a3b` and `qwen3.5:9b` pulled.

### Phase 2: Context Store

**Files:** `recurse/store/context_store.py`, `recurse/store/cache.py`

**Goal:** Ingest a directory into structured storage. Load it back as concatenated context.

**Test:**
```python
store = ContextStore("~/.recurse/threads")
result = store.ingest_directory("./my-project", thread_id="proj1")
context = store.load_context("proj1")
assert "=== FILE:" in context
```

### Phase 3: MCP Server + Tools

**Files:** `recurse/server.py`, `recurse/tools/query.py`, `recurse/tools/ingest.py`, `recurse/tools/status.py`, `recurse/tools/threads.py`

**Goal:** A working MCP server that Claude Code can connect to.

**Test:**
```bash
# Add to Claude Code
claude mcp add recurse --transport stdio -- python -m recurse.server

# In Claude Code, the tools are now available
# Claude Code can call recurse_ingest to index a project
# Then call recurse_query to ask questions about it
```

### Phase 4: Caching + Performance

**Goal:** Sub-call caching, async batch queries, progress reporting.

**Test:** Run the same query twice. Second run should be significantly faster with cache hits reported.

### Phase 5: Polish

**Goal:** README, error handling, edge cases, example configs, Claude Code setup guide.

---

## 10. Dependencies

```toml
[project]
name = "recurse-rlm"
version = "0.1.0"
requires-python = ">=3.11"
license = "MIT"

dependencies = [
    "openai>=2.14.0",          # async client for Qwen (OpenAI-compatible API)
    "mcp>=1.0.0",              # MCP server SDK
    "pydantic>=2.0.0",         # config validation
    "pyyaml>=6.0",             # config file parsing
    "rich>=13.0.0",            # terminal output (logging, progress)
]

[project.optional-dependencies]
docker = ["docker>=7.0.0"]     # for Docker sandbox mode
```

---

## 11. Key Decisions Log

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Model framework | Qwen 3.5 via Ollama | OpenAI-compatible API, runs locally, Apache 2.0, trivial setup |
| Root model | 35B-A3B | Only 3B active params — fast + cheap. Strong at code gen (RLM needs this) |
| Sub model | 9B | Beats 30B-class models. Fast enough for many sub-calls per query |
| MCP transport | stdio | Simplest for Claude Code local integration. HTTP can be added later |
| Sandbox default | subprocess | Faster than Docker for dev. Docker available as config option |
| Context format | Concatenated with file headers | Simple, the RLM can parse headers with regex. No complex indexing needed initially |
| Cache strategy | Content-hash + query-hash | Deterministic. Same question + same content = same answer |
| License | MIT | Maximum adoption. Matches alexzhang13/rlm |
| Async | Yes (asyncio) | MCP server is async. Qwen client uses AsyncOpenAI. Enables future batch parallelism |

---

## 12. Improvements Over Monolith

| Monolith | Recurse |
|----------|---------|
| GPT-5 + GPT-5-nano (OpenAI only) | Qwen 3.5 local (any OpenAI-compatible endpoint) |
| Requires Modal + Cloudflare | Runs entirely on localhost |
| Flat `context.txt` per thread | Structured per-file storage with manifest |
| No caching | Content-hash sub-call cache |
| Blocking sequential sub-calls | Async + `batch_llm_query()` ready |
| No progress feedback | `recurse_status` tool for long queries |
| `exec()` in host process | Subprocess isolation (Docker optional) |
| GPL-3.0 | MIT |
| No thinking mode management | Root thinks, sub doesn't (configurable) |
| No ingest/indexing tool | `recurse_ingest` pre-processes codebases |

---

## 13. Future Extensions (Not in v1)

- **Docker sandbox by default** — once the subprocess path is stable
- **Modal/cloud compute** — for users without GPU hardware
- **Multimodal context** — Qwen 3.5 handles images/video natively; could analyze screenshots, diagrams
- **Fine-tuned decomposition** — train a LoRA on Qwen 3.5 specifically for RLM-style REPL code generation
- **Prompt injection detection** — inspect each sub-call for adversarial content before execution
- **MCP resources** — expose thread trajectories as `recurse://threads/{id}/trajectory`
- **Session auto-upload** — Claude Code Stop hook to auto-capture transcripts (like Monolith)
- **Streaming partial answers** — send intermediate findings back to Claude Code as they're discovered