"""MCP server entry point for Recurse.

Register via Claude Code:
    claude mcp add recurse --transport stdio -- python -m recurse.server
"""

from __future__ import annotations

from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.models import InitializationOptions

from recurse.config import RecurseConfig
from recurse.engine.core import RecurseEngine
from recurse.tools.ingest import run_ingest
from recurse.tools.query import run_query
from recurse.tools.status import run_status
from recurse.tools.threads import run_threads

# Single instances — shared across all tool calls
config = RecurseConfig.load()
engine = RecurseEngine(config)

server = Server("recurse")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="recurse_query",
            description=(
                "Ask questions over a very large context (codebase, documents, conversation history) "
                "using local Qwen 3.5 models via the RLM approach. The context is never fed into the LLM "
                "directly — the model writes Python code to inspect and decompose it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to answer about the context.",
                    },
                    "context_source": {
                        "type": "string",
                        "description": (
                            "Where to load context from. One of:\n"
                            "  'thread:{thread_id}' — use stored thread context\n"
                            "  'path:/absolute/path' — read from filesystem (auto-ingested)\n"
                            "  'inline:...' — context provided directly in this string"
                        ),
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID for tracking and caching. Default: 'default'.",
                        "default": "default",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum RLM loop iterations. Default: 15.",
                        "default": 15,
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Max tokens to spend. Omit for unlimited.",
                    },
                },
                "required": ["query", "context_source"],
            },
        ),
        types.Tool(
            name="recurse_ingest",
            description=(
                "Pre-index a codebase or document set into a named thread. "
                "Makes subsequent recurse_query calls faster."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to a directory or file to ingest.",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID to store this context under. Default: 'default'.",
                        "default": "default",
                    },
                    "include_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to include (e.g. ['*.py', '*.ts']). Omit for all.",
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional glob patterns to exclude beyond defaults.",
                    },
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="recurse_status",
            description="Check the progress of a running or completed RLM query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID to check. Default: 'default'.",
                        "default": "default",
                    },
                },
            },
        ),
        types.Tool(
            name="recurse_threads",
            description="List, inspect, or delete stored threads.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "inspect", "delete"],
                        "description": "Action to perform. Default: 'list'.",
                        "default": "list",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID (required for 'inspect' and 'delete').",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    args = arguments or {}
    try:
        if name == "recurse_query":
            result = await run_query(
                engine=engine,
                config=config,
                query=args["query"],
                context_source=args["context_source"],
                thread_id=args.get("thread_id", "default"),
                max_iterations=args.get("max_iterations", config.engine.max_iterations),
                token_budget=args.get("token_budget"),
            )
        elif name == "recurse_ingest":
            result = await run_ingest(
                config=config,
                path=args["path"],
                thread_id=args.get("thread_id", "default"),
                include_patterns=args.get("include_patterns"),
                exclude_patterns=args.get("exclude_patterns"),
            )
        elif name == "recurse_status":
            result = await run_status(
                engine=engine,
                thread_id=args.get("thread_id", "default"),
            )
        elif name == "recurse_threads":
            result = await run_threads(
                config=config,
                action=args.get("action", "list"),
                thread_id=args.get("thread_id"),
            )
        else:
            raise ValueError(f"Unknown tool: {name}")

        import json
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as exc:
        import traceback
        error_text = f"Error in {name}: {exc}\n{traceback.format_exc()}"
        return [types.TextContent(type="text", text=error_text)]


def main() -> None:
    import asyncio

    async def _run() -> None:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="recurse",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
