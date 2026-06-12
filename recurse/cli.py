"""Recurse CLI.

Commands:
    recurse "<query>" [--path DIR] [--thread NAME] [--provider groq|openai|anthropic]
    recurse ingest DIR [--thread NAME]
    recurse threads [delete NAME]
    recurse history [--thread NAME] [--limit N]
    recurse init
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from recurse.config import PRESETS, load_config, write_default_config

console = Console()
err_console = Console(stderr=True, style="bold red")

SUBCOMMANDS = {"ingest", "threads", "history", "init"}

API_KEY_SETUP = """\
API keys are read from environment variables (never stored in the config file):

  groq       export GROQ_API_KEY=gsk_...       free tier — small contexts only (12K TPM cap)
             https://console.groq.com/keys
  openai     export OPENAI_API_KEY=sk-...      recommended (gpt-5-mini)
             https://platform.openai.com/api-keys
             NOTE: ChatGPT Pro does NOT include API credits — load ~$10 at
             platform.openai.com/billing before use.
  anthropic  export ANTHROPIC_API_KEY=sk-...   needs console.anthropic.com credits
             (separate from a Claude.ai subscription)
"""


def _engine(provider: str | None):
    # Imported lazily so `recurse init` works before any provider is usable
    from recurse.engine import RecurseEngine

    config = load_config()
    if provider:
        config.provider = provider
    return RecurseEngine(config)


def _format_usage(result) -> str:
    usage = result.usage_summary
    try:
        tokens = f"{usage.total_input_tokens:,} in / {usage.total_output_tokens:,} out tokens"
    except AttributeError:
        tokens = str(usage)
    return f"{result.execution_time:.1f}s | {tokens}"


def cmd_query(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="recurse", description="Ask a question over a large codebase/document set."
    )
    parser.add_argument("query", help="the question to answer")
    parser.add_argument("--path", help="directory to ingest into the thread before querying")
    parser.add_argument("--thread", default="default", help="thread name (default: default)")
    parser.add_argument("--provider", choices=sorted(PRESETS), help="override configured provider")
    args = parser.parse_args(argv)

    engine = _engine(args.provider)
    if args.path:
        source = f"path:{args.path}"
    elif engine.store.has_thread(args.thread):
        source = f"thread:{args.thread}"
    else:
        err_console.print(
            f"Thread '{args.thread}' has no context and no --path was given. "
            f"Run: recurse ingest DIR --thread {args.thread}"
        )
        return 1

    if engine.config.verbose:
        # The RLM loop prints its own rich iteration output; a spinner would clash.
        result = engine.query(args.query, source, args.thread)
    else:
        with console.status("[bold cyan]recursing..."):
            result = engine.query(args.query, source, args.thread)

    console.print(result.response)
    console.print(f"[dim]{_format_usage(result)} | thread: {args.thread}[/dim]")
    return 0


def cmd_ingest(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="recurse ingest", description="Index a directory into a thread.")
    parser.add_argument("path", help="directory to ingest")
    parser.add_argument("--thread", default="default", help="thread name (default: default)")
    args = parser.parse_args(argv)

    engine = _engine(None)
    res = engine.ingest(args.path, args.thread)
    console.print(
        f"Ingested [bold]{res.files_count}[/bold] files "
        f"(~{res.token_estimate:,} tokens, {res.skipped_count} skipped) "
        f"into thread '[bold]{res.thread_id}[/bold]'."
    )
    console.print(f"[dim]{res.file_tree}[/dim]")
    return 0


def cmd_threads(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="recurse threads", description="List or delete threads.")
    parser.add_argument("action", nargs="?", default="list", choices=["list", "delete"])
    parser.add_argument("name", nargs="?", help="thread name (required for delete)")
    args = parser.parse_args(argv)

    engine = _engine(None)
    if args.action == "delete":
        if not args.name:
            err_console.print("Usage: recurse threads delete NAME")
            return 1
        engine.store.delete_thread(args.name)
        console.print(f"Deleted thread '{args.name}'.")
        return 0

    threads = engine.store.list_threads()
    if not threads:
        console.print("No threads yet. Create one with: recurse ingest DIR --thread NAME")
        return 0
    for t in threads:
        console.print(
            f"[bold]{t['thread_id']}[/bold]  "
            f"[dim]{t['context_chars']:,} chars, {t['turns']} turns[/dim]"
        )
    return 0


def cmd_history(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="recurse history", description="Show a thread's Q&A history.")
    parser.add_argument("--thread", default="default")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    engine = _engine(None)
    turns = engine.store.get_turns(args.thread, limit=args.limit)
    if not turns:
        console.print(f"No history for thread '{args.thread}'.")
        return 0
    for turn in turns:
        console.print(f"[bold cyan]{turn['ts']}[/bold cyan]")
        console.print(f"  [bold]Q:[/bold] {turn['query']}")
        console.print(f"  [bold]A:[/bold] {turn['answer']}\n")
    return 0


def cmd_init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="recurse init", description="Write the default config file.")
    parser.parse_args(argv)

    path = write_default_config()
    console.print(f"Config: [bold]{path}[/bold]")
    console.print(API_KEY_SETUP)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv in (["-h"], ["--help"]):
        console.print(__doc__.strip())
        return 0

    try:
        if argv[0] in SUBCOMMANDS:
            handler = {
                "ingest": cmd_ingest,
                "threads": cmd_threads,
                "history": cmd_history,
                "init": cmd_init,
            }[argv[0]]
            return handler(argv[1:])
        return cmd_query(argv)
    except KeyboardInterrupt:
        err_console.print("Interrupted.")
        return 130
    except Exception as e:  # CLI boundary: show the message, not a traceback
        err_console.print(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
