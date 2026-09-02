"""Standalone CLI for running the compaction pipeline outside an MCP
client, useful for development and debugging."""

import argparse
import json
from pathlib import Path

from . import discovery, server


def _cmd_compact(args: argparse.Namespace) -> None:
    session_path = (
        Path(args.session_path) if args.session_path
        else discovery.most_recent_session(Path.cwd())
    )
    if session_path is None:
        print(json.dumps({"ok": False, "reason": "no_session_found"}))
        raise SystemExit(1)
    result = server._run_compaction(
        session_path,
        args.custom_instructions,
        args.context_budget,
        args.model,
        Path(args.output_path) if args.output_path else None,
        args.fallback_model,
    )
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


def _cmd_list(args: argparse.Namespace) -> None:
    cwd = Path(args.project_cwd) if args.project_cwd else Path.cwd()
    print(json.dumps(discovery.list_sessions(cwd), indent=2))


def _cmd_serve(_args: argparse.Namespace) -> None:
    server.main()


def main() -> None:
    parser = argparse.ArgumentParser(prog="cc-local-compact")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compact_parser = subparsers.add_parser("compact", help="Compact a session transcript")
    compact_parser.add_argument("session_path", nargs="?", default=None)
    compact_parser.add_argument("--instructions", dest="custom_instructions", default=None)
    compact_parser.add_argument("--context-budget", type=int, default=None)
    compact_parser.add_argument("--model", default=None)
    compact_parser.add_argument("--fallback-model", dest="fallback_model", default=None)
    compact_parser.add_argument("--output", dest="output_path", default=None)
    compact_parser.set_defaults(func=_cmd_compact)

    list_parser = subparsers.add_parser("list", help="List session transcripts for a project")
    list_parser.add_argument("project_cwd", nargs="?", default=None)
    list_parser.set_defaults(func=_cmd_list)

    serve_parser = subparsers.add_parser("serve", help="Run the MCP stdio server")
    serve_parser.set_defaults(func=_cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
