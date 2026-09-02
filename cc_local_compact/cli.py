"""Standalone CLI for running the compaction pipeline outside an MCP
client, useful for development and debugging."""

import argparse
import json
from pathlib import Path

from . import discovery, server


def _cmd_compact(args: argparse.Namespace) -> None:
    resolved_cwd = discovery.resolve_cwd()
    session_path, resolution_meta = discovery.resolve_session(args.session_path, resolved_cwd)
    if session_path is None:
        print(json.dumps({"ok": False, "reason": "no_session_found", "detail": f"no session found for project directory {resolved_cwd}"}))
        raise SystemExit(1)
    result = server._run_compaction(
        session_path,
        args.custom_instructions,
        args.context_budget,
        args.model,
        Path(args.output_path) if args.output_path else None,
        args.fallback_model,
        args.append_to_jsonl,
        resolution_meta,
    )
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


def _cmd_list(args: argparse.Namespace) -> None:
    cwd = Path(args.project_cwd) if args.project_cwd else discovery.resolve_cwd()
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
    compact_parser.add_argument(
        "--append-jsonl", dest="append_to_jsonl", action="store_true", default=False,
        help=(
            "Also append a compact_boundary sequence to the session's own JSONL, "
            "matching the real /compact's on-disk shape. Record-consistency only -- "
            "does not reduce cost on the session's next resumed turn. Avoid on a "
            "session the real Claude Code client currently has open."
        ),
    )
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
