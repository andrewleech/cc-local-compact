"""Standalone CLI for running the compaction pipeline outside an MCP
client, useful for development and debugging."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import discovery, server, transcript


def _prompt_for_session(candidates: list[dict]) -> Path | None:
    """Interactively pick one of several ambiguous candidates. Prints the
    picker to stderr (stdout is reserved for the final JSON result) and
    reads a choice from stdin. Returns None -- caller should exit(1) -- if
    stdin isn't a terminal (nothing to interact with) or the user cancels."""
    print(f"{len(candidates)} sessions exist for this project -- pick one:", file=sys.stderr)
    for i, candidate in enumerate(candidates, 1):
        when = datetime.fromtimestamp(candidate["mtime"]).strftime("%Y-%m-%d %H:%M")
        print(f"  [{i}] {when}  {candidate['display_name']}", file=sys.stderr)
        print(f"      {candidate['path']}", file=sys.stderr)
    if not sys.stdin.isatty():
        print(
            "stdin is not a terminal, so this can't prompt for a choice -- "
            "re-run with session_path set to one of the paths above.",
            file=sys.stderr,
        )
        return None
    while True:
        choice = input(f"Select session [1-{len(candidates)}] (blank to cancel): ").strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return Path(candidates[int(choice) - 1]["path"])
        print("Invalid selection, try again.", file=sys.stderr)


def _cmd_compact(args: argparse.Namespace) -> None:
    resolved_cwd = discovery.resolve_cwd()
    session_path, resolution_meta = discovery.resolve_session(args.session_path, resolved_cwd)
    if session_path is None:
        if resolution_meta.get("source") == "ambiguous":
            session_path = _prompt_for_session(resolution_meta["candidates"])
            if session_path is None:
                raise SystemExit(1)
            resolution_meta = {"source": "user_selected", "candidate_count": resolution_meta["candidate_count"]}
        else:
            print(json.dumps({"ok": False, "reason": "no_session_found", "detail": f"no session found for project directory {resolved_cwd}"}))
            raise SystemExit(1)

    lines_override = None
    trigger = "manual"
    if args.before_last_clear:
        all_lines = transcript.load_transcript(session_path)
        clear_indices = discovery.find_clear_indices(all_lines)
        if not clear_indices:
            print(json.dumps({
                "ok": False, "reason": "no_clear_boundary_found",
                "detail": "no /clear command found on this session's main thread",
            }))
            raise SystemExit(1)
        span_start = clear_indices[-2] + 1 if len(clear_indices) > 1 else 0
        lines_override = all_lines[span_start:clear_indices[-1]]
        if not lines_override:
            print(json.dumps({
                "ok": False, "reason": "empty_pre_clear_span",
                "detail": "nothing to summarize before the session's last /clear (since the previous /clear, or session start)",
            }))
            raise SystemExit(1)
        trigger = "continue_after_clear"

    result = server._run_compaction(
        session_path,
        args.custom_instructions,
        args.context_budget,
        args.model,
        Path(args.output_path) if args.output_path else None,
        args.fallback_model,
        args.append_to_jsonl,
        resolution_meta,
        lines=lines_override,
        trigger=trigger,
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
    compact_parser.add_argument(
        "--before-last-clear", dest="before_last_clear", action="store_true", default=False,
        help=(
            "Summarize only the span before the session's last /clear command "
            "(since the previous /clear if there was one, else since session "
            "start) instead of the whole transcript -- exercises the same "
            "slicing continue_after_clear (MCP tool) uses, for dry-running "
            "outside a live session. Fails with reason 'no_clear_boundary_found' "
            "if the session has no /clear, or 'empty_pre_clear_span' if there's "
            "nothing in that span. Does not apply continue_after_clear's "
            "resume-framed summary wrapping -- there's no live agent here to "
            "read it."
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
