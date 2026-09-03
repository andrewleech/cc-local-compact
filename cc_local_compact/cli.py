"""Standalone CLI for running the compaction pipeline outside an MCP
client, and for installing/running the /remind command + hook (see
register.py and cc_local_compact/README.md, "Recovering after a manual
/clear")."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from . import discovery, register, response, session_track, transcript

# `server` (and the anthropic/fastmcp/authlib stack it pulls in) is
# imported lazily, only by the subcommands that actually run a
# compaction (compact, serve, and remind-hook once it's past its
# fast-fail checks) -- register/unregister/list/remind-hook's common
# "nothing to recover" case have no need for it, and that stack is slow
# to import and noisy (fastmcp pulls in authlib, which currently emits
# AuthlibDeprecationWarning on import -- not this project's bug to fix,
# but no reason to pay for it or show it on every /remind invocation
# that doesn't even reach a real compaction call).


def _import_server():
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"authlib(\..*)?")
        from . import server
    return server


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
        lines_override, error = discovery.slice_since_last_clear(all_lines)
        if error is not None:
            print(json.dumps({
                "ok": False, "reason": error,
                "detail": "no /clear command found on this session's main thread" if error == "no_clear_boundary_found"
                else "nothing to summarize before the session's last /clear (since the previous /clear, or session start)",
            }))
            raise SystemExit(1)
        trigger = "remind"

    server = _import_server()
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
    _import_server().main()


def _remind_hook_text(payload: dict, pid: int) -> str:
    """Body of the /remind UserPromptExpansion hook: never raises -- any
    failure degrades to an explanatory string, since the hook's own
    contract (see _cmd_remind_hook) always emits valid JSON regardless of
    what happens here.

    `pid` identifies this window's own claude process (os.getppid() from
    a hook registered with the `args` form -- see session_track.py's
    module docstring for why that's a direct, reliable parent, not a
    shell-wrapper indirection). Used to look up the predecessor session
    session_track.py's Stop hook recorded for this exact process, rather
    than any file-boundary or timing heuristic -- immune to "more than
    one session open on this project" ambiguity, since PID is per-window,
    not per-project."""
    try:
        my_session_id = payload.get("session_id")
        predecessor = session_track.predecessor_session(pid, my_session_id) if my_session_id else None
        if predecessor is None:
            return (
                "/remind: no predecessor session tracked for this window -- "
                "either nothing ran here before the last /clear, or the Stop "
                "hook (see `cc-local-compact register`) hasn't recorded a turn yet."
            )

        session_path = Path(predecessor["transcript_path"])
        if not session_path.is_file():
            return f"/remind: predecessor session file no longer exists ({session_path})."

        server = _import_server()
        result = server._run_compaction(
            session_path, None, None, None, None, None, False, None, trigger="remind",
        )
        if not result.get("ok"):
            return f"/remind: couldn't recover a summary -- {result.get('detail') or result.get('reason')}"
        return response.build_resume_preamble(result["summary"], transcript_path=str(session_path))
    except Exception as error:
        return f"/remind: recovery failed unexpectedly -- {error}"


def _cmd_remind_hook(_args: argparse.Namespace) -> None:
    """UserPromptExpansion hook body for /remind (installed by `register`).
    Reads the hook's JSON input from stdin, always prints exactly one
    hookSpecificOutput JSON line and exits 0 -- a malformed/failed
    recovery degrades to an explanatory additionalContext, never a crash
    or non-JSON output reaching Claude Code."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    text = _remind_hook_text(payload, os.getppid())
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptExpansion",
            "additionalContext": text,
            "suppressOriginalPrompt": True,
        },
    }))


def _cmd_track_session(_args: argparse.Namespace) -> None:
    """Stop hook body (installed by `register`): records "this process's
    current session" after every completed turn, so /remind can later
    find the exact predecessor session across a /clear -- see
    session_track.py's module docstring for why this exists and why PID
    is the correlation key. Never raises: a tracking failure should never
    surface as a visible error on an ordinary turn."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        session_id = payload.get("session_id")
        transcript_path = payload.get("transcript_path")
        cwd = payload.get("cwd")
        if session_id and transcript_path:
            session_track.record_turn(os.getppid(), session_id, transcript_path, cwd or "")
    except Exception:
        pass
    print("{}")


def _cmd_register(args: argparse.Namespace) -> None:
    result = register.register(Path(args.claude_home) if args.claude_home else None)
    print(json.dumps(result, indent=2))


def _cmd_unregister(args: argparse.Namespace) -> None:
    result = register.unregister(Path(args.claude_home) if args.claude_home else None)
    print(json.dumps(result, indent=2))


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
            "start) instead of the whole transcript. NOTE: this detects an "
            "in-place /clear marker within a single transcript file -- the "
            "live /remind hook no longer works this way (Claude Code's /clear "
            "now rotates to a separate session file with no in-file boundary "
            "to find; /remind instead uses session_track.py's per-process "
            "predecessor tracking). This flag is kept for dry-running against "
            "older-style transcripts where /clear was recorded in place. Fails "
            "with reason 'no_clear_boundary_found' if the session has no "
            "in-place /clear marker, or 'empty_pre_clear_span' if there's "
            "nothing in that span."
        ),
    )
    compact_parser.set_defaults(func=_cmd_compact)

    list_parser = subparsers.add_parser("list", help="List session transcripts for a project")
    list_parser.add_argument("project_cwd", nargs="?", default=None)
    list_parser.set_defaults(func=_cmd_list)

    serve_parser = subparsers.add_parser("serve", help="Run the MCP stdio server")
    serve_parser.set_defaults(func=_cmd_serve)

    remind_hook_parser = subparsers.add_parser(
        "remind-hook", help="UserPromptExpansion hook body for /remind (installed by `register`, not run directly)",
    )
    remind_hook_parser.set_defaults(func=_cmd_remind_hook)

    track_session_parser = subparsers.add_parser(
        "track-session", help="Stop hook body that records this window's current session (installed by `register`, not run directly)",
    )
    track_session_parser.set_defaults(func=_cmd_track_session)

    register_parser = subparsers.add_parser(
        "register", help="Install the bare /remind command + its UserPromptExpansion hook into ~/.claude",
    )
    register_parser.add_argument("--claude-home", default=None, help="Override ~/.claude (for testing)")
    register_parser.set_defaults(func=_cmd_register)

    unregister_parser = subparsers.add_parser(
        "unregister", help="Remove the /remind command + hook installed by `register`",
    )
    unregister_parser.add_argument("--claude-home", default=None, help="Override ~/.claude (for testing)")
    unregister_parser.set_defaults(func=_cmd_unregister)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
