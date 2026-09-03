"""Installs/removes the bare `/remind` slash command and its two matching
hooks into the user's own Claude Code config (not a plugin -- see
cc_local_compact/README.md, "Recovering after a manual /clear", for why a
plugin-packaged command can't be invoked bare).

Two hooks, both required:
- UserPromptExpansion, matched to "remind": runs remind-hook, the actual
  recovery logic (see cli.py/session_track.py).
- Stop (unmatched -- fires after every completed turn): runs
  track-session, which records "this window's current session" so
  remind-hook can find the exact predecessor session across a /clear
  without guessing (see session_track.py's module docstring for why this
  exists).

Both are registered using the `args` form (`command` = the executable's
own path, `args` = the subcommand) rather than a single shell command
string -- confirmed live that this spawns the hook as a direct child of
the real `claude` process (no intermediate shell), which is what lets
os.getppid() inside the hook reliably resolve that process's own stable
PID. A plain shell `command` string does NOT give this guarantee.

`register`/`unregister` write directly under `claude_home` (defaulting to
~/.claude), merging into settings.json rather than overwriting it --
other hooks (e.g. this same user's claude-net PreCompact/PostCompact
hooks) must survive untouched.
"""

import json
import shutil
import sys
from pathlib import Path

COMMAND_NAME = "remind"

COMMAND_FILE_CONTENT = """---
description: Recover a resumable summary of what happened before your last /clear
---
No recovery hook ran for this command. If you were expecting a summary of
your pre-/clear conversation, check `cc-local-compact register` was run and
hooks are enabled; otherwise just continue.
"""

HOOK_SPECS = [
    {
        "event": "UserPromptExpansion", "matcher": COMMAND_NAME, "args": ["remind-hook"],
        # NOTE: confirmed in the binary that statusMessage's live-spinner
        # display is hard-gated to Stop/SubagentStop hook events only (a
        # dedicated filter function checks hookEvent against exactly those
        # two) -- it's inert for UserPromptExpansion, kept only in case
        # Claude Code adds support for it later.
        "statusMessage": (
            "Recovering context from before your last /clear (runs a real "
            "compaction pass against the local model -- can take several "
            "minutes on a large session)..."
        ),
        # Real compaction runs on a large session have been benchmarked at
        # up to ~16 minutes (see cc_local_compact/README.md); the default
        # hook timeout is 600s (10 min). Confirmed live: a real /remind run
        # was killed by hitting that default, silently falling back to the
        # command's own markdown body as if the hook had never run at all.
        # 3600s gives real headroom even under backend contention.
        "timeout": 3600,
    },
    {"event": "Stop", "matcher": None, "args": ["track-session"]},
]


def default_claude_home() -> Path:
    return Path.home() / ".claude"


def resolve_executable() -> str:
    """Best-effort absolute path to this installation's own console
    script, embedded in the hook command so it works regardless of the
    hook subprocess's own PATH -- matches this project's existing
    .mcp.json convention of an absolute interpreter path rather than a
    bare name resolved via PATH at hook-execution time."""
    resolved = Path(sys.argv[0]).resolve()
    if resolved.is_file():
        return str(resolved)
    which = shutil.which("cc-local-compact")
    if which:
        return which
    raise RuntimeError(
        "could not resolve cc-local-compact's own installed path -- "
        "run this via the installed console script, not e.g. `python cli.py`"
    )


def _find_matcher_entry(entries: list[dict], matcher: str | None) -> dict | None:
    for entry in entries:
        if entry.get("matcher") == matcher or (matcher is None and "matcher" not in entry):
            return entry
    return None


def register(claude_home: Path | None = None, executable: str | None = None) -> dict:
    home = claude_home or default_claude_home()
    executable = executable or resolve_executable()

    commands_dir = home / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    command_path = commands_dir / f"{COMMAND_NAME}.md"
    command_path.write_text(COMMAND_FILE_CONTENT)

    settings_path = home / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    hooks_root = settings.setdefault("hooks", {})

    hooks_already_present = {}
    for spec in HOOK_SPECS:
        entries = hooks_root.setdefault(spec["event"], [])
        matcher_entry = _find_matcher_entry(entries, spec["matcher"])
        if matcher_entry is None:
            matcher_entry = {"hooks": []}
            if spec["matcher"] is not None:
                matcher_entry["matcher"] = spec["matcher"]
            entries.append(matcher_entry)
        existing_hook = next(
            (h for h in matcher_entry["hooks"] if h.get("command") == executable and h.get("args") == spec["args"]),
            None,
        )
        optional_fields = ("statusMessage", "timeout")
        if existing_hook is None:
            new_hook = {"type": "command", "command": executable, "args": spec["args"]}
            for field in optional_fields:
                if field in spec:
                    new_hook[field] = spec[field]
            matcher_entry["hooks"].append(new_hook)
        else:
            # re-registering after an upgrade -- refresh any changed
            # optional field on an already-installed hook rather than
            # leaving it stale.
            for field in optional_fields:
                if field in spec and existing_hook.get(field) != spec[field]:
                    existing_hook[field] = spec[field]
        hooks_already_present[spec["event"]] = existing_hook is not None

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    return {
        "ok": True,
        "command_file": str(command_path),
        "settings_file": str(settings_path),
        "executable": executable,
        "hooks_already_present": hooks_already_present,
    }


def unregister(claude_home: Path | None = None, executable: str | None = None) -> dict:
    home = claude_home or default_claude_home()
    executable = executable or resolve_executable()

    command_path = home / "commands" / f"{COMMAND_NAME}.md"
    command_removed = command_path.is_file()
    if command_removed:
        command_path.unlink()

    settings_path = home / "settings.json"
    hooks_removed = {}
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text())
        hooks_root = settings.get("hooks", {})
        for spec in HOOK_SPECS:
            entries = hooks_root.get(spec["event"], [])
            matcher_entry = _find_matcher_entry(entries, spec["matcher"])
            removed = False
            if matcher_entry is not None:
                before = len(matcher_entry.get("hooks", []))
                matcher_entry["hooks"] = [
                    h for h in matcher_entry.get("hooks", [])
                    if not (h.get("command") == executable and h.get("args") == spec["args"])
                ]
                removed = len(matcher_entry["hooks"]) != before
            hooks_removed[spec["event"]] = removed
            entries[:] = [e for e in entries if e.get("hooks")]
            if not entries and spec["event"] in hooks_root:
                del hooks_root[spec["event"]]
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    else:
        for spec in HOOK_SPECS:
            hooks_removed[spec["event"]] = False

    return {
        "ok": True,
        "command_file_removed": command_removed,
        "hooks_removed": hooks_removed,
    }
