"""Installs/removes the bare `/remind` slash command and its matching
UserPromptExpansion hook into the user's own Claude Code config (not a
plugin -- see cc_local_compact/README.md, "Recovering after a manual
/clear", for why a plugin-packaged command can't be invoked bare).

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


def _hook_command(executable: str) -> str:
    return f"{executable} remind-hook"


def register(claude_home: Path | None = None, executable: str | None = None) -> dict:
    home = claude_home or default_claude_home()
    executable = executable or resolve_executable()
    command_str = _hook_command(executable)

    commands_dir = home / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    command_path = commands_dir / f"{COMMAND_NAME}.md"
    command_path.write_text(COMMAND_FILE_CONTENT)

    settings_path = home / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    entries = settings.setdefault("hooks", {}).setdefault("UserPromptExpansion", [])
    matcher_entry = next((e for e in entries if e.get("matcher") == COMMAND_NAME), None)
    if matcher_entry is None:
        matcher_entry = {"matcher": COMMAND_NAME, "hooks": []}
        entries.append(matcher_entry)
    hook_already_present = any(h.get("command") == command_str for h in matcher_entry["hooks"])
    if not hook_already_present:
        matcher_entry["hooks"].append({"type": "command", "command": command_str})
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    return {
        "ok": True,
        "command_file": str(command_path),
        "settings_file": str(settings_path),
        "hook_command": command_str,
        "hook_already_present": hook_already_present,
    }


def unregister(claude_home: Path | None = None, executable: str | None = None) -> dict:
    home = claude_home or default_claude_home()
    executable = executable or resolve_executable()
    command_str = _hook_command(executable)

    command_path = home / "commands" / f"{COMMAND_NAME}.md"
    command_removed = command_path.is_file()
    if command_removed:
        command_path.unlink()

    settings_path = home / "settings.json"
    hook_removed = False
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text())
        entries = settings.get("hooks", {}).get("UserPromptExpansion", [])
        for entry in entries:
            if entry.get("matcher") != COMMAND_NAME:
                continue
            before = len(entry.get("hooks", []))
            entry["hooks"] = [h for h in entry.get("hooks", []) if h.get("command") != command_str]
            if len(entry["hooks"]) != before:
                hook_removed = True
        entries[:] = [e for e in entries if e.get("hooks")]
        if not entries and "UserPromptExpansion" in settings.get("hooks", {}):
            del settings["hooks"]["UserPromptExpansion"]
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    return {
        "ok": True,
        "command_file_removed": command_removed,
        "hook_removed": hook_removed,
    }
