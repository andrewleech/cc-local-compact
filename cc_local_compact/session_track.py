"""Per-window (per-`claude`-process) session tracking, so /remind can find
the exact predecessor session across a /clear without guessing.

Why this exists: /clear (in the currently-installed Claude Code version)
creates an entirely new, unrelated session file -- confirmed by inspecting
a real cleared session's own transcript (root line has parentUuid: null,
no field anywhere referencing the prior session) and by capturing the
actual UserPromptExpansion hook payload live (session_id/transcript_path
point only at the new file). This reverses an earlier finding in this
project, based on an older real transcript, that /clear left an unbroken
parentUuid chain in the same file -- Claude Code's own behavior changed
between versions.

With no on-disk link and no hook firing at the moment /clear itself runs
(checked: UserPromptExpansion does not fire for built-in commands, only
for custom ones "expanding into a prompt"; there is no PreClear/PostClear
hook event either), the only reliable correlation left is process
identity: /clear never spawns a new OS process (confirmed: same PID
across the boundary), so a Stop hook (fires after every completed turn)
can track "the current session for this specific claude process" and the
window right after a /clear -- before that new session has completed its
own first turn -- is exactly the moment /remind needs to ask "what was
current a moment ago".

Registered hooks MUST use the `args` form (executable + argument list,
not a single shell command string) for this to work: confirmed live (by
walking /proc's parent chain from a running hook subprocess) that `args`
form spawns the hook as a DIRECT child of the real `claude` process, so
os.getppid() from inside the hook reliably gives that process's own,
stable PID -- immune to the "multiple sessions open on one project"
problem a timestamp-proximity heuristic would have had, since each
window's PID is a real, unambiguous OS-level identity, not a guess.

State lives under /tmp (namespaced per uid), not ~/.claude -- it's
process-lifetime scratch data, not durable configuration; losing it
across a reboot is fine (there's no process to look it up for).
"""

import json
import os
import tempfile
from pathlib import Path


def _state_dir() -> Path:
    d = Path(tempfile.gettempdir()) / f"cc-local-compact-{os.getuid()}"
    d.mkdir(mode=0o700, exist_ok=True)
    return d


def _marker_path(pid: int, state_dir: Path | None = None) -> Path:
    return (state_dir or _state_dir()) / f"pid-{pid}.json"


def record_turn(
    pid: int, session_id: str, transcript_path: str, cwd: str, state_dir: Path | None = None,
) -> None:
    """Call on every Stop event. If `session_id` differs from whatever is
    already recorded as `current` (a /clear, or any other session switch
    within this same process), shifts current -> previous first -- so
    `previous` always holds the last genuinely different session, not
    just whatever was seen two calls ago. A run of same-session Stop
    calls (the common case: several ordinary turns) only refreshes
    `current` in place and never touches `previous`."""
    path = _marker_path(pid, state_dir)
    marker = {"current": None, "previous": None}
    if path.is_file():
        try:
            marker = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            marker = {"current": None, "previous": None}

    current = marker.get("current")
    if current is None or current.get("session_id") != session_id:
        marker["previous"] = current
    marker["current"] = {"session_id": session_id, "transcript_path": transcript_path, "cwd": cwd}

    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(marker))
    tmp_path.replace(path)


def predecessor_session(pid: int, my_session_id: str, state_dir: Path | None = None) -> dict | None:
    """The session that was active in this same process right before
    `my_session_id` -- or None if nothing was ever tracked for this PID.

    If `current` already reflects `my_session_id`, at least one Stop
    event has already fired for this (post-clear) session, so record_turn
    already did the current->previous shift -- the answer is `previous`.
    If `current` is still some OTHER session, no Stop has fired for this
    one yet (the common case: /remind called right after /clear, before
    any new turn completed) -- the answer is `current` itself, not
    `previous` (which would be one session too old)."""
    path = _marker_path(pid, state_dir)
    if not path.is_file():
        return None
    try:
        marker = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    current = marker.get("current")
    if current is not None and current.get("session_id") == my_session_id:
        return marker.get("previous")
    return current
