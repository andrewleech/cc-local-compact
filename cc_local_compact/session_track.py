"""Per-window (per-`claude`-process) session tracking, so /remind can find
the exact predecessor session across a /clear without guessing.

Why this exists: /clear (in the currently-installed Claude Code version)
creates an entirely new, unrelated session file; confirmed by inspecting
a real cleared session's own transcript (root line has parentUuid: null,
no field anywhere referencing the prior session) and by capturing the
actual UserPromptExpansion hook payload live (session_id/transcript_path
point only at the new file). This reverses an earlier finding in this
project, based on an older real transcript, that /clear left an unbroken
parentUuid chain in the same file; Claude Code's own behavior changed
between versions.

With no on-disk link and no hook firing at the moment /clear itself runs
(checked: UserPromptExpansion does not fire for built-in commands, only
for custom ones "expanding into a prompt"; there is no PreClear/PostClear
hook event either), the only reliable correlation left is process
identity: /clear never spawns a new OS process (confirmed: same PID
across the boundary), so a Stop hook (fires after every completed turn)
can track "the current session for this specific claude process" and the
window right after a /clear - before that new session has completed its
own first turn - is exactly the moment /remind needs to ask "what was
current a moment ago".

Registered hooks use the `args` form (executable + argument list, not a
single shell command string): confirmed live (by walking /proc's parent
chain from a running hook subprocess) that this spawns a hook as a
DIRECT child of the real `claude` process in most cases, so
os.getppid() from inside the hook usually gives that process's own,
stable PID directly, immune to the "multiple sessions open on one
project" problem a timestamp-proximity heuristic would have had, since
each window's PID is a real, unambiguous OS-level identity, not a guess.

Not always, though: real testing found a case where the Stop hook
reliably recorded the right PID directly while a UserPromptExpansion
hook in the very same process did not; some invocation path spawns a
hook under an extra process layer rather than as `claude`'s direct
child. find_tracked_pid/predecessor_session compensate on the read side
by walking the caller's own ancestry chain and using whichever PID
already has tracked state, rather than trusting os.getppid() alone.

State lives under /tmp (namespaced per uid), not ~/.claude; it's
process-lifetime scratch data, not durable configuration; losing it
across a reboot is fine (there's no process to look it up for). /tmp
isn't guaranteed to be tmpfs for every user though, so record_turn
skips writing when the marker wouldn't actually change; Stop fires
after every turn, and most of those calls are the same session_id as
last time.
"""

import datetime
import json
import os
import tempfile
from pathlib import Path


def log(message: str, state_dir: Path | None = None) -> None:
    """Append one timestamped line to hook.log, next to the tracking
    files, for seeing what a live hook run actually did (pid
    resolution, ancestry walk, predecessor found or not, compaction
    start/end) without needing to reconstruct it from a real transcript
    after the fact. Never raises: a logging failure must not break the
    hook it's instrumenting."""
    try:
        path = (state_dir or _state_dir()) / "hook.log"
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with path.open("a") as f:
            f.write(f"{timestamp} pid={os.getpid()} {message}\n")
    except OSError:
        pass


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
    within this same process), shifts current -> previous first, so
    `previous` always holds the last genuinely different session, not
    just whatever was seen two calls ago. A run of same-session Stop
    calls (the common case: several ordinary turns) only refreshes
    `current` in place and never touches `previous`.

    Skips the write entirely if the resulting marker is byte-for-byte
    identical to what's already on disk; Stop fires after every turn,
    so most calls within one session would otherwise rewrite this file
    for no actual change. /tmp isn't guaranteed to be tmpfs for every
    user, so this isn't free I/O to skip."""
    path = _marker_path(pid, state_dir)
    existing = {"current": None, "previous": None}
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {"current": None, "previous": None}

    current = existing.get("current")
    previous = existing.get("previous")
    if current is None or current.get("session_id") != session_id:
        previous = current
    updated = {
        "current": {"session_id": session_id, "transcript_path": transcript_path, "cwd": cwd},
        "previous": previous,
    }

    if updated == existing:
        return

    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(updated))
    tmp_path.replace(path)


def _read_ppid(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _ancestor_chain(start_pid: int, max_levels: int = 10, ppid_of=_read_ppid) -> list[int]:
    chain = [start_pid]
    pid = start_pid
    for _ in range(max_levels):
        parent = ppid_of(pid)
        if parent is None or parent <= 1 or parent in chain:
            break
        chain.append(parent)
        pid = parent
    return chain


def find_tracked_pid(
    start_pid: int, state_dir: Path | None = None, max_levels: int = 10, ppid_of=_read_ppid,
) -> int | None:
    """Real evidence (a Stop hook that reliably records the right PID
    directly, paired with a UserPromptExpansion hook that sometimes
    doesn't) shows a hook can occasionally be spawned under an extra
    process layer rather than as a direct child of the real `claude`
    process, even when registered with the `args` form. Rather than try
    to identify "the real claude process" by name (fragile, varies by
    install, e.g. a patched binary), walk the ancestry chain from
    `start_pid` upward and use the first PID that already has tracked
    state on disk: Stop already proved that PID is the one being written
    to, whatever spawned this particular hook invocation."""
    for pid in _ancestor_chain(start_pid, max_levels, ppid_of):
        if _marker_path(pid, state_dir).is_file():
            return pid
    return None


def predecessor_session(
    pid: int, my_session_id: str, state_dir: Path | None = None, ppid_of=_read_ppid,
) -> dict | None:
    """The session that was active in this same process right before
    `my_session_id`, or None if nothing was ever tracked anywhere in
    `pid`'s own ancestry (see find_tracked_pid).

    If `current` already reflects `my_session_id`, at least one Stop
    event has already fired for this (post-clear) session, so record_turn
    already did the current->previous shift, the answer is `previous`.
    If `current` is still some OTHER session, no Stop has fired for this
    one yet (the common case: /remind called right after /clear, before
    any new turn completed), the answer is `current` itself, not
    `previous` (which would be one session too old)."""
    tracked_pid = find_tracked_pid(pid, state_dir, ppid_of=ppid_of)
    if tracked_pid is None:
        return None
    try:
        marker = json.loads(_marker_path(tracked_pid, state_dir).read_text())
    except (json.JSONDecodeError, OSError):
        return None

    current = marker.get("current")
    if current is not None and current.get("session_id") == my_session_id:
        return marker.get("previous")
    return current
