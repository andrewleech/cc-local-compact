"""Session-transcript discovery: cwd -> project-dir slug -> session .jsonl.

The slug algorithm (cwd path with every "/" replaced by "-") is inferred
from observed real directory names (e.g. ~/.claude/projects/-home-corona-
mpy-debugpy/ for cwd /home/corona/mpy-debugpy), not confirmed against
Claude Code's own source -- if a lookup ever comes up empty for a project
that clearly has sessions, treat that as a sign this needs revisiting
rather than assuming the session doesn't exist.

CLAUDE_CODE_SESSION_ID identifies the caller directly. This reverses an
earlier finding in this module (docs and the public MCP protocol document
no such mechanism) -- confirmed wrong by extracting the actual MCP stdio
spawn code from the installed Claude Code binary (2.1.258): every stdio
MCP server is launched with
`env:{...inherited,CLAUDE_PROJECT_DIR:mn(),CLAUDE_CODE_SESSION_ID:Q(),CLAUDECODE:"1",...serverEnv}`,
where `Q()` is the same current-session-id accessor used for prompt
`${CLAUDE_SESSION_ID}` substitution elsewhere in the app. Neither
claude-net's nor cc-local-router's patcher providers inject this (grepped
both, no hits) -- it's stock behaviour, not something layered on by
cc-patcher. `resolve_session` uses it as the primary source; a caller with
this env var set gets an authoritative path, not a guess, even with
several sessions open on the same project.

It's undocumented (internal to the app, no stability guarantee across
versions), so a fallback still matters for whenever it's absent (e.g. this
tool's own CLI run standalone outside an MCP session, or a Claude Code
version that doesn't set it). There is deliberately no mtime-based guess
in that fallback: picking "most recently modified" silently can pick the
wrong session whenever more than one exists for the project, and a wrong
pick is a worse failure mode than making the caller choose. Instead,
`resolve_session` returns no path and a candidate list -- each entry
carries a human-identifiable `display_name` from `describe_session`, so a
human (via cli.py's interactive picker) or an agent (via the MCP tool's
returned candidates) can actually tell the sessions apart well enough to
choose correctly, rather than being handed bare paths and mtimes.
"""

import html
import os
import re
import shutil
from pathlib import Path

from . import transcript

_RENAME_RE = re.compile(
    r"<command-name>/rename</command-name>.*?<command-args>(.*?)</command-args>",
    re.DOTALL,
)

_CLEAR_MARKER = "<command-name>/clear</command-name>"


def resolve_cwd(explicit: Path | None = None) -> Path:
    """Prefer an explicitly-given directory, then CLAUDE_PROJECT_DIR (the
    stable project root Claude Code passes to spawned MCP server
    subprocesses -- documented in mcp.md), then the process' own cwd as a
    last resort (right when this tool runs standalone via the CLI, not
    necessarily right for an MCP server subprocess whose own cwd isn't
    guaranteed to match the project root)."""
    if explicit is not None:
        return explicit
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if project_dir:
        return Path(project_dir)
    return Path.cwd()


def project_dir_slug(cwd: Path) -> str:
    return str(cwd).replace("/", "-")


def project_sessions_dir(cwd: Path, claude_home: Path | None = None) -> Path:
    home = claude_home or (Path.home() / ".claude")
    return home / "projects" / project_dir_slug(cwd)


def _terminal_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size(fallback=(default, 24)).columns
    except OSError:
        return default


def _visible_text(content) -> str | None:
    """Text a person actually reads on screen for one message's content --
    "text" content blocks only, never tool_use/tool_result/image/thinking."""
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "".join(parts)
        return text or None
    return None


def _condense(text: str, width: int) -> str:
    snippet = re.sub(r"\s+", " ", text).strip()
    if len(snippet) > width:
        snippet = snippet[: max(width - 1, 0)].rstrip() + "…"
    return snippet


def describe_session(path: Path) -> dict:
    """Derive a human-identifiable display name for a session transcript,
    for telling several ambiguous candidates apart with no other signal.
    Returns {"display_name": str, "display_name_source": str}.

    Two sources, in priority order, read off the main thread (via
    transcript.load_transcript, not raw file order -- a rewound/abandoned
    branch's stale rename or last message must not win over what's
    actually live):
      - "renamed": the last `/rename` command on the thread. Claude Code
        records these as type:"system",subtype:"local_command" lines with
        the new title in a <command-args> tag -- the title the user
        themselves chose for the session, and the most reliable signal
        available short of an explicit session_path.
      - "last_message": failing that, the last visible message on the
        thread (text content blocks only), condensed to one line and
        truncated to about one terminal row -- what a person would
        actually see on screen at the end of the session.
      - "empty": neither exists (an essentially-empty transcript).
      - "error": the transcript couldn't be read/parsed at all.
    """
    try:
        lines = transcript.load_transcript(path)
    except (OSError, ValueError):
        return {"display_name": "(unreadable transcript)", "display_name_source": "error"}

    width = _terminal_width()
    last_rename: str | None = None
    last_text: str | None = None

    for line in lines:
        if line.get("type") == "system" and line.get("subtype") == "local_command":
            content = line.get("content")
            if isinstance(content, str):
                match = _RENAME_RE.search(content)
                if match:
                    title = html.unescape(match.group(1)).strip()
                    last_rename = title or None
            continue
        if line.get("type") in ("user", "assistant") and not line.get("isMeta"):
            message = line.get("message") or {}
            text = _visible_text(message.get("content"))
            if text and text.strip():
                last_text = text

    if last_rename:
        return {"display_name": last_rename, "display_name_source": "renamed"}
    if last_text:
        return {"display_name": _condense(last_text, width), "display_name_source": "last_message"}
    return {"display_name": "(no visible content)", "display_name_source": "empty"}


def find_clear_indices(lines: list[dict]) -> list[int]:
    """Indices in `lines` (transcript.load_transcript's main-thread output,
    causal order) of every /clear command line, oldest first.

    /clear is recorded differently from /rename: it's a type:"user" line
    whose message.content is the literal command-tag string Claude Code's
    slash-command dispatcher writes ("<command-name>/clear</command-name>
    ..."), immediately followed on the thread by a
    type:"system",subtype:"local_command" line carrying an empty
    <local-command-stdout></local-command-stdout> reply -- unlike /rename
    (_RENAME_RE), which IS itself the system/local_command line, not a
    user line naming the command. /clear does not break the parentUuid
    chain or start a new session file (confirmed against a real transcript
    containing one), so a caller wanting only the span since the most
    recent /clear must use the last two entries here, not just the last
    one -- see continue_after_clear in server.py."""
    indices = []
    for i, line in enumerate(lines):
        if line.get("type") != "user":
            continue
        content = (line.get("message") or {}).get("content")
        if isinstance(content, str) and _CLEAR_MARKER in content:
            indices.append(i)
    return indices


def list_sessions(cwd: Path, claude_home: Path | None = None) -> list[dict]:
    """Available session transcripts for `cwd`'s project, each with a
    describe_session-derived display_name to tell them apart. Sorted
    newest-first for display purposes only -- see this module's docstring
    for why that ordering is never used to auto-pick a winner."""
    sessions_dir = project_sessions_dir(cwd, claude_home)
    if not sessions_dir.is_dir():
        return []
    entries = []
    for path in sessions_dir.glob("*.jsonl"):
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append({
            "path": str(path),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            **describe_session(path),
        })
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def resolve_session(
    explicit_path: str | None, cwd: Path, claude_home: Path | None = None,
) -> tuple[Path | None, dict]:
    """Resolve which session .jsonl to operate on, with metadata about how
    sure that resolution actually is. Returns (path, meta):
      - explicit_path given: (path, {"source": "explicit"}) -- no
        ambiguity, the caller told us directly.
      - explicit_path omitted, CLAUDE_CODE_SESSION_ID set and its .jsonl
        exists in the resolved project directory: (path, {"source":
        "claude_code_session_id_env", "session_id": ..., "ambiguous":
        False}) -- not a guess, see this module's docstring.
      - explicit_path omitted, env var unset/stale, exactly one session
        exists for the project: (path, {"source": "only_candidate",
        "candidate_count": 1}) -- unambiguous by elimination.
      - explicit_path omitted, env var unset/stale, zero sessions exist:
        (None, {"source": "no_session_found", "candidate_count": 0}).
      - explicit_path omitted, env var unset/stale, more than one session
        exists: (None, {"source": "ambiguous", "candidate_count": N,
        "candidates": [...]}) -- deliberately refuses to guess (no
        mtime-based auto-pick); `candidates` is list_sessions' full output
        (path/mtime/size/display_name/display_name_source) so the caller
        can present them for a human or agent to choose from."""
    if explicit_path:
        return Path(explicit_path), {"source": "explicit"}
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        candidate = project_sessions_dir(cwd, claude_home) / f"{sid}.jsonl"
        if candidate.is_file():
            return candidate, {
                "source": "claude_code_session_id_env",
                "session_id": sid,
                "ambiguous": False,
            }
    candidates = list_sessions(cwd, claude_home)
    if not candidates:
        return None, {"source": "no_session_found", "candidate_count": 0}
    if len(candidates) == 1:
        return Path(candidates[0]["path"]), {"source": "only_candidate", "candidate_count": 1}
    return None, {
        "source": "ambiguous",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
