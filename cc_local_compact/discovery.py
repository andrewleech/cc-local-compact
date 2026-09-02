"""Session-transcript discovery: cwd -> project-dir slug -> most recently
modified .jsonl in that directory.

The slug algorithm (cwd path with every "/" replaced by "-") is inferred
from observed real directory names (e.g. ~/.claude/projects/-home-corona-
mpy-debugpy/ for cwd /home/corona/mpy-debugpy), not confirmed against
Claude Code's own source -- if a lookup ever comes up empty for a project
that clearly has sessions, treat that as a sign this needs revisiting
rather than assuming the session doesn't exist.

IMPORTANT LIMITATION, confirmed via investigation (not assumed): there is
no documented mechanism -- no environment variable, no MCP protocol field,
nothing -- for a standard (non-self-hosted) Claude Code MCP server to
learn which session is actually calling it. `most_recent_session` is a
heuristic (most-recently-modified .jsonl in the resolved project
directory), not a reliable identification of the caller. It's exactly
right when exactly one session exists for that project; it's a guess, and
can silently pick the wrong file, whenever more than one does (e.g. two
Claude Code windows open on the same repo). Callers (server.py) surface
this ambiguity explicitly rather than trusting the guess quietly -- see
server.py's compact_session docstring and its session_path_source /
session_path_warning result fields.
"""

import os
from pathlib import Path


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


def most_recent_session(cwd: Path, claude_home: Path | None = None) -> Path | None:
    sessions_dir = project_sessions_dir(cwd, claude_home)
    if not sessions_dir.is_dir():
        return None
    candidates = [p for p in sessions_dir.glob("*.jsonl") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def list_sessions(cwd: Path, claude_home: Path | None = None) -> list[dict]:
    sessions_dir = project_sessions_dir(cwd, claude_home)
    if not sessions_dir.is_dir():
        return []
    entries = []
    for path in sessions_dir.glob("*.jsonl"):
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append({"path": str(path), "mtime": stat.st_mtime, "size": stat.st_size})
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def resolve_session(
    explicit_path: str | None, cwd: Path, claude_home: Path | None = None,
) -> tuple[Path | None, dict]:
    """Resolve which session .jsonl to operate on, with metadata about how
    sure that resolution actually is. Returns (path, meta):
      - explicit_path given: ("explicit", the given path -- no ambiguity,
        the caller told us directly).
      - explicit_path omitted: falls back to the most-recently-modified
        .jsonl in the resolved project directory. meta["ambiguous"] is
        True whenever more than one session exists for that project --
        there is no reliable way for this tool to know which one is
        actually the caller's own session (see this module's docstring),
        so a caller getting True back should treat the resolved path as a
        guess, not a guarantee, and prefer passing session_path explicitly
        when precision matters."""
    if explicit_path:
        return Path(explicit_path), {"source": "explicit"}
    candidates = list_sessions(cwd, claude_home)
    if not candidates:
        return None, {"source": "auto_discovered_heuristic", "candidate_count": 0, "ambiguous": False}
    resolved = Path(candidates[0]["path"])
    return resolved, {
        "source": "auto_discovered_heuristic",
        "candidate_count": len(candidates),
        "ambiguous": len(candidates) > 1,
    }
