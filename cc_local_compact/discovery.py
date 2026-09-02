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
versions), so `most_recent_session` -- most-recently-modified .jsonl in
the resolved project directory -- stays as the fallback for whenever it's
absent (e.g. this tool's own CLI run standalone outside an MCP session, or
a Claude Code version that doesn't set it). That fallback is a real guess:
exactly right when exactly one session exists for the project, and can
silently pick the wrong file whenever more than one does. Callers
(server.py) surface which path was taken and whether it was a guess --
see compact_session's docstring and the session_path_source /
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
      - explicit_path omitted, CLAUDE_CODE_SESSION_ID set and its .jsonl
        exists in the resolved project directory: ("claude_code_session_id_env",
        not a guess -- see this module's docstring) that env var identifies
        the calling session directly, injected by Claude Code itself into
        every stdio MCP server's environment.
      - otherwise: falls back to the most-recently-modified .jsonl in the
        resolved project directory. meta["ambiguous"] is True whenever more
        than one session exists for that project -- this heuristic can't
        tell them apart, so a caller getting True back should treat the
        resolved path as a guess, not a guarantee, and prefer passing
        session_path explicitly when precision matters."""
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
        return None, {"source": "auto_discovered_heuristic", "candidate_count": 0, "ambiguous": False}
    resolved = Path(candidates[0]["path"])
    return resolved, {
        "source": "auto_discovered_heuristic",
        "candidate_count": len(candidates),
        "ambiguous": len(candidates) > 1,
    }
