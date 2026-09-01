"""Session-transcript discovery: cwd -> project-dir slug -> most recently
modified .jsonl in that directory.

The slug algorithm (cwd path with every "/" replaced by "-") is inferred
from observed real directory names (e.g. ~/.claude/projects/-home-corona-
mpy-debugpy/ for cwd /home/corona/mpy-debugpy), not confirmed against
Claude Code's own source -- if a lookup ever comes up empty for a project
that clearly has sessions, treat that as a sign this needs revisiting
rather than assuming the session doesn't exist.
"""

from pathlib import Path


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
