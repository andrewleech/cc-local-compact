import os
import time
from pathlib import Path

from cc_local_compact import discovery


def test_project_dir_slug():
    assert discovery.project_dir_slug(Path("/home/x/my-project")) == "-home-x-my-project"


def test_resolve_cwd_prefers_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "env"))
    explicit = tmp_path / "explicit"
    assert discovery.resolve_cwd(explicit) == explicit


def test_resolve_cwd_prefers_env_var_over_process_cwd(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(env_dir))
    assert discovery.resolve_cwd() == env_dir


def test_resolve_cwd_falls_back_to_process_cwd(monkeypatch):
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    assert discovery.resolve_cwd() == Path.cwd()


def _make_session_dir(tmp_path, cwd: Path) -> Path:
    claude_home = tmp_path / "claude_home"
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    sessions_dir.mkdir(parents=True)
    return claude_home


def test_list_sessions_empty_when_no_dir(tmp_path):
    claude_home = tmp_path / "claude_home"
    assert discovery.list_sessions(tmp_path / "some" / "project", claude_home) == []


def test_list_sessions_sorted_newest_first(tmp_path):
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)

    old = sessions_dir / "old.jsonl"
    old.write_text("{}")
    time.sleep(0.01)
    new = sessions_dir / "new.jsonl"
    new.write_text("{}")

    entries = discovery.list_sessions(cwd, claude_home)
    assert [Path(e["path"]).name for e in entries] == ["new.jsonl", "old.jsonl"]


def test_resolve_session_explicit_path_no_ambiguity():
    path, meta = discovery.resolve_session("/some/explicit/path.jsonl", Path("/irrelevant"))
    assert path == Path("/some/explicit/path.jsonl")
    assert meta == {"source": "explicit"}


def test_resolve_session_auto_discover_single_candidate_not_ambiguous(tmp_path):
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    (sessions_dir / "only.jsonl").write_text("{}")

    path, meta = discovery.resolve_session(None, cwd, claude_home)
    assert path.name == "only.jsonl"
    assert meta["source"] == "auto_discovered_heuristic"
    assert meta["candidate_count"] == 1
    assert meta["ambiguous"] is False


def test_resolve_session_auto_discover_multiple_candidates_is_ambiguous(tmp_path):
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    (sessions_dir / "a.jsonl").write_text("{}")
    time.sleep(0.01)
    (sessions_dir / "b.jsonl").write_text("{}")

    path, meta = discovery.resolve_session(None, cwd, claude_home)
    assert path.name == "b.jsonl"  # most recently modified
    assert meta["candidate_count"] == 2
    assert meta["ambiguous"] is True


def test_resolve_session_no_candidates(tmp_path):
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    path, meta = discovery.resolve_session(None, cwd, claude_home)
    assert path is None
    assert meta["candidate_count"] == 0
    assert meta["ambiguous"] is False
