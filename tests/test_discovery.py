import json
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


def _write_transcript(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines))


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


def test_list_sessions_includes_display_name(tmp_path):
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    _write_transcript(sessions_dir / "s.jsonl", [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"content": "hi"}},
        {"type": "assistant", "uuid": "u2", "parentUuid": "u1", "message": {"content": [{"type": "text", "text": "hello there"}]}},
    ])

    entries = discovery.list_sessions(cwd, claude_home)
    assert entries[0]["display_name"] == "hello there"
    assert entries[0]["display_name_source"] == "last_message"


def test_resolve_session_explicit_path_no_ambiguity():
    path, meta = discovery.resolve_session("/some/explicit/path.jsonl", Path("/irrelevant"))
    assert path == Path("/some/explicit/path.jsonl")
    assert meta == {"source": "explicit"}


def test_resolve_session_uses_claude_code_session_id_env(monkeypatch, tmp_path):
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    sid = "5b070f8b-d177-4d47-984e-8072959500ca"
    (sessions_dir / f"{sid}.jsonl").write_text("{}")
    # A decoy that's newer would win the old mtime heuristic; proves the
    # env var path is checked first, not just first when nothing else matches.
    time.sleep(0.01)
    (sessions_dir / "decoy.jsonl").write_text("{}")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", sid)

    path, meta = discovery.resolve_session(None, cwd, claude_home)
    assert path.name == f"{sid}.jsonl"
    assert meta == {"source": "claude_code_session_id_env", "session_id": sid, "ambiguous": False}


def test_resolve_session_env_var_set_but_no_matching_file_falls_back(monkeypatch, tmp_path):
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    (sessions_dir / "only.jsonl").write_text("{}")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "no-such-session-id")

    path, meta = discovery.resolve_session(None, cwd, claude_home)
    assert path.name == "only.jsonl"
    assert meta == {"source": "only_candidate", "candidate_count": 1}


def test_resolve_session_single_candidate_not_ambiguous(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    (sessions_dir / "only.jsonl").write_text("{}")

    path, meta = discovery.resolve_session(None, cwd, claude_home)
    assert path.name == "only.jsonl"
    assert meta == {"source": "only_candidate", "candidate_count": 1}


def test_resolve_session_multiple_candidates_refuses_to_guess(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    (sessions_dir / "a.jsonl").write_text("{}")
    time.sleep(0.01)
    (sessions_dir / "b.jsonl").write_text("{}")

    path, meta = discovery.resolve_session(None, cwd, claude_home)
    assert path is None
    assert meta["source"] == "ambiguous"
    assert meta["candidate_count"] == 2
    assert {Path(c["path"]).name for c in meta["candidates"]} == {"a.jsonl", "b.jsonl"}
    assert all("display_name" in c and "display_name_source" in c for c in meta["candidates"])


def test_resolve_session_no_candidates(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    cwd = tmp_path / "project"
    claude_home = _make_session_dir(tmp_path, cwd)
    path, meta = discovery.resolve_session(None, cwd, claude_home)
    assert path is None
    assert meta == {"source": "no_session_found", "candidate_count": 0}


def test_describe_session_uses_last_rename(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_transcript(path, [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"content": "hello"}},
        {"type": "assistant", "uuid": "u2", "parentUuid": "u1", "message": {"content": [{"type": "text", "text": "hi"}]}},
        {
            "type": "system", "subtype": "local_command", "uuid": "u3", "parentUuid": "u2",
            "content": (
                "<command-name>/rename</command-name>\n            "
                "<command-message>rename</command-message>\n            "
                "<command-args>My Title</command-args>"
            ),
        },
        {"type": "user", "uuid": "u4", "parentUuid": "u3", "message": {"content": "more talk"}},
    ])

    assert discovery.describe_session(path) == {"display_name": "My Title", "display_name_source": "renamed"}


def test_describe_session_last_rename_wins_over_earlier_one(tmp_path):
    path = tmp_path / "s.jsonl"

    def rename_line(uuid, parent, title):
        return {
            "type": "system", "subtype": "local_command", "uuid": uuid, "parentUuid": parent,
            "content": f"<command-name>/rename</command-name>\n<command-args>{title}</command-args>",
        }

    _write_transcript(path, [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"content": "hello"}},
        rename_line("u2", "u1", "First"),
        {"type": "user", "uuid": "u3", "parentUuid": "u2", "message": {"content": "more"}},
        rename_line("u4", "u3", "Second"),
    ])

    assert discovery.describe_session(path) == {"display_name": "Second", "display_name_source": "renamed"}


def test_describe_session_falls_back_to_last_visible_message(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_transcript(path, [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"content": "hello"}},
        {"type": "assistant", "uuid": "u2", "parentUuid": "u1", "message": {"content": [{"type": "text", "text": "Final answer here"}]}},
    ])

    assert discovery.describe_session(path) == {"display_name": "Final answer here", "display_name_source": "last_message"}


def test_describe_session_ignores_tool_only_messages(tmp_path):
    path = tmp_path / "s.jsonl"
    _write_transcript(path, [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"content": "hello"}},
        {"type": "assistant", "uuid": "u2", "parentUuid": "u1", "message": {"content": [{"type": "text", "text": "hi"}]}},
        {"type": "assistant", "uuid": "u3", "parentUuid": "u2", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}},
        {"type": "user", "uuid": "u4", "parentUuid": "u3", "message": {"content": [{"type": "tool_result", "content": "output"}]}},
    ])

    # The tool_use/tool_result-only turns carry no "text" block, so the last
    # real visible text stays "hi".
    assert discovery.describe_session(path) == {"display_name": "hi", "display_name_source": "last_message"}


def test_describe_session_condenses_and_truncates_to_one_line(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "_terminal_width", lambda default=80: 15)
    path = tmp_path / "s.jsonl"
    _write_transcript(path, [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"content": "hi"}},
        {
            "type": "assistant", "uuid": "u2", "parentUuid": "u1",
            "message": {"content": [{"type": "text", "text": "line one\nline two   with   extra   space"}]},
        },
    ])

    result = discovery.describe_session(path)
    assert result["display_name_source"] == "last_message"
    assert len(result["display_name"]) <= 15
    assert result["display_name"].endswith("…")
    assert "\n" not in result["display_name"]


def test_describe_session_empty_when_no_content(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("")

    assert discovery.describe_session(path) == {"display_name": "(no visible content)", "display_name_source": "empty"}


def _clear_line(uuid: str) -> dict:
    return {
        "type": "user", "uuid": uuid,
        "message": {
            "content": (
                "<command-name>/clear</command-name>\n            "
                "<command-message>clear</command-message>\n            "
                "<command-args></command-args>"
            ),
        },
    }


def test_find_clear_indices_detects_single_clear():
    lines = [
        {"type": "user", "uuid": "u1", "message": {"content": "hello"}},
        _clear_line("u2"),
        {"type": "system", "subtype": "local_command", "uuid": "u3", "content": "<local-command-stdout></local-command-stdout>"},
        {"type": "user", "uuid": "u4", "message": {"content": "post-clear message"}},
    ]
    assert discovery.find_clear_indices(lines) == [1]


def test_find_clear_indices_returns_all_in_order():
    lines = [
        {"type": "user", "uuid": "u1", "message": {"content": "hello"}},
        _clear_line("u2"),
        {"type": "user", "uuid": "u3", "message": {"content": "more"}},
        _clear_line("u4"),
        {"type": "user", "uuid": "u5", "message": {"content": "even more"}},
    ]
    assert discovery.find_clear_indices(lines) == [1, 3]


def test_find_clear_indices_empty_when_absent():
    lines = [{"type": "user", "uuid": "u1", "message": {"content": "hello"}}]
    assert discovery.find_clear_indices(lines) == []


def test_find_clear_indices_ignores_non_string_content():
    lines = [{
        "type": "user", "uuid": "u1",
        "message": {"content": [{"type": "text", "text": "please explain <command-name>/clear</command-name>"}]},
    }]
    assert discovery.find_clear_indices(lines) == []


def test_find_clear_indices_ignores_non_user_lines():
    lines = [{
        "type": "assistant", "uuid": "u1",
        "message": {"content": "<command-name>/clear</command-name>"},
    }]
    assert discovery.find_clear_indices(lines) == []


def _turn(label: str) -> list[dict]:
    return [
        {"type": "user", "uuid": f"u_{label}", "message": {"content": f"turn {label}"}},
        {"type": "assistant", "uuid": f"a_{label}", "message": {"content": f"reply {label}", "id": f"m_{label}"}},
    ]


def test_slice_since_last_clear_no_clear_at_all():
    lines = _turn("a")
    span, error = discovery.slice_since_last_clear(lines)
    assert span is None
    assert error == "no_clear_boundary_found"


def test_slice_since_last_clear_single_clear_takes_everything_before_it():
    lines = _turn("a") + [_clear_line("c1")] + _turn("post")
    span, error = discovery.slice_since_last_clear(lines)
    assert error is None
    assert span == _turn("a")


def test_slice_since_last_clear_uses_span_since_previous_clear_not_whole_history():
    lines = _turn("a") + [_clear_line("c1")] + _turn("b") + [_clear_line("c2")] + _turn("post")
    span, error = discovery.slice_since_last_clear(lines)
    assert error is None
    # +1: includes clear1's own line index in the slice start boundary math,
    # not clear1's line itself; span starts right after it
    assert span == _turn("b")


def test_slice_since_last_clear_empty_when_clear_is_first_since_previous_clear():
    lines = [_clear_line("c1")] + _turn("post")
    span, error = discovery.slice_since_last_clear(lines)
    assert span is None
    assert error == "empty_pre_clear_span"


def test_slice_since_last_clear_empty_between_two_adjacent_clears():
    lines = _turn("a") + [_clear_line("c1")] + [_clear_line("c2")] + _turn("post")
    span, error = discovery.slice_since_last_clear(lines)
    assert span is None
    assert error == "empty_pre_clear_span"


def _clear_line(uuid: str) -> dict:
    return {
        "type": "user", "uuid": uuid,
        "message": {
            "content": (
                "<command-name>/clear</command-name>\n            "
                "<command-message>clear</command-message>\n            "
                "<command-args></command-args>"
            ),
        },
    }
