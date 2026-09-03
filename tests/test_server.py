import json

from cc_local_compact import client as client_mod
from cc_local_compact import discovery, server
from cc_local_compact.client import AttemptResult


def _turn(label: str, parent: str | None) -> tuple[list[dict], str]:
    """One user+assistant group, chained onto `parent`. Content carries a
    MARKER_<label> substring so tests can assert which turns did/didn't
    make it into a summarize call. Returns (lines, new_tail_uuid)."""
    u_uuid = f"u_{label}"
    a_uuid = f"a_{label}"
    lines = [
        {
            "type": "user", "uuid": u_uuid, "parentUuid": parent,
            "message": {"role": "user", "content": f"user turn {label} MARKER_{label}"},
        },
        {
            "type": "assistant", "uuid": a_uuid, "parentUuid": u_uuid,
            "message": {"role": "assistant", "content": f"assistant turn {label} MARKER_{label}", "id": f"m_{label}"},
        },
    ]
    return lines, a_uuid


def _clear(uuid_prefix: str, parent: str | None) -> tuple[list[dict], str]:
    clear_uuid = f"{uuid_prefix}_clear"
    reply_uuid = f"{uuid_prefix}_reply"
    lines = [
        {
            "type": "user", "uuid": clear_uuid, "parentUuid": parent,
            "message": {
                "content": (
                    "<command-name>/clear</command-name>\n            "
                    "<command-message>clear</command-message>\n            "
                    "<command-args></command-args>"
                ),
            },
        },
        {
            "type": "system", "subtype": "local_command", "uuid": reply_uuid, "parentUuid": clear_uuid,
            "content": "<local-command-stdout></local-command-stdout>",
        },
    ]
    return lines, reply_uuid


def _write_transcript(path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def _fake_summarize(calls):
    def fake(client, model, group, custom_instructions, strip_media_flag, max_tokens):
        calls.append(group)
        return AttemptResult(ok=True, summary_text="<summary>\n1. Primary Request:\n   test\n</summary>", total_usage=None)
    return fake


def _call_continue(session_path=None, **kwargs):
    kwargs.setdefault("context_budget", 1_000_000)
    session_path_arg = str(session_path) if session_path is not None else None
    return server.continue_after_clear.fn(session_path=session_path_arg, **kwargs)


def test_continue_after_clear_no_clear_boundary(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize(calls))
    lines, _ = _turn("a1", None)
    path = tmp_path / "session.jsonl"
    _write_transcript(path, lines)

    result = _call_continue(path)
    assert result == {
        "ok": False,
        "reason": "no_clear_boundary_found",
        "detail": (
            "No /clear command found on this session's main thread -- "
            "continue_after_clear is only meaningful right after running "
            "/clear. Run /clear first, or use compact_session for a "
            "whole-session summary instead."
        ),
    }
    assert calls == []


def test_continue_after_clear_empty_pre_clear_span(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize(calls))
    lines, tail = _clear("c1", None)
    turn_lines, _ = _turn("post", tail)
    path = tmp_path / "session.jsonl"
    _write_transcript(path, lines + turn_lines)

    result = _call_continue(path)
    assert result["ok"] is False
    assert result["reason"] == "empty_pre_clear_span"
    assert calls == []


def test_continue_after_clear_ambiguous_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        discovery, "resolve_session",
        lambda session_path, cwd, claude_home=None: (None, {
            "source": "ambiguous", "candidate_count": 2,
            "candidates": [{"path": "/a.jsonl", "display_name": "A"}, {"path": "/b.jsonl", "display_name": "B"}],
        }),
    )
    result = _call_continue()
    assert result["ok"] is False
    assert result["reason"] == "ambiguous_session"
    assert len(result["candidates"]) == 2


def test_continue_after_clear_no_session_found(tmp_path, monkeypatch):
    monkeypatch.setattr(
        discovery, "resolve_session",
        lambda session_path, cwd, claude_home=None: (None, {"source": "no_session_found", "candidate_count": 0}),
    )
    result = _call_continue()
    assert result == {
        "ok": False,
        "reason": "no_session_found",
        "detail": f"no session found for project directory {discovery.resolve_cwd()}",
    }


def test_continue_after_clear_summarizes_only_pre_clear_span(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize(calls))

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    clear, tail = _clear("c1", tail)
    post, tail = _turn("post", tail)
    lines = a1 + a2 + clear + post
    path = tmp_path / "session.jsonl"
    _write_transcript(path, lines)

    result = _call_continue(path)
    assert result["ok"] is True

    all_summarized_content = json.dumps(calls)
    assert "MARKER_a1" in all_summarized_content
    assert "MARKER_a2" in all_summarized_content
    assert "MARKER_post" not in all_summarized_content


def test_continue_after_clear_last_clear_wins(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize(calls))

    a1, tail = _turn("a1", None)
    clear1, tail = _clear("c1", tail)
    b1, tail = _turn("b1", tail)
    b2, tail = _turn("b2", tail)
    clear2, tail = _clear("c2", tail)
    c1, tail = _turn("c1post", tail)
    lines = a1 + clear1 + b1 + b2 + clear2 + c1
    path = tmp_path / "session.jsonl"
    _write_transcript(path, lines)

    result = _call_continue(path)
    assert result["ok"] is True

    all_summarized_content = json.dumps(calls)
    assert "MARKER_a1" not in all_summarized_content
    assert "MARKER_b1" in all_summarized_content
    assert "MARKER_b2" in all_summarized_content
    assert "MARKER_c1post" not in all_summarized_content
    # +1: the span includes clear1's own empty-stdout system reply line,
    # which trails immediately after the /clear command line that's excluded
    assert result["pre_clear_line_count"] == len(b1 + b2) + 1


def test_continue_after_clear_summary_is_resume_framed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize(calls))

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    clear, tail = _clear("c1", tail)
    path = tmp_path / "session.jsonl"
    _write_transcript(path, a1 + a2 + clear)

    result = _call_continue(path)
    assert result["ok"] is True
    assert "Continue the conversation from where it left off" in result["summary"]
    assert "Continue the conversation from where it left off" not in result["summary_cleaned"]


def test_continue_after_clear_append_to_jsonl_anchors_to_pre_clear_line(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize(calls))

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    clear, tail = _clear("c1", tail)
    post, tail = _turn("post", tail)
    lines = a1 + a2 + clear + post
    path = tmp_path / "session.jsonl"
    _write_transcript(path, lines)

    result = _call_continue(path, append_to_jsonl=True)
    assert result["ok"] is True
    assert result["jsonl_appended"] is True

    all_raw = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    boundary = next(l for l in all_raw if l.get("subtype") == "compact_boundary")
    pre_clear_last_uuid = a2[-1]["uuid"]  # "a_a2"
    file_true_last_uuid = post[-1]["uuid"]  # "a_post"
    assert boundary["logicalParentUuid"] == pre_clear_last_uuid
    assert boundary["logicalParentUuid"] != file_true_last_uuid


def test_compact_session_happy_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize(calls))

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    path = tmp_path / "session.jsonl"
    _write_transcript(path, a1 + a2)

    result = server.compact_session.fn(session_path=str(path), context_budget=1_000_000)
    assert result["ok"] is True
    assert "summary" in result
    assert "Continue the conversation from where it left off" not in result["summary"]
