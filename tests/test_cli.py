import json
import sys
from pathlib import Path

from cc_local_compact import cli, client as client_mod, discovery
from cc_local_compact.client import AttemptResult


def _candidates():
    return [
        {"path": "/a.jsonl", "mtime": 100, "size": 1, "display_name": "First", "display_name_source": "renamed"},
        {"path": "/b.jsonl", "mtime": 200, "size": 2, "display_name": "Second", "display_name_source": "last_message"},
    ]


def test_prompt_for_session_valid_choice(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")
    assert cli._prompt_for_session(_candidates()) == Path("/b.jsonl")


def test_prompt_for_session_blank_cancels(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert cli._prompt_for_session(_candidates()) is None


def test_prompt_for_session_retries_on_invalid_then_accepts(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    responses = iter(["nope", "99", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    assert cli._prompt_for_session(_candidates()) == Path("/a.jsonl")


def test_prompt_for_session_non_interactive_returns_none(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli._prompt_for_session(_candidates()) is None


def _turn(label: str, parent: str | None) -> tuple[list[dict], str]:
    lines = [
        {"type": "user", "uuid": f"u_{label}", "parentUuid": parent, "message": {"role": "user", "content": f"turn {label} MARKER_{label}"}},
        {"type": "assistant", "uuid": f"a_{label}", "parentUuid": f"u_{label}", "message": {"role": "assistant", "content": f"reply {label}", "id": f"m_{label}"}},
    ]
    return lines, f"a_{label}"


def _clear(uuid_prefix: str, parent: str | None) -> tuple[list[dict], str]:
    clear_uuid, reply_uuid = f"{uuid_prefix}_clear", f"{uuid_prefix}_reply"
    lines = [
        {
            "type": "user", "uuid": clear_uuid, "parentUuid": parent,
            "message": {"content": "<command-name>/clear</command-name>\n<command-args></command-args>"},
        },
        {"type": "system", "subtype": "local_command", "uuid": reply_uuid, "parentUuid": clear_uuid, "content": "<local-command-stdout></local-command-stdout>"},
    ]
    return lines, reply_uuid


def _write_transcript(path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def _fake_summarize():
    def fake(client, model, group, custom_instructions, strip_media_flag, max_tokens):
        return AttemptResult(ok=True, summary_text="<summary>\n1. Primary Request:\n   test\n</summary>", total_usage=None)
    return fake


def test_remind_hook_text_no_clear_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize())
    lines, _ = _turn("a1", None)
    path = tmp_path / "session.jsonl"
    _write_transcript(path, lines)

    text = cli._remind_hook_text({"transcript_path": str(path)})
    assert "no /clear command found" in text


def test_remind_hook_text_resolves_via_transcript_path_and_returns_resume_framed_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize())
    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    clear, tail = _clear("c1", tail)
    post, tail = _turn("post", tail)
    path = tmp_path / "session.jsonl"
    _write_transcript(path, a1 + a2 + clear + post)

    text = cli._remind_hook_text({"transcript_path": str(path)})
    assert "Continue the conversation from where it left off" in text


def test_remind_hook_text_falls_back_to_cwd_discovery_when_transcript_path_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize())
    cwd = tmp_path / "project"
    claude_home = tmp_path / ".claude"  # discovery.resolve_session's own default is Path.home() / ".claude"
    sessions_dir = claude_home / "projects" / discovery.project_dir_slug(cwd)
    sessions_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    clear, tail = _clear("c1", tail)
    _write_transcript(sessions_dir / "only.jsonl", a1 + a2 + clear)

    text = cli._remind_hook_text({"cwd": str(cwd)})
    assert "Continue the conversation from where it left off" in text


def test_remind_hook_text_no_session_resolvable(tmp_path):
    empty_cwd = tmp_path / "nowhere"
    text = cli._remind_hook_text({"cwd": str(empty_cwd)})
    assert "no session transcript could be resolved" in text


def test_remind_hook_text_never_raises_on_garbage_payload():
    text = cli._remind_hook_text({"transcript_path": 12345, "cwd": {"not": "a string"}})
    assert isinstance(text, str)


def test_cmd_remind_hook_prints_valid_hook_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize())
    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    clear, tail = _clear("c1", tail)
    path = tmp_path / "session.jsonl"
    _write_transcript(path, a1 + a2 + clear)

    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({"transcript_path": str(path)})))
    cli._cmd_remind_hook(None)

    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptExpansion"
    assert out["hookSpecificOutput"]["suppressOriginalPrompt"] is True
    assert "Continue the conversation from where it left off" in out["hookSpecificOutput"]["additionalContext"]


def test_cmd_remind_hook_handles_malformed_stdin_gracefully(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("not json at all"))
    cli._cmd_remind_hook(None)

    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptExpansion"
    assert "no session transcript could be resolved" in out["hookSpecificOutput"]["additionalContext"]
