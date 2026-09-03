import json
import sys
from pathlib import Path

from cc_local_compact import cli, client as client_mod, session_track
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


def _write_transcript(path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def _fake_summarize():
    def fake(client, model, group, custom_instructions, strip_media_flag, max_tokens):
        return AttemptResult(ok=True, summary_text="<summary>\n1. Primary Request:\n   test\n</summary>", total_usage=None)
    return fake


def test_remind_hook_text_passes_command_args_as_custom_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    seen_instructions = []

    def fake(client, model, group, custom_instructions, strip_media_flag, max_tokens):
        seen_instructions.append(custom_instructions)
        return AttemptResult(ok=True, summary_text="<summary>\ntest\n</summary>", total_usage=None)

    monkeypatch.setattr(client_mod, "summarize_group", fake)

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    old_path = tmp_path / "old_session.jsonl"
    _write_transcript(old_path, a1 + a2)
    session_track.record_turn(9999, "s-old", str(old_path), str(tmp_path))

    cli._remind_hook_text({"session_id": "s-new", "command_args": "focusing on the API redesign"}, pid=9999)
    assert seen_instructions[0] == "focusing on the API redesign"


def test_remind_hook_text_empty_command_args_means_no_custom_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    seen_instructions = []

    def fake(client, model, group, custom_instructions, strip_media_flag, max_tokens):
        seen_instructions.append(custom_instructions)
        return AttemptResult(ok=True, summary_text="<summary>\ntest\n</summary>", total_usage=None)

    monkeypatch.setattr(client_mod, "summarize_group", fake)

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    old_path = tmp_path / "old_session.jsonl"
    _write_transcript(old_path, a1 + a2)
    session_track.record_turn(9999, "s-old", str(old_path), str(tmp_path))

    cli._remind_hook_text({"session_id": "s-new", "command_args": ""}, pid=9999)
    assert seen_instructions[0] is None


def test_remind_hook_text_no_predecessor_tracked(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    text = cli._remind_hook_text({"session_id": "s-new"}, pid=9999)
    assert "no predecessor session tracked" in text


def test_remind_hook_text_uses_tracked_predecessor_and_returns_passive_framed_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize())

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    old_path = tmp_path / "old_session.jsonl"
    _write_transcript(old_path, a1 + a2)

    session_track.record_turn(9999, "s-old", str(old_path), str(tmp_path))

    text = cli._remind_hook_text({"session_id": "s-new"}, pid=9999)
    # deliberately NOT build_resume_preamble's "resume directly, don't ask
    # questions" framing; confirmed live that wording made a real /remind
    # run launch straight into flashing physical hardware with no human
    # confirmation on the turn that injected it (suppressOriginalPrompt means
    # this text is the ONLY thing the model sees on that turn). See
    # response.build_remind_preamble's docstring.
    assert "wait for the user's next message" in text
    assert "Continue the conversation from where it left off" not in text
    assert "Resume directly" not in text


def test_remind_hook_text_predecessor_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    session_track.record_turn(9999, "s-old", str(tmp_path / "gone.jsonl"), str(tmp_path))

    text = cli._remind_hook_text({"session_id": "s-new"}, pid=9999)
    assert "no longer exists" in text


def test_remind_hook_text_never_raises_on_garbage_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    text = cli._remind_hook_text({"session_id": {"not": "a string"}}, pid=9999)
    assert isinstance(text, str)

    text = cli._remind_hook_text({}, pid=9999)
    assert isinstance(text, str)
    assert "no predecessor session tracked" in text


def test_cmd_remind_hook_prints_valid_hook_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(client_mod, "summarize_group", _fake_summarize())
    monkeypatch.setattr("os.getppid", lambda: 4242)

    a1, tail = _turn("a1", None)
    a2, tail = _turn("a2", tail)
    old_path = tmp_path / "old_session.jsonl"
    _write_transcript(old_path, a1 + a2)
    session_track.record_turn(4242, "s-old", str(old_path), str(tmp_path))

    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({"session_id": "s-new"})))
    cli._cmd_remind_hook(None)

    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptExpansion"
    assert out["hookSpecificOutput"]["suppressOriginalPrompt"] is True
    assert "wait for the user's next message" in out["hookSpecificOutput"]["additionalContext"]
    assert "Continue the conversation from where it left off" not in out["hookSpecificOutput"]["additionalContext"]


def test_cmd_remind_hook_handles_malformed_stdin_gracefully(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("os.getppid", lambda: 4242)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("not json at all"))
    cli._cmd_remind_hook(None)

    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptExpansion"
    assert "no predecessor session tracked" in out["hookSpecificOutput"]["additionalContext"]


def test_cmd_track_session_records_current_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("os.getppid", lambda: 5555)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps({
        "session_id": "s-1", "transcript_path": "/some/path.jsonl", "cwd": "/some/cwd",
        "hook_event_name": "Stop",
    })))

    cli._cmd_track_session(None)

    out = json.loads(capsys.readouterr().out)
    assert out == {}
    predecessor = session_track.predecessor_session(5555, "some-other-session")
    assert predecessor == {"session_id": "s-1", "transcript_path": "/some/path.jsonl", "cwd": "/some/cwd"}


def test_cmd_track_session_never_raises_on_malformed_stdin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("not json"))
    cli._cmd_track_session(None)
    assert json.loads(capsys.readouterr().out) == {}
