import sys
from pathlib import Path

from cc_local_compact import cli


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
