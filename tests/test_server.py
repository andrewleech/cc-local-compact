import json

from cc_local_compact import client as client_mod
from cc_local_compact import server
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


def _write_transcript(path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def _fake_summarize(calls):
    def fake(client, model, group, custom_instructions, strip_media_flag, max_tokens):
        calls.append(group)
        return AttemptResult(ok=True, summary_text="<summary>\n1. Primary Request:\n   test\n</summary>", total_usage=None)
    return fake


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
