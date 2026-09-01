import json
from pathlib import Path

from cc_local_compact import transcript


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_load_transcript_reconstructs_main_thread(tmp_path):
    lines = [
        {"type": "user", "uuid": "u0", "parentUuid": None, "message": {"role": "user", "content": "start"}},
        {"type": "mode", "uuid": "mode1", "parentUuid": "u0"},
        {
            "type": "assistant", "uuid": "a1", "parentUuid": "mode1",
            "message": {"role": "assistant", "id": "m1", "content": "ok"},
        },
        {"type": "user", "uuid": "x1", "parentUuid": "u0", "message": {"role": "user", "content": "abandoned branch"}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1", "message": {"role": "user", "content": "continue"}},
        {
            "type": "assistant", "uuid": "sc1", "parentUuid": "u2", "isSidechain": True,
            "message": {"role": "assistant", "id": "m2", "content": "subagent turn"},
        },
    ]
    path = tmp_path / "session.jsonl"
    _write_jsonl(path, lines)

    result = transcript.load_transcript(path)

    # "mode1" (a dropped control type) sat on the causal chain but is
    # excluded from the output; "x1" is an abandoned branch off u0 never
    # reached by walking back from the true tail; "sc1" is excluded from
    # tail selection because it's a sidechain line.
    assert [line["uuid"] for line in result] == ["u0", "a1", "u2"]


def test_load_transcript_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert transcript.load_transcript(path) == []
