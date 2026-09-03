import json

import pytest

from cc_local_compact import jsonl_append


SESSION_METADATA = {
    "sessionId": "sess-1",
    "cwd": "/home/x/project",
    "version": "2.1.252",
    "gitBranch": "main",
    "userType": "external",
    "entrypoint": "cli",
}


def test_build_boundary_line_shape():
    line = jsonl_append.build_boundary_line(
        trigger="manual", pre_tokens=1000, post_tokens=100, duration_ms=5000,
        preserved_uuids=["p1", "p2"], anchor_uuid="anchor-1",
        logical_parent_uuid="last-1", session_metadata=SESSION_METADATA,
        boundary_uuid="boundary-1",
    )
    assert line["type"] == "system"
    assert line["subtype"] == "compact_boundary"
    assert line["content"] == "Conversation compacted"
    assert line["isMeta"] is False
    assert line["parentUuid"] is None
    assert line["logicalParentUuid"] == "last-1"
    assert line["uuid"] == "boundary-1"
    assert line["compactMetadata"]["trigger"] == "manual"
    assert line["compactMetadata"]["preTokens"] == 1000
    assert line["compactMetadata"]["postTokens"] == 100
    assert line["compactMetadata"]["cumulativeDroppedTokens"] == 900
    assert line["compactMetadata"]["durationMs"] == 5000
    assert line["compactMetadata"]["preservedSegment"] == {
        "headUuid": "p1", "anchorUuid": "anchor-1", "tailUuid": "p2",
    }
    assert line["compactMetadata"]["preservedMessages"] == {
        "anchorUuid": "anchor-1", "uuids": ["p1", "p2"], "allUuids": ["p1", "p2"],
    }
    assert line["sessionId"] == "sess-1"


def test_build_boundary_line_no_preserved_omits_segment_fields():
    line = jsonl_append.build_boundary_line(
        trigger="manual", pre_tokens=1000, post_tokens=100, duration_ms=5000,
        preserved_uuids=[], anchor_uuid="anchor-1", logical_parent_uuid="last-1",
        session_metadata=SESSION_METADATA,
    )
    assert "preservedSegment" not in line["compactMetadata"]
    assert "preservedMessages" not in line["compactMetadata"]


def test_build_boundary_line_discovered_tools_sorted():
    line = jsonl_append.build_boundary_line(
        trigger="manual", pre_tokens=1, post_tokens=1, duration_ms=1,
        preserved_uuids=[], anchor_uuid="a", logical_parent_uuid=None,
        session_metadata=SESSION_METADATA, discovered_tools=["Zeta", "Alpha"],
    )
    assert line["compactMetadata"]["preCompactDiscoveredTools"] == ["Alpha", "Zeta"]
    assert "logicalParentUuid" not in line


def test_build_summary_line_shape():
    line = jsonl_append.build_summary_line("hello summary", "boundary-1", "anchor-1", SESSION_METADATA)
    assert line["type"] == "user"
    assert line["parentUuid"] == "boundary-1"
    assert line["uuid"] == "anchor-1"
    assert line["isCompactSummary"] is True
    assert line["isVisibleInTranscriptOnly"] is True
    assert line["message"] == {"role": "user", "content": "hello summary"}
    assert line["cwd"] == "/home/x/project"


def test_rechain_preserved_tail_forms_correct_chain():
    tail = [
        {"type": "user", "uuid": "t1", "parentUuid": "old-parent", "message": {"role": "user", "content": "a"}},
        {"type": "assistant", "uuid": "t2", "parentUuid": "t1", "message": {"role": "assistant", "content": "b"}},
    ]
    rechained = jsonl_append.rechain_preserved_tail(tail, "anchor-1")
    assert rechained[0]["parentUuid"] == "anchor-1"
    assert rechained[0]["uuid"] == "t1"
    assert rechained[1]["parentUuid"] == "t1"
    assert rechained[1]["uuid"] == "t2"
    # untouched fields preserved
    assert rechained[0]["message"] == {"role": "user", "content": "a"}


def test_rechain_preserved_tail_empty():
    assert jsonl_append.rechain_preserved_tail([], "anchor-1") == []


def _write_jsonl(path, lines):
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_append_compaction_writes_valid_sequence(tmp_path):
    session_path = tmp_path / "session.jsonl"
    original_lines = [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"role": "user", "content": "hi"}, **SESSION_METADATA},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "message": {"role": "assistant", "content": "hello"}, **SESSION_METADATA},
    ]
    _write_jsonl(session_path, original_lines)

    preserved_tail = [
        {"type": "user", "uuid": "u2", "parentUuid": "a1", "message": {"role": "user", "content": "more"}},
    ]

    result = jsonl_append.append_compaction(
        session_path=session_path,
        summary_text="<summary>\n1. Primary Request:\n   test\n</summary>",
        preserved_tail=preserved_tail,
        trigger="manual",
        pre_tokens=500,
        post_tokens=50,
        duration_ms=1234,
    )

    all_lines = [json.loads(l) for l in session_path.read_text().splitlines() if l.strip()]
    assert len(all_lines) == len(original_lines) + 3  # boundary + summary + 1 preserved

    # original lines untouched
    assert all_lines[0] == original_lines[0]
    assert all_lines[1] == original_lines[1]

    boundary, summary, rechained_u2 = all_lines[2], all_lines[3], all_lines[4]
    assert boundary["subtype"] == "compact_boundary"
    assert boundary["logicalParentUuid"] == "a1"
    assert boundary["uuid"] == result.boundary_uuid
    assert summary["parentUuid"] == boundary["uuid"]
    assert summary["uuid"] == result.anchor_uuid
    assert summary["isCompactSummary"] is True
    assert "Summary:" in summary["message"]["content"]
    assert rechained_u2["uuid"] == "u2"
    assert rechained_u2["parentUuid"] == result.anchor_uuid
    assert result.lines_appended == 3
    assert result.logical_parent_uuid == "a1"


def test_append_compaction_skips_trailing_control_lines_with_no_uuid(tmp_path):
    # Real bug, found via a live throwaway session: the raw file's literal
    # last line is often a control-type entry (e.g. "last-prompt", "mode",
    # "queue-operation") with no uuid at all. append_compaction must anchor
    # to the last line that actually has one, not raw_lines[-1] blindly.
    session_path = tmp_path / "session.jsonl"
    original_lines = [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"role": "user", "content": "hi"}, **SESSION_METADATA},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "message": {"role": "assistant", "content": "hello"}, **SESSION_METADATA},
        {"type": "last-prompt"},
        {"type": "mode", "mode": "normal"},
    ]
    _write_jsonl(session_path, original_lines)

    result = jsonl_append.append_compaction(
        session_path=session_path, summary_text="<summary>\ntest\n</summary>",
        preserved_tail=[], trigger="manual", pre_tokens=10, post_tokens=5, duration_ms=1,
    )
    assert result.logical_parent_uuid == "a1"
    all_lines = [json.loads(l) for l in session_path.read_text().splitlines() if l.strip()]
    boundary = all_lines[len(original_lines)]
    assert boundary["logicalParentUuid"] == "a1"


def test_append_compaction_respects_anchor_override(tmp_path):
    session_path = tmp_path / "session.jsonl"
    original_lines = [
        {"type": "user", "uuid": "u1", "parentUuid": None, "message": {"role": "user", "content": "pre-clear"}, **SESSION_METADATA},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "message": {"role": "assistant", "content": "reply"}, **SESSION_METADATA},
        {"type": "user", "uuid": "u2", "parentUuid": "a1", "message": {"role": "user", "content": "post-clear"}, **SESSION_METADATA},
    ]
    _write_jsonl(session_path, original_lines)

    result = jsonl_append.append_compaction(
        session_path=session_path,
        summary_text="<summary>\ntest\n</summary>",
        preserved_tail=[],
        trigger="continue_after_clear",
        pre_tokens=10, post_tokens=5, duration_ms=1,
        logical_parent_uuid_override="a1",
    )

    assert result.logical_parent_uuid == "a1"  # the pre-clear span's last line, not u2 (the file's true last line)
    all_lines = [json.loads(l) for l in session_path.read_text().splitlines() if l.strip()]
    boundary = all_lines[len(original_lines)]
    assert boundary["logicalParentUuid"] == "a1"
    # session-level metadata still comes from the file's true last line regardless of the override
    assert boundary["sessionId"] == SESSION_METADATA["sessionId"]


def test_append_compaction_raises_on_empty_session(tmp_path):
    session_path = tmp_path / "empty.jsonl"
    session_path.write_text("")
    with pytest.raises(ValueError):
        jsonl_append.append_compaction(
            session_path=session_path, summary_text="x", preserved_tail=[],
            trigger="manual", pre_tokens=1, post_tokens=1, duration_ms=1,
        )
