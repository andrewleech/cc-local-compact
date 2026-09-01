from cc_local_compact import grouping


def _user(uuid, content="u"):
    return {"type": "user", "uuid": uuid, "message": {"role": "user", "content": content}}


def _assistant(uuid, message_id, content="a", virtual=False, resumed=False):
    line = {"type": "assistant", "uuid": uuid, "message": {"role": "assistant", "id": message_id, "content": content}}
    if virtual:
        line["isVirtual"] = True
    if resumed:
        line["resumedFromIncompleteThinking"] = True
    return line


def test_basic_two_turn_grouping():
    # A group boundary falls exactly at each assistant message: whatever
    # accumulated before it closes out as the previous group, and the
    # assistant message itself opens the next accumulating group. So the
    # leading user message before the first assistant response forms its
    # own (assistant-less) group, and the trailing assistant response with
    # nothing after it forms its own single-line group.
    lines = [_user("u1"), _assistant("a1", "m1"), _user("u2"), _assistant("a2", "m2")]
    groups = grouping.group_messages(lines)
    assert len(groups) == 3
    assert [line["uuid"] for line in groups[0]] == ["u1"]
    assert [line["uuid"] for line in groups[1]] == ["a1", "u2"]
    assert [line["uuid"] for line in groups[2]] == ["a2"]


def test_progress_lines_dropped():
    lines = [_user("u1"), {"type": "progress", "uuid": "p1"}, _assistant("a1", "m1")]
    groups = grouping.group_messages(lines)
    assert len(groups) == 2
    all_uuids = [line["uuid"] for group in groups for line in group]
    assert "p1" not in all_uuids
    assert [line["uuid"] for line in groups[0]] == ["u1"]
    assert [line["uuid"] for line in groups[1]] == ["a1"]


def test_virtual_messages_join_current_group():
    virtual_user = _user("uv", content="virtual")
    virtual_user["isVirtual"] = True
    lines = [_user("u1"), _assistant("a1", "m1"), virtual_user, _assistant("av", "m2", virtual=True)]
    groups = grouping.group_messages(lines)
    assert len(groups) == 2
    assert [line["uuid"] for line in groups[0]] == ["u1"]
    assert [line["uuid"] for line in groups[1]] == ["a1", "uv", "av"]


def test_resumed_from_incomplete_thinking_does_not_start_new_group():
    lines = [
        _user("u1"),
        _assistant("a1", "m1"),
        _assistant("a1b", "m2", resumed=True),
        _user("u2"),
        _assistant("a2", "m3"),
    ]
    groups = grouping.group_messages(lines)
    assert len(groups) == 3
    assert [line["uuid"] for line in groups[0]] == ["u1"]
    assert [line["uuid"] for line in groups[1]] == ["a1", "a1b", "u2"]
    assert [line["uuid"] for line in groups[2]] == ["a2"]


def test_empty_input():
    assert grouping.group_messages([]) == []
