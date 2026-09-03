from pathlib import Path

from cc_local_compact import session_track


def test_predecessor_session_none_when_never_tracked(tmp_path):
    assert session_track.predecessor_session(1234, "s-new", state_dir=tmp_path) is None


def test_predecessor_is_current_when_no_new_turn_completed_yet(tmp_path):
    # /clear happened, /remind is called before any turn has completed in
    # the new session -- record_turn was never called with the new
    # session_id, so `current` still holds the pre-clear session.
    session_track.record_turn(1234, "s-old", "/path/old.jsonl", "/cwd", state_dir=tmp_path)
    predecessor = session_track.predecessor_session(1234, "s-new", state_dir=tmp_path)
    assert predecessor == {"session_id": "s-old", "transcript_path": "/path/old.jsonl", "cwd": "/cwd"}


def test_predecessor_is_previous_after_a_turn_completed_in_the_new_session(tmp_path):
    session_track.record_turn(1234, "s-old", "/path/old.jsonl", "/cwd", state_dir=tmp_path)
    # a turn completes in the new (post-clear) session -- Stop fires, shifting current->previous
    session_track.record_turn(1234, "s-new", "/path/new.jsonl", "/cwd", state_dir=tmp_path)
    predecessor = session_track.predecessor_session(1234, "s-new", state_dir=tmp_path)
    assert predecessor == {"session_id": "s-old", "transcript_path": "/path/old.jsonl", "cwd": "/cwd"}


def test_repeated_turns_in_the_same_session_do_not_disturb_previous(tmp_path):
    session_track.record_turn(1234, "s-old", "/path/old.jsonl", "/cwd", state_dir=tmp_path)
    session_track.record_turn(1234, "s-new", "/path/new.jsonl", "/cwd", state_dir=tmp_path)
    # several more ordinary turns in s-new
    for _ in range(5):
        session_track.record_turn(1234, "s-new", "/path/new.jsonl", "/cwd", state_dir=tmp_path)
    predecessor = session_track.predecessor_session(1234, "s-new", state_dir=tmp_path)
    assert predecessor["session_id"] == "s-old"


def test_second_clear_in_the_same_window_replaces_previous_correctly(tmp_path):
    session_track.record_turn(1234, "s-a", "/a.jsonl", "/cwd", state_dir=tmp_path)
    session_track.record_turn(1234, "s-b", "/b.jsonl", "/cwd", state_dir=tmp_path)  # first /clear -> s-b
    # /remind never called; user just clears again
    session_track.record_turn(1234, "s-c", "/c.jsonl", "/cwd", state_dir=tmp_path)  # second /clear -> s-c
    predecessor = session_track.predecessor_session(1234, "s-c", state_dir=tmp_path)
    # previous is s-b (the one immediately before s-c), not s-a
    assert predecessor["session_id"] == "s-b"


def test_different_pids_are_fully_independent(tmp_path):
    session_track.record_turn(111, "window1-old", "/w1old.jsonl", "/cwd", state_dir=tmp_path)
    session_track.record_turn(222, "window2-old", "/w2old.jsonl", "/cwd", state_dir=tmp_path)

    p1 = session_track.predecessor_session(111, "window1-new", state_dir=tmp_path)
    p2 = session_track.predecessor_session(222, "window2-new", state_dir=tmp_path)
    assert p1["session_id"] == "window1-old"
    assert p2["session_id"] == "window2-old"


def test_state_dir_is_created_with_restrictive_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    d = session_track._state_dir()
    assert d.is_dir()
    assert oct(d.stat().st_mode)[-3:] == "700"


def test_record_turn_skips_write_when_nothing_changed(tmp_path, monkeypatch):
    write_calls = []
    real_write_text = Path.write_text

    def spy_write_text(self, *args, **kwargs):
        write_calls.append(self)
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)

    session_track.record_turn(1234, "s-a", "/a.jsonl", "/cwd", state_dir=tmp_path)
    assert len(write_calls) == 1  # first-ever write for this pid always happens

    # repeating the exact same turn info -- nothing actually changed
    session_track.record_turn(1234, "s-a", "/a.jsonl", "/cwd", state_dir=tmp_path)
    assert len(write_calls) == 1  # no second write

    predecessor = session_track.predecessor_session(1234, "s-new", state_dir=tmp_path)
    assert predecessor["session_id"] == "s-a"


def test_record_turn_writes_when_session_id_changes(tmp_path, monkeypatch):
    write_calls = []
    real_write_text = Path.write_text

    def spy_write_text(self, *args, **kwargs):
        write_calls.append(self)
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)

    session_track.record_turn(1234, "s-a", "/a.jsonl", "/cwd", state_dir=tmp_path)
    session_track.record_turn(1234, "s-a", "/a.jsonl", "/cwd", state_dir=tmp_path)  # no-op
    session_track.record_turn(1234, "s-b", "/b.jsonl", "/cwd", state_dir=tmp_path)  # real change
    assert len(write_calls) == 2
