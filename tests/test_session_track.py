from pathlib import Path

from cc_local_compact import session_track


def test_predecessor_session_none_when_never_tracked(tmp_path):
    # deterministic ppid_of (no real ancestry) -- nothing anywhere is tracked
    assert session_track.predecessor_session(1234, "s-new", state_dir=tmp_path, ppid_of=lambda pid: None) is None


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


def _fake_ancestry(chain: dict[int, int | None]):
    """chain: {pid: parent_pid_or_None} -- a fake ppid_of for deterministic tests."""
    return lambda pid: chain.get(pid)


def test_find_tracked_pid_returns_start_pid_when_it_already_has_state(tmp_path):
    session_track.record_turn(7000, "s-a", "/a.jsonl", "/cwd", state_dir=tmp_path)
    assert session_track.find_tracked_pid(7000, state_dir=tmp_path) == 7000


def test_find_tracked_pid_walks_up_to_find_tracked_ancestor(tmp_path):
    # simulates the real observed bug: the hook's own os.getppid() (5000)
    # has no state, but its grandparent (7000) does -- Stop, running under
    # a different (correct) parent resolution, already tracked 7000.
    session_track.record_turn(7000, "s-a", "/a.jsonl", "/cwd", state_dir=tmp_path)
    ppid_of = _fake_ancestry({5000: 6000, 6000: 7000, 7000: 100})
    assert session_track.find_tracked_pid(5000, state_dir=tmp_path, ppid_of=ppid_of) == 7000


def test_find_tracked_pid_none_when_no_ancestor_tracked(tmp_path):
    ppid_of = _fake_ancestry({5000: 6000, 6000: 7000, 7000: 100})
    assert session_track.find_tracked_pid(5000, state_dir=tmp_path, ppid_of=ppid_of) is None


def test_find_tracked_pid_respects_max_levels(tmp_path):
    # tracked state exists, but only beyond max_levels -- must not be found
    session_track.record_turn(9000, "s-a", "/a.jsonl", "/cwd", state_dir=tmp_path)
    long_chain = {i: i + 1 for i in range(1000, 1000 + 15)}
    long_chain[1015] = 9000
    ppid_of = _fake_ancestry(long_chain)
    assert session_track.find_tracked_pid(1000, state_dir=tmp_path, max_levels=3, ppid_of=ppid_of) is None
    assert session_track.find_tracked_pid(1000, state_dir=tmp_path, max_levels=20, ppid_of=ppid_of) == 9000


def test_predecessor_session_walks_ancestry_when_direct_pid_has_no_state(tmp_path):
    session_track.record_turn(7000, "s-old", "/old.jsonl", "/cwd", state_dir=tmp_path)
    ppid_of = _fake_ancestry({5000: 6000, 6000: 7000, 7000: 100})

    predecessor = session_track.predecessor_session(5000, "s-new", state_dir=tmp_path, ppid_of=ppid_of)
    assert predecessor == {"session_id": "s-old", "transcript_path": "/old.jsonl", "cwd": "/cwd"}
