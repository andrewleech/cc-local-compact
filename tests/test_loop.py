from cc_local_compact import loop
from cc_local_compact.client import AttemptResult


def _line(type_, uuid, content="x", message_id=None):
    message = {"role": type_, "content": content}
    if message_id is not None:
        message["id"] = message_id
    return {"type": type_, "uuid": uuid, "message": message}


def _build_groups(n):
    """n user/assistant pairs group into n+1 groups: a leading group with
    just the first user message (no assistant -- group boundaries fall at
    each assistant message, so nothing precedes it into the same group),
    n-1 middle [assistant, next_user] groups, and a trailing single-
    assistant group with nothing after it."""
    lines = []
    for i in range(n):
        lines.append(_line("user", f"u{i}", content=f"user turn {i}"))
        lines.append(_line("assistant", f"a{i}", content=f"assistant turn {i}", message_id=f"m{i}"))
    return lines


# -- _accumulate_backward / _backoff_step: direct arithmetic tests --------

def test_accumulate_backward_gap_guided():
    per_group_tokens = [100, 100, 100, 100, 100]
    assert loop._accumulate_backward(per_group_tokens, upto_exclusive=4, gap=150) == 2


def test_accumulate_backward_falls_back_to_half():
    per_group_tokens = [100, 100, 100, 100, 100]
    assert loop._accumulate_backward(per_group_tokens, upto_exclusive=4, gap=1_000_000) == 2  # floor(4/2)


def test_backoff_step_none_gap_returns_one():
    assert loop._backoff_step(None, [100, 100], upto_exclusive=2) == 1


def test_backoff_step_delegates_to_accumulate_backward():
    per_group_tokens = [10, 10, 10]
    assert loop._backoff_step(15, per_group_tokens, upto_exclusive=3) == loop._accumulate_backward(
        per_group_tokens, 3, 15,
    )


# -- run_loop: control-flow tests, budget large enough to skip seeding ----

def test_too_few_groups():
    def unreachable(*_args):
        raise AssertionError("summarize_fn should not be called")

    result = loop.run_loop([], unreachable, loop.LoopConfig(context_budget=100_000))
    assert result.ok is False
    assert result.reason == "too_few_groups"
    assert result.attempts == 0


def test_leading_group_without_assistant_bails_as_too_few_groups():
    # _build_groups(1) -> groups=[[u0],[a0]], total_groups=2. The first
    # (only, at groups_preserved=1) summarize slice is just the leading
    # group, which never contains an assistant message by construction --
    # this is the real, easily-reachable way ZZ's "no assistant in the
    # summarize set" bail-out fires, not a purely defensive dead branch.
    def unreachable(*_args):
        raise AssertionError("summarize_fn should not be called")

    result = loop.run_loop(_build_groups(1), unreachable, loop.LoopConfig(context_budget=100_000))
    assert result.ok is False
    assert result.reason == "too_few_groups"
    assert result.attempts == 0


def test_success_first_attempt_no_seeding():
    lines = _build_groups(4)
    calls = []

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        calls.append((len(summarize_set), instructions, strip_media_flag))
        return AttemptResult(ok=True, summary_text="Summary:\ndone", total_usage=None)

    result = loop.run_loop(
        lines, fake_summarize, loop.LoopConfig(context_budget=1_000_000), custom_instructions="focus on X",
    )
    assert result.ok is True
    assert result.attempts == 1
    assert result.groups_preserved == 1
    assert len(calls) == 1
    assert calls[0][1] == "focus on X"
    assert calls[0][2] is False


def test_prompt_too_long_then_success_grows_preserved():
    # _build_groups(5) -> total_groups=6. First attempt summarizes groups
    # [:5] (9 lines: u0,a0,u1,a1,u2,a2,u3,a3,u4); token_gap=1 grows
    # groups_preserved from 1 to 2, so the retry summarizes groups[:4]
    # (7 lines: u0,a0,u1,a1,u2,a2,u3).
    lines = _build_groups(5)
    calls = []
    outcomes = [
        AttemptResult(ok=False, reason="prompt_too_long", token_gap=1),
        AttemptResult(ok=True, summary_text="Summary:\ndone", total_usage=None),
    ]

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        calls.append(len(summarize_set))
        return outcomes.pop(0)

    result = loop.run_loop(lines, fake_summarize, loop.LoopConfig(context_budget=1_000_000))
    assert result.ok is True
    assert result.attempts == 2
    assert result.groups_preserved == 2
    assert calls == [9, 7]


def test_media_too_large_strips_once_then_succeeds():
    lines = _build_groups(3)
    calls = []
    outcomes = [
        AttemptResult(ok=False, reason="media_too_large"),
        AttemptResult(ok=True, summary_text="Summary:\ndone", total_usage=None),
    ]

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        calls.append(strip_media_flag)
        return outcomes.pop(0)

    result = loop.run_loop(lines, fake_summarize, loop.LoopConfig(context_budget=1_000_000))
    assert result.ok is True
    assert calls == [False, True]
    # the stripped-media retry doesn't count as a separate attempt
    assert result.attempts == 1


def test_media_too_large_twice_returns_unstrippable():
    outcomes = [
        AttemptResult(ok=False, reason="media_too_large"),
        AttemptResult(ok=False, reason="media_too_large"),
    ]

    def fake_summarize(*_args):
        return outcomes.pop(0)

    result = loop.run_loop(_build_groups(3), fake_summarize, loop.LoopConfig(context_budget=1_000_000))
    assert result.ok is False
    assert result.reason == "media_unstrippable"


def test_error_returns_immediately():
    def fake_summarize(*_args):
        return AttemptResult(ok=False, reason="error", detail="boom")

    result = loop.run_loop(_build_groups(3), fake_summarize, loop.LoopConfig(context_budget=1_000_000))
    assert result.ok is False
    assert result.reason == "error"
    assert result.detail == "boom"
    assert result.attempts == 1


def test_aborted_returns_immediately():
    def fake_summarize(*_args):
        return AttemptResult(ok=False, reason="aborted")

    result = loop.run_loop(_build_groups(3), fake_summarize, loop.LoopConfig(context_budget=1_000_000))
    assert result.ok is False
    assert result.reason == "aborted"


def test_proactive_seeding_reduces_first_summarize_set():
    lines = _build_groups(5)
    calls = []

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        calls.append(len(summarize_set))
        return AttemptResult(ok=True, summary_text="Summary:\ndone", total_usage=None)

    # A near-zero budget forces seeding before the first call ever happens.
    config = loop.LoopConfig(
        context_budget=1, response_max_tokens=0, prompt_overhead_tokens=0, safety_margin_pct=0.0,
    )
    result = loop.run_loop(lines, fake_summarize, config)
    assert result.ok is True
    assert result.groups_preserved > 1
    assert len(calls) == 1
    assert calls[0] < 9  # smaller than the unseeded first attempt (9 lines)
