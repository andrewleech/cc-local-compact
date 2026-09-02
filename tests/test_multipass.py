from cc_local_compact import multipass
from cc_local_compact.client import AttemptResult
from cc_local_compact.loop import LoopConfig


def _line(type_, uuid, content="x", message_id=None):
    message = {"role": type_, "content": content}
    if message_id is not None:
        message["id"] = message_id
    return {"type": type_, "uuid": uuid, "message": message}


def _build_groups(n, prefix=""):
    lines = []
    for i in range(n):
        lines.append(_line("user", f"{prefix}u{i}", content=f"user turn {i}" * 200))
        lines.append(_line(
            "assistant", f"{prefix}a{i}", content=f"assistant turn {i}" * 200, message_id=f"{prefix}m{i}",
        ))
    return lines


def test_single_pass_converges_when_residual_fits():
    lines = _build_groups(3)

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=True, summary_text="Summary:\ndone", total_usage=None)

    config = LoopConfig(context_budget=1_000_000)
    result = multipass.run_multi_pass(lines, fake_summarize, config)
    assert result.ok is True
    assert result.reason is None
    assert len(result.passes) == 1
    assert result.final_summary_text == "Summary:\ndone"


def test_multi_pass_chains_when_residual_too_large():
    # Small context_budget forces aggressive proactive seeding (most groups
    # preserved), and the preserved tail alone still exceeds usable_budget,
    # forcing a second pass.
    lines = _build_groups(20)
    calls = []

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        calls.append(len(summarize_set))
        return AttemptResult(ok=True, summary_text="Summary:\ndone", total_usage=None)

    config = LoopConfig(context_budget=2000, response_max_tokens=200, prompt_overhead_tokens=50, safety_margin_pct=0.0)
    result = multipass.run_multi_pass(lines, fake_summarize, config, max_passes=20)
    assert result.ok is True
    assert len(result.passes) > 1
    assert len(calls) == len(result.passes)
    # each pass' recorded pre_tokens should shrink (or at least not grow
    # unboundedly) as the residual gets progressively compacted
    assert result.passes[-1].pre_tokens <= result.passes[0].pre_tokens


def test_custom_instructions_only_applied_to_first_pass():
    lines = _build_groups(20)
    seen_instructions = []

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        seen_instructions.append(instructions)
        return AttemptResult(ok=True, summary_text="Summary:\ndone", total_usage=None)

    config = LoopConfig(context_budget=2000, response_max_tokens=200, prompt_overhead_tokens=50, safety_margin_pct=0.0)
    result = multipass.run_multi_pass(
        lines, fake_summarize, config, custom_instructions="focus on X", max_passes=20,
    )
    assert len(seen_instructions) >= 2
    assert seen_instructions[0] == "focus on X"
    assert all(instr is None for instr in seen_instructions[1:])


def test_first_pass_failure_returns_not_ok():
    def fake_summarize(*_args):
        return AttemptResult(ok=False, reason="error", detail="boom")

    result = multipass.run_multi_pass(_build_groups(3), fake_summarize, LoopConfig(context_budget=1_000_000))
    assert result.ok is False
    assert result.reason == "error"
    assert result.detail == "boom"
    assert result.passes == ()


def test_later_pass_failure_returns_prior_result():
    lines = _build_groups(20)
    outcomes = [
        AttemptResult(ok=True, summary_text="Summary:\nfirst pass", total_usage=None),
        AttemptResult(ok=False, reason="error", detail="backend died"),
    ]

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        return outcomes.pop(0)

    config = LoopConfig(context_budget=2000, response_max_tokens=200, prompt_overhead_tokens=50, safety_margin_pct=0.0)
    result = multipass.run_multi_pass(lines, fake_summarize, config, max_passes=20)
    assert result.ok is True
    assert result.reason == "later_pass_failed"
    assert result.detail == "backend died"
    assert len(result.passes) == 1
    assert result.final_summary_text == "Summary:\nfirst pass"


def test_max_passes_reached_returns_best_effort():
    lines = _build_groups(20)

    def fake_summarize(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=True, summary_text="Summary:\ndone", total_usage=None)

    config = LoopConfig(context_budget=2000, response_max_tokens=200, prompt_overhead_tokens=50, safety_margin_pct=0.0)
    result = multipass.run_multi_pass(lines, fake_summarize, config, max_passes=2)
    assert result.ok is True
    assert result.reason == "max_passes_reached"
    assert len(result.passes) == 2


def test_residual_lines_shape():
    preserved = [{"type": "user", "uuid": "p1", "message": {"role": "user", "content": "hi"}}]
    residual = multipass._residual_lines("<summary>\ndone\n</summary>", preserved)
    assert residual[0]["type"] == "user"
    assert residual[0]["isCompactSummary"] is True
    # deliberately NOT the "continue the task" resume-preamble framing --
    # see multipass.py's _residual_lines docstring for why
    assert "continued from a previous conversation" not in residual[0]["message"]["content"]
    assert "already compacted in a prior pass" in residual[0]["message"]["content"]
    assert "done" in residual[0]["message"]["content"]
    assert residual[1] == preserved[0]
