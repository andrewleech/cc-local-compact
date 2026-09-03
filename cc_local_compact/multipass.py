"""Multi-pass compaction: chains loop.run_loop calls so a session larger
than what fits in a single pass' context_budget still gets fully
compacted, rather than leaving most of it as an untouched preserved tail.

Real evidence this is needed, both from this tool and from the built-in
mechanism it replicates: a genuine Claude Code /compact on a 488K-token
session compacted it to ~14.8K tokens in one pass, preserving only 4
messages verbatim; only possible because that session's own model had a
context window large enough to hold nearly the whole thing in one
summarization call. A single-pass run of this tool against the same
session at context_budget=48000 could only summarize the oldest 26 of 138
groups, preserving the other 112 (~106K tokens) completely untouched,
not a quality problem, a per-pass input-window problem. The built-in
mechanism has the identical limitation reactively: its retry loop (ZZ)
only ever shrinks the *current* summarize batch to fit one model call: it
has no mechanism to re-process what's already been pushed into
"preserve," so a session too large for the configured model's window would
exhibit the same "small compacted head, mostly-untouched tail" result, not
genuine end-to-end compaction. There is no known fallback to a larger-
context model on a context_size failure specifically (see
docs/compact-architecture.md's unresolved-functions list); this module
is what closes that gap for a local model with a much smaller window than
Claude's.

The fix: after each pass, if what's left over (this pass' cleaned summary
plus its preserved tail) would still be too large to fit as a single
pass' own input, feed that combined residual back in as the next pass'
input and run again, continuing until the residual would fit in one
more pass, or nothing further can be compacted.
"""

import dataclasses

from . import response
from .loop import LoopConfig, SummarizeFn, run_loop
from .tokens import count_chars_over_n, estimate_group_tokens


DEFAULT_MAX_PASSES = 10
"""Safety cap against runaway iteration; each pass is a real network
round-trip taking minutes, so this bounds worst-case wall-clock time, not
expected behavior. A session needing this many passes to fit inside one
pass' budget would be a genuinely extreme case, not the common one."""


@dataclasses.dataclass(frozen=True)
class PassRecord:
    """One completed pass' bookkeeping, kept for the final report."""
    pass_number: int
    attempts: int
    total_groups: int
    groups_preserved: int
    pre_tokens: int
    post_tokens: int
    used_fallback: bool = False
    """True if this pass' result came from a fallback model rather than
    the primary; see fallback.py."""


@dataclasses.dataclass(frozen=True)
class MultiPassResult:
    ok: bool
    reason: str | None = None
    """Set when ok is False (the first pass' own failure reason), or as a
    non-fatal caveat when ok is True but convergence wasn't clean
    ("max_passes_reached" or "later_pass_failed")."""
    detail: str | None = None
    passes: tuple[PassRecord, ...] = ()
    final_summary_text: str | None = None
    """Raw (uncleaned) summary text from the LAST pass. Since each pass
    after the first summarizes (prior summary + newly-exposed older tail),
    this is the single summary that best represents the whole compacted
    history, not just the final pass' own batch."""
    final_preserved_tail: list[dict] | None = None
    final_total_usage: dict | None = None


def _residual_lines(summary_text: str, preserved_tail: list[dict]) -> list[dict]:
    """Build the next pass' input: the previous pass' cleaned summary as a
    single synthetic user-role line, followed by the previous pass'
    preserved tail verbatim. A user-role line doesn't itself start a new
    group (see grouping.py), so it naturally folds into whichever group
    the residual grouping produces first, and gets swept into that pass'
    own summarize batch if the boundary lands past it.

    DEVIATION: does NOT use response.build_resume_preamble's "This session
    is being continued... Continue the conversation from where it left
    off" framing here, even though that's the real app's own wording for
    an injected isCompactSummary message. That framing is written to tell
    an ASSISTANT to resume a live task, which is the wrong implication for
    something being fed back in as input to ANOTHER summarization pass, so
    this plain "here is prior compacted history" framing is used instead
    on principle. This choice is unrelated to the qwen3.5-9b failures
    documented in validate.py/fallback.py: the same failure reproduces
    even through a single standalone call with no synthetic preamble and
    no multi-pass framing at all, confirming it's content-specific (dense
    self-referential/meta-technical material qwen3.5-9b struggles with,
    not qwen3.8-27b), not a framing artifact. See validate.py and
    fallback.py for the actual mitigation."""
    cleaned_summary = response.clean_summary(summary_text)
    summary_line = {
        "type": "user",
        "isCompactSummary": True,
        "message": {
            "role": "user",
            "content": (
                "[The following is a summary of earlier conversation history, "
                f"already compacted in a prior pass:]\n\n{cleaned_summary}"
            ),
        },
    }
    return [summary_line, *preserved_tail]


def run_multi_pass(
    lines: list[dict],
    summarize_fn: SummarizeFn,
    config: LoopConfig,
    custom_instructions: str | None = None,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> MultiPassResult:
    """Chain run_loop passes until the residual (summary + preserved tail)
    would itself fit as a single pass' input, or no further pass can make
    progress. custom_instructions only apply to the first pass; later
    passes are summarizing this tool's own prior output, not fresh
    user-requested content, so repeating them would be misapplied."""
    usable = config.usable_budget()
    current_lines = lines
    current_instructions = custom_instructions
    passes: list[PassRecord] = []
    last_summary_text: str | None = None
    last_preserved_tail: list[dict] | None = None
    last_usage: dict | None = None

    for pass_number in range(1, max_passes + 1):
        result = run_loop(current_lines, summarize_fn, config, current_instructions)

        if not result.ok:
            if passes:
                # A later pass couldn't proceed on the residual; surface
                # what the prior pass already achieved rather than losing it.
                return MultiPassResult(
                    ok=True, reason="later_pass_failed", detail=result.detail,
                    passes=tuple(passes), final_summary_text=last_summary_text,
                    final_preserved_tail=last_preserved_tail, final_total_usage=last_usage,
                )
            return MultiPassResult(ok=False, reason=result.reason, detail=result.detail)

        preserved = result.messages_to_preserve or []
        preserved_tokens = estimate_group_tokens(preserved)
        pre_tokens = estimate_group_tokens(current_lines)
        cleaned_summary = response.clean_summary(result.summary_text)
        passes.append(PassRecord(
            pass_number=pass_number, attempts=result.attempts, total_groups=result.total_groups,
            groups_preserved=result.groups_preserved, pre_tokens=pre_tokens,
            post_tokens=count_chars_over_n(cleaned_summary) + preserved_tokens,
            used_fallback=result.used_fallback,
        ))
        last_summary_text, last_preserved_tail, last_usage = (
            result.summary_text, preserved, result.total_usage,
        )

        if preserved_tokens <= usable:
            return MultiPassResult(
                ok=True, passes=tuple(passes), final_summary_text=result.summary_text,
                final_preserved_tail=preserved, final_total_usage=result.total_usage,
            )

        current_lines = _residual_lines(result.summary_text, preserved)
        current_instructions = None

    return MultiPassResult(
        ok=True, reason="max_passes_reached", passes=tuple(passes),
        final_summary_text=last_summary_text, final_preserved_tail=last_preserved_tail,
        final_total_usage=last_usage,
    )
