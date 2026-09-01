"""Summarize/preserve loop, ported from Claude Code's ZZ/_Ct/$Bn (chunk-
4scc8rka.js, module 356 of the 2.1.252 build -- see
docs/compact-architecture.md in the cc-patcher repo, "Adaptive
summarize/preserve loop").

Adapted for a KNOWN local-model context budget, since the real app instead
discovers Claude's much larger limit reactively via live API errors:

- The initial summarize/preserve split is ALWAYS proactively seeded against
  `context_budget` (the real app only does this on its auto-trigger path,
  when it already has an initialTokenGap estimate) -- a small local model
  will hit "prompt too long" far more often than the real app ever does
  against Claude's limit, so skipping the doomed first attempt matters more
  here.
- The reactive backoff is kept as a correctness safety net for when the
  chars/4 estimate is wrong.
- DEVIATION FROM SOURCE: the real app trusts Anthropic's own structured
  actualTokens/limitTokens error fields to size the backoff step. CONFIRMED
  against the live backend (llama-swap fronting llama.cpp on titan:8080,
  2026-09-01): it does not emit that shape at all -- it emits its own
  {"type": "exceed_context_size_error", "n_prompt_tokens", "n_ctx"} shape,
  which client.py's _classify_error now parses first, ahead of the
  Anthropic-shaped check. That gives an exact token_gap straight from the
  server on every real overflow observed so far, so this module's self-
  computed gap fallback (this module's own chars/4 estimate of the request
  just sent, minus the usable budget, rather than the source's dumb
  step:1) has not actually been exercised against this backend -- it
  remains in place only for a backend that emits neither known shape.
"""

import dataclasses
from typing import Callable

from . import grouping, tokens
from .client import AttemptResult


@dataclasses.dataclass(frozen=True)
class LoopConfig:
    context_budget: int
    response_max_tokens: int | None = None
    """Defaults to config._default_response_max_tokens(context_budget) if
    not given -- a fraction of context_budget, floored, rather than one
    small fixed constant. See config.py's RESPONSE_TOKENS_FRACTION/FLOOR
    docstring for why."""
    prompt_overhead_tokens: int = 1200
    safety_margin_pct: float = 0.10

    def __post_init__(self):
        if self.response_max_tokens is None:
            from . import config as _config
            object.__setattr__(
                self, "response_max_tokens",
                _config._default_response_max_tokens(self.context_budget),
            )

    def usable_budget(self) -> int:
        return int(self.context_budget * (1 - self.safety_margin_pct)) - self.response_max_tokens


@dataclasses.dataclass(frozen=True)
class LoopResult:
    ok: bool
    reason: str | None = None
    attempts: int = 0
    total_groups: int = 0
    groups_preserved: int = 0
    summary_text: str | None = None
    total_usage: dict | None = None
    messages_to_preserve: list[dict] | None = None
    detail: str | None = None


SummarizeFn = Callable[[list[dict], "str | None", bool], AttemptResult]
"""(summarize_set, custom_instructions, strip_media) -> AttemptResult.
Injected so this loop's control flow is fully unit-testable without a
network call -- see tests/test_loop.py."""


def _accumulate_backward(per_group_tokens: list[int], upto_exclusive: int, gap: int) -> int:
    """Port of _Ct: walk backward from `upto_exclusive - 1`, accumulating
    per-group token estimates, stopping as soon as the accumulation meets
    or exceeds `gap`. Falls back to floor(upto_exclusive/2) if accumulating
    nearly everything still doesn't close the gap. This is a single greedy
    forward-accumulation pass, not binary search -- each call re-estimates
    fresh from the latest reported gap."""
    accumulated = 0
    count = 0
    for index in range(upto_exclusive - 1, -1, -1):
        accumulated += per_group_tokens[index]
        count += 1
        if accumulated >= gap:
            break
    if count >= upto_exclusive - 1:
        return max(1, upto_exclusive // 2)
    return count


def _backoff_step(token_gap: int | None, per_group_tokens: list[int], upto_exclusive: int) -> int:
    """Port of $Bn."""
    if token_gap is None:
        return 1
    return _accumulate_backward(per_group_tokens, upto_exclusive, token_gap)


def run_loop(
    lines: list[dict],
    summarize_fn: SummarizeFn,
    config: LoopConfig,
    custom_instructions: str | None = None,
) -> LoopResult:
    """Port of ZZ, budget-seeded (see module docstring for the deviation
    from the source's reactive-only discovery)."""
    groups = grouping.group_messages(lines)
    total_groups = len(groups)
    if total_groups < 2:
        return LoopResult(ok=False, reason="too_few_groups", attempts=0, total_groups=total_groups)

    per_group_tokens = [tokens.estimate_group_tokens(g) for g in groups]
    usable = config.usable_budget()

    groups_preserved = 1
    total_estimate = sum(per_group_tokens) + config.prompt_overhead_tokens
    if total_estimate > usable:
        seed_gap = total_estimate - usable
        step = _accumulate_backward(per_group_tokens, total_groups - 1, seed_gap)
        groups_preserved = 1 + step

    attempts = 0
    strip_media_flag = False

    while groups_preserved < total_groups:
        attempts += 1
        summarize_count = total_groups - groups_preserved
        summarize_groups = groups[:summarize_count]
        preserve_groups = groups[summarize_count:]
        summarize_set = [line for group in summarize_groups for line in group]

        if not any(line.get("type") == "assistant" for line in summarize_set):
            return LoopResult(
                ok=False,
                reason="exhausted" if attempts > 1 else "too_few_groups",
                attempts=attempts - 1,
                total_groups=total_groups,
            )

        result = summarize_fn(summarize_set, custom_instructions, strip_media_flag)

        if result.ok:
            preserve_set = [line for group in preserve_groups for line in group]
            return LoopResult(
                ok=True,
                attempts=attempts,
                total_groups=total_groups,
                groups_preserved=groups_preserved,
                summary_text=result.summary_text,
                total_usage=result.total_usage,
                messages_to_preserve=preserve_set,
            )

        if result.reason == "aborted":
            return LoopResult(ok=False, reason="aborted", attempts=attempts, total_groups=total_groups)
        if result.reason == "error":
            return LoopResult(
                ok=False, reason="error", attempts=attempts, total_groups=total_groups, detail=result.detail,
            )
        if result.reason == "media_too_large":
            if not strip_media_flag:
                strip_media_flag = True
                attempts -= 1
                continue
            return LoopResult(ok=False, reason="media_unstrippable", attempts=attempts, total_groups=total_groups)

        # prompt_too_long: grow groups_preserved and retry.
        token_gap = result.token_gap
        if token_gap is None:
            # DEVIATION from $Bn: self-computed gap fallback, see module docstring.
            sent_estimate = tokens.estimate_group_tokens(summarize_set) + config.prompt_overhead_tokens
            computed_gap = sent_estimate - usable
            token_gap = computed_gap if computed_gap > 0 else None
        step = _backoff_step(token_gap, per_group_tokens, summarize_count)
        groups_preserved += step

    return LoopResult(ok=False, reason="exhausted", attempts=attempts, total_groups=total_groups)
