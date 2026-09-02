"""Model fallback: wraps a primary and fallback SummarizeFn so a
structurally invalid response from the primary triggers a retry against
the fallback, without loop.py/multipass.py needing to know two models are
involved at all -- they only ever see one SummarizeFn.

Real motivation: qwen3.5-9b (fast, ~1/13th the wall-clock of qwen3.8-27b
on this project's own benchmark) reliably compacts most content, but was
empirically confused by dense self-referential/meta-technical content in
a way qwen3.8-27b never was, across 4 independently-reproduced failures on
the same real session (see validate.py's module docstring). That failure
was proven content-specific, not multi-pass-specific: it reproduced even
feeding the exact same content through a single standalone call with no
multi-pass framing at all. So there's no reason to distrust the fast model
everywhere -- validating its output per-call and only paying the slower
model's cost on the specific batches that actually need it gets the speed
where it's safe and the reliability where it isn't.
"""

import dataclasses

from .loop import SummarizeFn
from .validate import is_well_formed_summary


def with_fallback(
    primary: SummarizeFn,
    fallback: SummarizeFn,
    validate=is_well_formed_summary,
) -> SummarizeFn:
    """Returns a SummarizeFn that calls `primary` first. If `primary`
    fails outright, or succeeds but its raw summary_text doesn't pass
    `validate`, calls `fallback` instead and returns that result, stamped
    with used_fallback=True. Runs `primary` exactly once per call either
    way -- this is not a retry loop on the primary, it's a single
    try-then-escalate."""
    def wrapped(summarize_set, instructions, strip_media_flag):
        result = primary(summarize_set, instructions, strip_media_flag)
        if result.ok and validate(result.summary_text):
            return result
        fallback_result = fallback(summarize_set, instructions, strip_media_flag)
        return dataclasses.replace(fallback_result, used_fallback=True)
    return wrapped
