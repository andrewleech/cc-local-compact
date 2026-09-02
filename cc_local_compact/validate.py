"""Structural sanity-check on a raw model response, used to gate the
model-fallback mechanism (fallback.py).

Not a semantic or quality judgment -- a cheap, deterministic check for
"did this response even attempt the shape the prompt asked for." Designed
directly from four independently-reproduced real failures against
qwen3.5-9b on self-referential/meta-technical content (a conversation
about building a compaction tool, containing raw tool_result JSON about
compaction itself), none of which qwen3.8-27b produced on the same
content:

  1. "I see what happened -- the model backend is a summarization-focused
     agent designed to produce the `" (cut off mid-word, no tags at all)
  2. "" (empty)
  3. "I apologize for the confusion... Let me fix the `_classify_error`
     function..." (no tags, model tried to continue the ORIGINAL task
     instead of compacting)
  4. "<c irritation>\\nThis transcript shows a saved \"fake compact\"
     session obfuscated..." (anomalous framing, hallucinated the data was
     fake, before eventually reaching a real <analysis> tag)

All four are missing what prompts.INSTRUCTIONS_BODY explicitly demands: a
<summary> tag wrapping the 9 numbered sections ("1. Primary Request and
Intent:" through "9. Optional Next Step:"). That's the one structural
signal common to every observed failure, so it's what this checks.
"""

import re

from . import response


MIN_SUMMARY_CHARS = 200
"""A real compacted summary of anything non-trivial runs well past this;
a garbled/short failure response typically doesn't. Deliberately low
relative to a real summary's expected length, since this is a floor
against near-empty responses, not an attempt to judge thoroughness."""

MIN_SECTION_HEADERS = 5
"""Of the 9 numbered sections the prompt asks for (see
prompts.INSTRUCTIONS_BODY), require at least this many numbered-list
headers to appear. Not all 9, since a model can reasonably compress or
reorder sections for a short conversation; the four real failures
collected so far had 0, so this threshold has wide margin against them
without being so strict it rejects legitimate variance."""

_SECTION_HEADER_RE = re.compile(r"(?m)^\s*\d+\.\s+\S")


def is_well_formed_summary(raw_text: str | None) -> bool:
    """True if `raw_text` (the model's raw response, before clean_summary)
    looks like a genuine attempt at the requested <analysis>/<summary>
    structure. False triggers a fallback in fallback.py."""
    if not raw_text:
        return False
    if "<summary>" not in raw_text:
        return False
    cleaned = response.clean_summary(raw_text)
    if len(cleaned) < MIN_SUMMARY_CHARS:
        return False
    if len(_SECTION_HEADER_RE.findall(cleaned)) < MIN_SECTION_HEADERS:
        return False
    return True
