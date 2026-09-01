"""Response post-processing, ported from Claude Code's FBn and Vq (chunk-
4scc8rka.js, module 356 of the 2.1.252 build -- see
docs/compact-architecture.md in the cc-patcher repo, "Post-processing and
final message assembly").

Vq's wrapper text is written to be injected as a live synthetic user turn
immediately before the app's next assistant response -- its imperative
wording ("resume directly, do not acknowledge the summary") is written for
that context. This module keeps it verbatim as `build_resume_preamble` for
possible reuse in a later JSONL-append phase, but markdown_out.py does not
use it as the output's primary framing.
"""

import re


_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_BLANK_LINES_RE = re.compile(r"\n\n+")


def clean_summary(raw_text: str) -> str:
    """Port of FBn: strip <analysis>, extract <summary> prefixed
    'Summary:\\n', collapse repeated blank lines."""
    text = _ANALYSIS_RE.sub("", raw_text, count=1)
    match = _SUMMARY_RE.search(text)
    if match:
        body = (match.group(1) or "").strip()
        text = _SUMMARY_RE.sub(f"Summary:\n{body}", text, count=1)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def build_resume_preamble(
    raw_summary_text: str,
    transcript_path: str | None = None,
    recent_messages_preserved: bool = False,
    repl_state_cleared: bool = False,
    suppress_follow_up_questions: bool = True,
) -> str:
    """Port of Vq. Cleans `raw_summary_text` internally, mirroring the
    source (which calls FBn on the raw model output, not a pre-cleaned
    string)."""
    text = (
        "This session is being continued from a previous conversation that "
        "ran out of context. The summary below covers the earlier portion "
        f"of the conversation.\n\n{clean_summary(raw_summary_text)}"
    )
    if transcript_path:
        text += (
            "\n\nIf you need specific details from before compaction (like "
            "exact code snippets, error messages, or content you "
            f"generated), read the full transcript at: {transcript_path}"
        )
    if recent_messages_preserved:
        text += "\n\nRecent messages are preserved verbatim."
    if repl_state_cleared:
        text += (
            "\n\nYour REPL VM state has been cleared as part of this "
            "compaction. Variables defined in REPL calls before this point "
            "are no longer accessible — redefine any you still need."
        )
    if suppress_follow_up_questions:
        text += (
            "\nContinue the conversation from where it left off without "
            "asking the user any further questions. Resume directly — do "
            "not acknowledge the summary, do not recap what was happening, "
            'do not preface with "I\'ll continue" or similar. Pick up the '
            "last task as if the break never happened."
        )
    return text
