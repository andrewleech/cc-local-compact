"""Response post-processing, ported from Claude Code's FBn and Vq (chunk-
4scc8rka.js, module 356 of the 2.1.252 build; see
docs/compact-architecture.md in the cc-patcher repo, "Post-processing and
final message assembly").

Vq's wrapper text is written to be injected as a live synthetic user turn
immediately before the app's next assistant response; its imperative
wording ("resume directly, do not acknowledge the summary") is written for
that context. This module keeps it verbatim as `build_resume_preamble` for
possible reuse in a later JSONL-append phase, but markdown_out.py does not
use it as the output's primary framing.
"""

import re


_ANALYSIS_RE = re.compile(r"<analysis>.*?(?:</analysis>|\Z)", re.DOTALL)
_SUMMARY_OPEN_RE = re.compile(r"<summary>")
_SUMMARY_CLOSE_RE = re.compile(r"</summary>")
_BLANK_LINES_RE = re.compile(r"\n\n+")


def clean_summary(raw_text: str) -> str:
    """Port of FBn: strip <analysis>, extract <summary> prefixed
    'Summary:\\n', collapse repeated blank lines.

    DEVIATION FROM SOURCE (FBn): a local model can run out of max_tokens
    before closing its <summary> tag, observed against qwen3.8-27b on a
    large real transcript, where the response was cut off mid-sentence
    with no closing tag at all. FBn's regex requires both tags and would
    leave the literal "<summary>" marker in the output verbatim in that
    case. Here, an opening <summary> with no matching close is treated as
    running to the end of the string rather than left unprocessed; same
    treatment applied to <analysis> for the symmetric case."""
    text = _ANALYSIS_RE.sub("", raw_text, count=1)

    open_match = _SUMMARY_OPEN_RE.search(text)
    if open_match:
        close_match = _SUMMARY_CLOSE_RE.search(text, open_match.end())
        body_end = close_match.start() if close_match else len(text)
        tail_start = close_match.end() if close_match else len(text)
        body = text[open_match.end():body_end].strip()
        text = text[:open_match.start()] + f"Summary:\n{body}" + text[tail_start:]

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


def build_remind_preamble(raw_summary_text: str, transcript_path: str | None = None) -> str:
    """Framing for /remind's injected additionalContext; deliberately
    NOT build_resume_preamble's "resume directly, don't ask questions"
    wording, even though /remind's summary comes from the same
    compaction pipeline. That wording is written for the real app's own
    /compact: injected as a live synthetic turn immediately before the
    SAME ongoing task's next response, with a human actively watching who
    hit a genuine context limit mid-flow; continuing automatically is
    the right call there. /remind's situation differs in two ways that
    make that framing actively dangerous here instead: /clear is a
    deliberate human action, not an accidental limit, so there's no
    guarantee the user wants the same task continued at all, let alone
    immediately; and because /remind's hook sets suppressOriginalPrompt,
    this injected text is the ONLY thing the model sees on this turn;
    there is no accompanying human message to interrupt a bad plan before
    it executes. Confirmed live: build_resume_preamble's wording made a
    real run launch straight into flashing physical hardware with zero
    human confirmation on the very turn that injected it. This framing
    explicitly tells the model to wait rather than act, matching this
    project's own established precedent in multipass.py, which already
    avoids build_resume_preamble's framing for intermediate pass
    summaries for the same underlying reason (wrong implication for
    something that isn't a live task to resume)."""
    text = (
        "The conversation from before your last /clear has been recovered "
        "and summarized below, for background only. This is not a request "
        "to continue or resume any task, and no user message accompanies "
        "it (this text was injected by a hook, not typed by the user). Do "
        "not run commands, edit files, call tools, or otherwise act on "
        "this summary. Read it for context, then wait for the user's next "
        f"message before doing anything.\n\n{clean_summary(raw_summary_text)}"
    )
    if transcript_path:
        text += (
            "\n\nIf you need specific details from before the /clear (like "
            "exact code snippets, error messages, or content generated "
            f"earlier), read the full transcript at: {transcript_path}"
        )
    return text
