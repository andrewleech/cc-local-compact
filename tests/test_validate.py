from cc_local_compact import validate


GOOD_SUMMARY = """<analysis>
Some reasoning about the conversation.
</analysis>

<summary>
1. Primary Request and Intent:
   The user asked for X.

2. Key Technical Concepts:
   - Concept A
   - Concept B

3. Files and Code Sections:
   - file.py

4. Errors and fixes:
   - None.

5. Problem Solving:
   Solved the thing.

6. All user messages:
   - "Do X."

7. Pending Tasks:
   - None.

8. Current Work:
   Finishing up.

9. Optional Next Step:
   Ship it.
</summary>"""


# Verbatim (truncated where noted) real failures collected against
# qwen3.5-9b on this project's own 488K-token benchmark -- see
# validate.py's module docstring.
FAILURE_1_NO_TAGS_TRUNCATED = (
    "I see what happened — the model backend is a summarization-focused "
    "agent designed to produce the `"
)
FAILURE_2_EMPTY = ""
FAILURE_3_WRONG_TASK = (
    "I apologize for the confusion – that prompt text accidentally slipped "
    "into my shell command. Let me fix the `_classify_error` function in "
    "`client.py` to properly detect and parse the backend's "
    "`exceed_context_size_error` shape."
)
FAILURE_4_HALLUCINATED_PREAMBLE = (
    '<c irritation>\nThis transcript shows a saved "fake compact" session '
    "obfuscated in the user's .transport.trace, with fake metadata, no real "
    'session context, but still requesting a /compact without the actual '
    "session data."
)


def test_well_formed_summary_passes():
    assert validate.is_well_formed_summary(GOOD_SUMMARY) is True


def test_none_fails():
    assert validate.is_well_formed_summary(None) is False


def test_real_failure_1_no_tags_truncated_fails():
    assert validate.is_well_formed_summary(FAILURE_1_NO_TAGS_TRUNCATED) is False


def test_real_failure_2_empty_fails():
    assert validate.is_well_formed_summary(FAILURE_2_EMPTY) is False


def test_real_failure_3_wrong_task_fails():
    assert validate.is_well_formed_summary(FAILURE_3_WRONG_TASK) is False


def test_real_failure_4_hallucinated_preamble_fails():
    # Only a partial capture (response was cut short in the original
    # diagnostic), but it's a valid fixture on its own: no <summary> tag
    # appears anywhere in what was captured.
    assert validate.is_well_formed_summary(FAILURE_4_HALLUCINATED_PREAMBLE) is False


def test_missing_summary_tag_fails_even_if_long():
    text = "x" * 1000
    assert validate.is_well_formed_summary(text) is False


def test_too_short_fails_even_with_summary_tag():
    text = "<summary>too short</summary>"
    assert validate.is_well_formed_summary(text) is False


def test_too_few_section_headers_fails():
    text = "<summary>\n" + ("padding " * 100) + "\n1. Only one section here.\n</summary>"
    assert validate.is_well_formed_summary(text) is False
