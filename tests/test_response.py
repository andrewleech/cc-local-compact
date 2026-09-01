from cc_local_compact import response


RAW = (
    "<analysis>\nsome reasoning\n</analysis>\n\n"
    "<summary>\n1. Primary Request:\n   did stuff\n</summary>"
)


def test_clean_summary_strips_analysis_and_wraps_summary():
    cleaned = response.clean_summary(RAW)
    assert "<analysis>" not in cleaned
    assert "<summary>" not in cleaned
    assert cleaned.startswith("Summary:")
    assert "1. Primary Request" in cleaned


def test_clean_summary_collapses_blank_lines():
    raw = "Summary:\n\n\n\nline one\n\n\nline two"
    cleaned = response.clean_summary(raw)
    assert "\n\n\n" not in cleaned


def test_build_resume_preamble_default_suppresses_follow_up():
    text = response.build_resume_preamble(RAW, transcript_path="/tmp/session.jsonl")
    assert "This session is being continued" in text
    assert "read the full transcript at: /tmp/session.jsonl" in text
    assert "Continue the conversation from where it left off" in text
    assert "Summary:" in text


def test_build_resume_preamble_repl_state_cleared():
    text = response.build_resume_preamble(RAW, repl_state_cleared=True)
    assert "REPL VM state has been cleared" in text


def test_build_resume_preamble_no_follow_up_suppression():
    text = response.build_resume_preamble(RAW, suppress_follow_up_questions=False)
    assert "Continue the conversation from where it left off" not in text
