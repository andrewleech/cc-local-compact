from cc_local_compact import fallback
from cc_local_compact.client import AttemptResult


GOOD_SUMMARY = (
    "<summary>\n"
    "1. Primary Request and Intent:\n   The user asked for a detailed thing to be done.\n"
    "2. Key Technical Concepts:\n   Concept A, concept B, concept C were all discussed.\n"
    "3. Files and Code Sections:\n   file.py and other_file.py were both examined closely.\n"
    "4. Errors and fixes:\n   None encountered during this particular session of work.\n"
    "5. Problem Solving:\n   The core problem was solved by careful analysis of the data.\n"
    "</summary>"
)


def test_primary_succeeds_and_valid_is_used_as_is():
    calls = []

    def primary(summarize_set, instructions, strip_media_flag):
        calls.append("primary")
        return AttemptResult(ok=True, summary_text=GOOD_SUMMARY)

    def unreachable(*_args):
        raise AssertionError("fallback should not be called")

    wrapped = fallback.with_fallback(primary, unreachable)
    result = wrapped([], None, False)
    assert result.ok is True
    assert result.used_fallback is False
    assert calls == ["primary"]


def test_primary_succeeds_but_structurally_invalid_falls_back():
    def primary(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=True, summary_text="garbled, no tags at all")

    def fallback_fn(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=True, summary_text=GOOD_SUMMARY)

    wrapped = fallback.with_fallback(primary, fallback_fn)
    result = wrapped([], None, False)
    assert result.ok is True
    assert result.used_fallback is True
    assert result.summary_text == GOOD_SUMMARY


def test_primary_fails_outright_falls_back():
    def primary(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=False, reason="error", detail="boom")

    def fallback_fn(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=True, summary_text=GOOD_SUMMARY)

    wrapped = fallback.with_fallback(primary, fallback_fn)
    result = wrapped([], None, False)
    assert result.ok is True
    assert result.used_fallback is True


def test_both_primary_and_fallback_fail_returns_fallback_failure():
    def primary(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=False, reason="error", detail="primary boom")

    def fallback_fn(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=False, reason="error", detail="fallback boom")

    wrapped = fallback.with_fallback(primary, fallback_fn)
    result = wrapped([], None, False)
    assert result.ok is False
    assert result.detail == "fallback boom"
    assert result.used_fallback is True


def test_custom_validator_is_used():
    def primary(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=True, summary_text="anything")

    def fallback_fn(summarize_set, instructions, strip_media_flag):
        return AttemptResult(ok=True, summary_text="fallback result")

    always_invalid = lambda _text: False
    wrapped = fallback.with_fallback(primary, fallback_fn, validate=always_invalid)
    result = wrapped([], None, False)
    assert result.summary_text == "fallback result"
    assert result.used_fallback is True


def test_arguments_forwarded_to_primary():
    seen = {}

    def primary(summarize_set, instructions, strip_media_flag):
        seen["args"] = (summarize_set, instructions, strip_media_flag)
        return AttemptResult(ok=True, summary_text=GOOD_SUMMARY)

    def unreachable(*_args):
        raise AssertionError("fallback should not be called")

    wrapped = fallback.with_fallback(primary, unreachable)
    wrapped(["line1"], "focus on X", True)
    assert seen["args"] == (["line1"], "focus on X", True)
