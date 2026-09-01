from cc_local_compact import prompts


def test_build_prompt_without_custom_instructions():
    text = prompts.build_prompt(None)
    assert text.startswith(prompts.PREAMBLE)
    assert "Additional Instructions:" not in text
    assert text.endswith(prompts.REMINDER_SUFFIX)
    assert prompts.INSTRUCTIONS_BODY in text


def test_build_prompt_with_custom_instructions():
    text = prompts.build_prompt("focus on tests")
    assert "Additional Instructions:\nfocus on tests" in text
    assert text.endswith(prompts.REMINDER_SUFFIX)


def test_build_prompt_ignores_blank_custom_instructions():
    text = prompts.build_prompt("   ")
    assert "Additional Instructions:" not in text


def test_verbatim_markers_present():
    assert "<analysis>" in prompts.INSTRUCTIONS_BODY
    assert "<summary>" in prompts.INSTRUCTIONS_BODY
    assert "Do NOT call any tools" in prompts.PREAMBLE
    assert "REMINDER: Do NOT call any tools" in prompts.REMINDER_SUFFIX
