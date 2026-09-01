from cc_local_compact import markdown_out


def _sample_output(tmp_path):
    session = tmp_path / "session.jsonl"
    session.write_text("")
    return markdown_out.CompactionOutput(
        source_session=session,
        trigger="manual",
        model="qwen3.8-27b",
        backend_base_url="http://titan:8080",
        total_groups=5,
        groups_summarized=3,
        groups_preserved=2,
        attempts=2,
        context_budget=32000,
        pre_tokens_estimate=71400,
        post_tokens_estimate=6100,
        custom_instructions="focus on tests",
        summary_text="<analysis>x</analysis>\n\n<summary>\n1. Primary Request:\n   test\n</summary>",
        preserved_tail=[
            {"uuid": "p1", "message": {"role": "user", "content": "hello"}},
            {"uuid": "p2", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}},
        ],
        preserved_segment=markdown_out.PreservedSegment(
            head_uuid="p1", anchor_uuid="p1", tail_uuid="p2", uuids=["p1", "p2"],
        ),
    )


def test_render_markdown_contains_sections(tmp_path):
    output = _sample_output(tmp_path)
    text = markdown_out.render_markdown(output)
    assert "# Compact Summary" in text
    assert "## Metadata" in text
    assert "## Summary" in text
    assert "## Resume Preamble" in text
    assert "## Preserved Tail" in text
    assert "- model: qwen3.8-27b" in text
    assert "- groups_preserved: 2" in text
    assert "- preserved_segment.uuid_count: 2" in text
    assert '["p1", "p2"]' in text
    assert "Summary:" in text
    assert "- [user] hello" in text


def test_write_markdown_creates_file(tmp_path):
    output = _sample_output(tmp_path)
    out_path = tmp_path / "out" / "summary.md"
    written = markdown_out.write_markdown(output, out_path)
    assert written == out_path
    assert out_path.exists()
    assert "# Compact Summary" in out_path.read_text()


def test_default_output_path_uses_session_parent(tmp_path):
    session = tmp_path / "abcd-1234.jsonl"
    path = markdown_out.default_output_path(session)
    assert path.parent.parent == tmp_path
    assert path.parent.name == "compact-summaries"
    assert path.name.startswith("abcd-1234-compact-")
