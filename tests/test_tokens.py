import json

from cc_local_compact import tokens


def test_count_chars_over_n():
    assert tokens.count_chars_over_n("abcd", chars_per_token=4) == 1
    assert tokens.count_chars_over_n("abcdefgh", chars_per_token=4) == 2
    assert tokens.count_chars_over_n(None) == 0
    assert tokens.count_chars_over_n(123) == 0


def test_estimate_block_text():
    block = {"type": "text", "text": "abcdefgh"}
    assert tokens.estimate_block(block) == 2


def test_estimate_block_image_and_document_flat_cost():
    assert tokens.estimate_block({"type": "image"}) == 2000
    assert tokens.estimate_block({"type": "document"}) == 2000


def test_estimate_block_tool_use_uses_json_input():
    block = {"type": "tool_use", "name": "Read", "input": {"path": "x"}}
    expected = tokens.count_chars_over_n("Read" + json.dumps({"path": "x"}))
    assert tokens.estimate_block(block) == expected


def test_estimate_block_tool_result_recurses():
    block = {"type": "tool_result", "content": [{"type": "text", "text": "abcd"}]}
    assert tokens.estimate_block(block) == 1


def test_estimate_block_thinking_and_redacted_thinking():
    assert tokens.estimate_block({"type": "thinking", "thinking": "abcd"}) == 1
    assert tokens.estimate_block({"type": "redacted_thinking", "data": "abcdefgh"}) == 2


def test_estimate_block_thinking_counts_signature_too():
    # DEVIATION from the real nt: a thinking block with empty visible text
    # but a real signature blob is real payload data, not free -- see
    # tokens.py's estimate_block for why this diverges from the source.
    block = {"type": "thinking", "thinking": "", "signature": "abcdefgh"}
    assert tokens.estimate_block(block) == 2
    block_with_both = {"type": "thinking", "thinking": "abcd", "signature": "abcdefgh"}
    assert tokens.estimate_block(block_with_both) == 1 + 2


def test_estimate_content_sums_blocks():
    content = [{"type": "text", "text": "abcd"}, {"type": "image"}]
    assert tokens.estimate_content(content) == 1 + 2000


def test_estimate_content_plain_string():
    assert tokens.estimate_content("abcdefgh") == 2


def test_estimate_content_empty():
    assert tokens.estimate_content(None) == 0
    assert tokens.estimate_content("") == 0
    assert tokens.estimate_content([]) == 0


def test_estimate_group_tokens_sums_lines():
    lines = [
        {"type": "user", "message": {"content": "abcd"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "abcdefgh"}]}},
        {"type": "progress"},
    ]
    assert tokens.estimate_group_tokens(lines) == 1 + 2


def test_estimate_transcript_tokens_uses_usage_anchor():
    lines = [
        {"type": "user", "message": {"content": "abcd"}},
        {
            "type": "assistant",
            "message": {
                "id": "m1", "model": "qwen3.8-27b",
                "content": [{"type": "text", "text": "abcdefgh"}],
                "usage": {
                    "input_tokens": 100, "output_tokens": 10,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                },
            },
        },
        {"type": "user", "message": {"content": "abcdefghijkl"}},
    ]
    assert tokens.estimate_transcript_tokens(lines) == 110 + 3


def test_estimate_transcript_tokens_falls_back_without_usage():
    lines = [
        {"type": "user", "message": {"content": "abcd"}},
        {"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": "abcdefgh"}]}},
    ]
    assert tokens.estimate_transcript_tokens(lines) == tokens.estimate_group_tokens(lines)
