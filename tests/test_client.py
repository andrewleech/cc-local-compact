import httpx
import anthropic

from cc_local_compact import client


def _status_error(body: dict, status_code: int = 400) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "http://titan:8080/v1/messages")
    response = httpx.Response(status_code, request=request)
    return anthropic.APIStatusError("boom", response=response, body=body)


def test_classify_error_llamaswap_context_overflow():
    # Confirmed shape from the real backend (llama-swap fronting llama.cpp
    # on titan:8080, observed 2026-09-01) -- see client.py's _classify_error
    # docstring. Does NOT match Anthropic's own actualTokens/limitTokens
    # error shape.
    body = {
        "error": {
            "code": 400,
            "message": (
                "request (4000010 tokens) exceeds the available context "
                "size (262144 tokens), try increasing it"
            ),
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 4000010,
            "n_ctx": 262144,
        }
    }
    result = client._classify_error(_status_error(body))
    assert result.ok is False
    assert result.reason == "prompt_too_long"
    assert result.token_gap == 4000010 - 262144


def test_classify_error_anthropic_shape_still_supported():
    body = {
        "error": {
            "message": "Prompt is too long: 250000 tokens > 200000 maximum",
            "actualTokens": 250000,
            "limitTokens": 200000,
        }
    }
    result = client._classify_error(_status_error(body))
    assert result.reason == "prompt_too_long"
    assert result.token_gap == 50000


def test_classify_error_falls_back_to_generic_error():
    body = {"error": {"message": "internal server error", "type": "server_error"}}
    result = client._classify_error(_status_error(body, status_code=500))
    assert result.reason == "error"
    assert result.token_gap is None
    assert result.detail == "internal server error"


def test_strip_media_replaces_image_and_document_blocks():
    group = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image", "source": {"data": "..."}},
                    {"type": "tool_result", "content": [{"type": "document"}]},
                ],
            },
        },
        {"type": "assistant", "message": {"role": "assistant", "content": "ok"}},
    ]
    stripped = client.strip_media(group)
    user_content = stripped[0]["message"]["content"]
    assert user_content[0] == {"type": "text", "text": "look at this"}
    assert user_content[1] == {"type": "text", "text": "[image]"}
    assert user_content[2]["content"] == [{"type": "text", "text": "[document]"}]
    # non-media lines pass through unchanged
    assert stripped[1] == group[1]


def test_strip_media_leaves_lines_without_media_untouched():
    group = [{"type": "user", "message": {"role": "user", "content": "plain text"}}]
    assert client.strip_media(group) == group
