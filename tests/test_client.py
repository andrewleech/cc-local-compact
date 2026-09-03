import httpx
import anthropic

from cc_local_compact import client, config


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _FakeMessage:
    def __init__(self):
        self.content = [_FakeTextBlock("Summary:\ndone")]
        self.usage = _FakeUsage()


class _FakeMessagesResource:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeMessage()


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessagesResource()


def _status_error(body: dict, status_code: int = 400) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "http://titan:8080/v1/messages")
    response = httpx.Response(status_code, request=request)
    return anthropic.APIStatusError("boom", response=response, body=body)


def test_classify_error_llamaswap_context_overflow():
    # Confirmed shape from the real backend (llama-swap fronting llama.cpp
    # on titan:8080, observed 2026-09-01); see client.py's _classify_error
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


def test_to_api_messages_sanitizes_tool_reference_inside_tool_result():
    # Real bug, bisected against a live backend (qwen3.5-9b on vLLM):
    # a tool_result's content can contain a Claude-Code-internal
    # "tool_reference" block (e.g. for EnterPlanMode, which has no real
    # result payload); llama.cpp's Anthropic-compat shim tolerates it,
    # vLLM's rejects the whole request with "Unexpected item type in
    # content." See client.py's STANDARD_CONTENT_BLOCK_TYPES.
    group = [{
        "type": "user",
        "message": {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"type": "tool_reference", "tool_name": "EnterPlanMode"}],
            }],
        },
    }]
    messages = client._to_api_messages(group)
    tool_result = messages[0]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"] == [{"type": "text", "text": "[tool_reference: EnterPlanMode]"}]


def test_to_api_messages_sanitizes_unknown_top_level_block():
    group = [{
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "some_future_block", "data": "x"}]},
    }]
    messages = client._to_api_messages(group)
    block = messages[0]["content"][0]
    assert block["type"] == "text"
    assert "some_future_block" in block["text"]


def test_to_api_messages_leaves_standard_blocks_untouched():
    group = [{
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    }]
    messages = client._to_api_messages(group)
    assert messages[0]["content"] == [{"type": "text", "text": "hi"}]


def test_summarize_group_sends_thinking_override_for_allowlisted_model():
    fake = _FakeClient()
    group = [{"type": "user", "message": {"role": "user", "content": "hi"}}]
    assert "qwen3.8-27b" in config.MODELS_WITH_THINKING_TOGGLE
    client.summarize_group(fake, "qwen3.8-27b", group, None, False, 100)
    assert fake.messages.last_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_summarize_group_omits_thinking_override_for_unlisted_model():
    # Confirmed harmful for qwen3.5-9b specifically: never changes output
    # (content blocks are ['text'] with or without it) but cost ~43x the
    # latency in a real side-by-side test. See config.MODELS_WITH_THINKING_TOGGLE.
    fake = _FakeClient()
    group = [{"type": "user", "message": {"role": "user", "content": "hi"}}]
    assert "qwen3.5-9b" not in config.MODELS_WITH_THINKING_TOGGLE
    client.summarize_group(fake, "qwen3.5-9b", group, None, False, 100)
    assert "extra_body" not in fake.messages.last_kwargs
