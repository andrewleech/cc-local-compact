"""Model client: talks to the local Anthropic-protocol-compatible backend
(llama-swap on titan:8080), ported from Claude Code's NBn/rw and the
response-classification helpers Eb/LD/j9/tR/zse (chunk-4scc8rka.js, module
356 of the 2.1.252 build; see docs/compact-architecture.md in the
cc-patcher repo, "The summarization call").

Deliberately dropped from the source: the `grt` system-prompt/tool-list
scaffold (see prompts.py), the precompute cache (Bln, no live session to
speculate against), and PreCompact/PostCompact hook dispatch (Z4/DPe, a
later phase, not v1).
"""

import dataclasses
import json

import anthropic

from . import config, prompts


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    ok: bool
    summary_text: str | None = None
    total_usage: dict | None = None
    reason: str | None = None
    """"prompt_too_long" | "media_too_large" | "error" | "aborted"; only
    set when ok is False."""
    token_gap: int | None = None
    detail: str | None = None
    used_fallback: bool = False
    """Set by fallback.with_fallback when this result came from the
    fallback model, not the primary. Always False from summarize_group
    itself; it has no concept of "fallback," only the composing wrapper
    does."""


PROMPT_TOO_LONG_PREFIX = "Prompt is too long"


def strip_media(group: list[dict]) -> list[dict]:
    """Port of yRe: replace image/document content blocks (including
    inside nested tool_result content) with [image]/[document] placeholder
    text, only allocating new objects for lines that actually contained
    media."""
    result: list[dict] = []
    for line in group:
        if line.get("type") != "user":
            result.append(line)
            continue
        message = line.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            result.append(line)
            continue

        changed = False
        new_content = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            if block.get("type") == "image":
                changed = True
                new_content.append({"type": "text", "text": "[image]"})
            elif block.get("type") == "document":
                changed = True
                new_content.append({"type": "text", "text": "[document]"})
            elif block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                tr_changed = False
                new_tr_content = []
                for tr_block in block["content"]:
                    if isinstance(tr_block, dict) and tr_block.get("type") == "image":
                        tr_changed = True
                        new_tr_content.append({"type": "text", "text": "[image]"})
                    elif isinstance(tr_block, dict) and tr_block.get("type") == "document":
                        tr_changed = True
                        new_tr_content.append({"type": "text", "text": "[document]"})
                    else:
                        new_tr_content.append(tr_block)
                if tr_changed:
                    changed = True
                    new_content.append({**block, "content": new_tr_content})
                else:
                    new_content.append(block)
            else:
                new_content.append(block)

        if changed:
            result.append({**line, "message": {**message, "content": new_content}})
        else:
            result.append(line)
    return result


STANDARD_CONTENT_BLOCK_TYPES = {
    "text", "image", "document", "tool_use", "tool_result", "thinking", "redacted_thinking",
}
"""The Anthropic Messages API's own content block types (matching what
tokens.py's estimate_block already recognizes). Claude Code's own JSONL
also carries block types that are internal to the real client and never
part of the actual API schema, e.g. `tool_reference` (seen inside a
tool_result's content for control-flow tools like EnterPlanMode, which
don't have a real result payload). CONFIRMED against the live backend:
llama.cpp's Anthropic-compat shim (qwen3.8-27b, gemma) tolerates an
unrecognized block type, passing it through; vLLM's shim (qwen3.5-9b) does
strict schema validation and rejects the whole request with "Unexpected
item type in content." - bisected down to exactly this block type on a
real 488K-token transcript. Sanitizing before sending is required for
correctness against any strict backend, not just a qwen3.5-9b quirk."""


def _sanitize_block(block):
    """Replace a non-standard content block with a plain text block
    carrying a best-effort textual description, so a strict backend never
    sees a block type outside the real Anthropic API schema. Recurses into
    tool_result's own nested content, where the non-standard types have
    actually been observed."""
    if not isinstance(block, dict):
        return block
    block_type = block.get("type")
    if block_type == "tool_result" and isinstance(block.get("content"), list):
        return {**block, "content": [_sanitize_block(b) for b in block["content"]]}
    if block_type in STANDARD_CONTENT_BLOCK_TYPES:
        return block
    if block_type == "tool_reference":
        return {"type": "text", "text": f"[tool_reference: {block.get('tool_name', '?')}]"}
    return {"type": "text", "text": json.dumps(block)}


def _sanitize_content(content):
    if not isinstance(content, list):
        return content
    return [_sanitize_block(block) for block in content]


def _to_api_messages(group: list[dict]) -> list[dict]:
    """Reduce transcript lines to a minimal Anthropic messages array:
    {"role", "content"} only, dropping Claude-Code-only wrapper fields
    (uuid, parentUuid, timestamp, cwd, etc), and sanitizing any non-
    standard content block type (see STANDARD_CONTENT_BLOCK_TYPES). Lines
    without a user/assistant role message (attachment/system control
    lines) are skipped; they aren't valid Anthropic message-array
    entries."""
    messages = []
    for line in group:
        message = line.get("message")
        if not message or message.get("role") not in ("user", "assistant"):
            continue
        messages.append({
            "role": message["role"],
            "content": _sanitize_content(message.get("content")),
        })
    return messages


def _classify_error(error: anthropic.APIStatusError) -> AttemptResult:
    """Ports Eb (prompt-too-long detection) / LD (token-overage
    extraction) / j9 (media-too-large detection) / tR (error text
    extraction). The real app reads structured actualTokens/limitTokens
    fields Anthropic's own API returns on a context-overflow error.

    CONFIRMED against the live backend (llama-swap fronting llama.cpp on
    titan:8080, 2026-09-01): it does NOT emit Anthropic's shape at all. A
    genuine overflow comes back as HTTP 400 with:
        {"error": {"code": 400, "message": "request (N tokens) exceeds
         the available context size (M tokens), try increasing it",
         "type": "exceed_context_size_error",
         "n_prompt_tokens": N, "n_ctx": M}}
    This is checked FIRST, ahead of the Anthropic-shaped check (kept as a
    fallback in case a future backend or engine swap emits that shape
    instead). Because this gives an exact prompt-token count and context
    limit straight from the server, the resulting token_gap is a precise
    value, not an estimate; more precise than loop.py's self-estimated-
    gap fallback, which now only matters if a backend emits neither shape.

    The media-too-large check here is a keyword heuristic on the error
    message, since the source's own detector (NZt) wasn't resolved during
    reverse engineering, and no real media-too-large error has been
    observed against this backend to confirm its shape."""
    body = getattr(error, "body", None) or {}
    err = body.get("error") if isinstance(body, dict) else None
    message_text = (err or {}).get("message") if isinstance(err, dict) else None
    message_text = message_text or str(error)

    if isinstance(err, dict) and err.get("type") == "exceed_context_size_error":
        n_prompt_tokens = err.get("n_prompt_tokens")
        n_ctx = err.get("n_ctx")
        token_gap = None
        if isinstance(n_prompt_tokens, int) and isinstance(n_ctx, int) and n_prompt_tokens > n_ctx:
            token_gap = n_prompt_tokens - n_ctx
        return AttemptResult(ok=False, reason="prompt_too_long", token_gap=token_gap, detail=message_text)

    if message_text.startswith(PROMPT_TOO_LONG_PREFIX):
        token_gap = None
        if isinstance(err, dict):
            actual = err.get("actualTokens", err.get("actual_tokens"))
            limit = err.get("limitTokens", err.get("limit_tokens"))
            if isinstance(actual, int) and isinstance(limit, int) and actual > limit:
                token_gap = actual - limit
        return AttemptResult(ok=False, reason="prompt_too_long", token_gap=token_gap, detail=message_text)

    lowered = message_text.lower()
    if "media" in lowered and "large" in lowered:
        return AttemptResult(ok=False, reason="media_too_large", detail=message_text)

    return AttemptResult(ok=False, reason="error", detail=message_text)


REQUEST_TIMEOUT_SECONDS = 1800.0
"""The anthropic SDK refuses a non-streaming call above ~21K max_tokens
with no explicit timeout (its own client-side guard, assuming a request
that large will take over 10 minutes and should stream to avoid a proxy
timeout); hit in practice once context_budget-derived max_tokens grew
past that for qwen3.8-27b's large window. This backend's own streaming
implementation was tried as the fix and hung indefinitely (confirmed: even
a trivial 2+2 request via client.messages.stream() never returned);
whatever its Anthropic-compat SSE framing is doing, this SDK's stream
consumer doesn't see a terminating event. Passing any explicit `timeout`
to create() bypasses the SDK's guard entirely (it only computes/applies
that guard when the caller hasn't set one), so this stays on the
known-working non-streaming path with a generous ceiling instead."""


def summarize_group(
    client: anthropic.Anthropic,
    model: str,
    group: list[dict],
    custom_instructions: str | None,
    strip_media_flag: bool,
    max_tokens: int,
) -> AttemptResult:
    """Port of NBn: one summarization attempt over one summarize-set. The
    model input is the group's own messages followed by one trailing
    synthetic user turn carrying the summarization instructions; the
    model literally continues the conversation, with no separate system
    prompt and no tools offered.

    Native reasoning is explicitly disabled for this call, for models
    confirmed to need it (see config.MODELS_WITH_THINKING_TOGGLE).
    CONFIRMED against the live backend: qwen3.8-27b has reasoning enabled
    by default, emitting a hidden `thinking` content block that counts
    against max_tokens even for a trivial response (27 output tokens for
    "2+2=4"), and this module's own prompt already asks the model to
    externalize its reasoning as visible <analysis> text; a second,
    hidden reasoning channel is pure budget waste for this task, not a
    quality benefit, and was a real contributor to the truncation bug
    documented in response.py. Anthropic's own `thinking` param and
    `reasoning_effort=low/none` were both tried and silently ignored by
    this backend; only `extra_body.chat_template_kwargs.enable_thinking`
    (a llama.cpp/Jinja chat-template mechanism, not an Anthropic API
    parameter) actually suppressed it, confirmed by content blocks
    dropping from ['thinking','text'] to just ['text'].

    NOT sent unconditionally: confirmed NOT harmless for every model.
    qwen3.5-9b never emits a thinking block whether the override is sent
    or not (content blocks are ['text'] either way), but sending it anyway
    cost ~43x the latency for identical output in a real side-by-side test
    (147.4s vs 3.4s, same input_tokens/output_tokens both runs); some
    backends appear to take a much slower code path when handling a
    non-default chat_template_kwargs value even when it changes nothing
    observable. See config.MODELS_WITH_THINKING_TOGGLE for the allowlist
    and its own per-model evidence."""
    working_group = strip_media(group) if strip_media_flag else group
    messages = _to_api_messages(working_group)
    messages.append({"role": "user", "content": prompts.build_prompt(custom_instructions)})

    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if model in config.MODELS_WITH_THINKING_TOGGLE else None
    )

    try:
        response = client.messages.create(
            model=model, max_tokens=max_tokens, messages=messages,
            **({"extra_body": extra_body} if extra_body else {}),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except anthropic.APIStatusError as error:
        return _classify_error(error)
    except anthropic.APIError as error:
        return AttemptResult(ok=False, reason="error", detail=str(error))

    summary_text = next(
        (block.text for block in response.content if getattr(block, "type", None) == "text"),
        None,
    )
    if not summary_text:
        return AttemptResult(ok=False, reason="error", detail="summarization produced empty response")

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    }
    return AttemptResult(ok=True, summary_text=summary_text, total_usage=usage)
