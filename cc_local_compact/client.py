"""Model client: talks to the local Anthropic-protocol-compatible backend
(llama-swap on titan:8080), ported from Claude Code's NBn/rw and the
response-classification helpers Eb/LD/j9/tR/zse (chunk-4scc8rka.js, module
356 of the 2.1.252 build -- see docs/compact-architecture.md in the
cc-patcher repo, "The summarization call").

Deliberately dropped from the source: the `grt` system-prompt/tool-list
scaffold (see prompts.py), the precompute cache (Bln -- no live session to
speculate against), and PreCompact/PostCompact hook dispatch (Z4/DPe -- a
later phase, not v1).
"""

import dataclasses

import anthropic

from . import prompts


@dataclasses.dataclass(frozen=True)
class AttemptResult:
    ok: bool
    summary_text: str | None = None
    total_usage: dict | None = None
    reason: str | None = None
    """"prompt_too_long" | "media_too_large" | "error" | "aborted" -- only
    set when ok is False."""
    token_gap: int | None = None
    detail: str | None = None


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


def _to_api_messages(group: list[dict]) -> list[dict]:
    """Reduce transcript lines to a minimal Anthropic messages array:
    {"role", "content"} only, dropping Claude-Code-only wrapper fields
    (uuid, parentUuid, timestamp, cwd, etc). Lines without a user/assistant
    role message (attachment/system control lines) are skipped -- they
    aren't valid Anthropic message-array entries."""
    messages = []
    for line in group:
        message = line.get("message")
        if not message or message.get("role") not in ("user", "assistant"):
            continue
        messages.append({"role": message["role"], "content": message.get("content")})
    return messages


def _classify_error(error: anthropic.APIStatusError) -> AttemptResult:
    """Ports Eb (prompt-too-long detection) / LD (token-overage
    extraction) / j9 (media-too-large detection) / tR (error text
    extraction). The real app reads structured actualTokens/limitTokens
    fields Anthropic's own API returns on a context-overflow error; whether
    llama-swap's Anthropic-compatible adapter emits the same shape is
    unverified (backend unreachable at implementation time). The media-
    too-large check here is a keyword heuristic on the error message, since
    the source's own detector (NZt) wasn't resolved during reverse
    engineering -- see loop.py for the self-estimated-gap fallback this
    feeds into."""
    body = getattr(error, "body", None) or {}
    err = body.get("error") if isinstance(body, dict) else None
    message_text = (err or {}).get("message") if isinstance(err, dict) else None
    message_text = message_text or str(error)

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
    synthetic user turn carrying the summarization instructions -- the
    model literally continues the conversation, with no separate system
    prompt and no tools offered."""
    working_group = strip_media(group) if strip_media_flag else group
    messages = _to_api_messages(working_group)
    messages.append({"role": "user", "content": prompts.build_prompt(custom_instructions)})

    try:
        response = client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
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
