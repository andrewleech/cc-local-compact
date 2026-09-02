"""Token estimation, ported from Claude Code's Nc/nt/DHe/ljn/th/xh (chunk-
4scc8rka.js, module 356 of the 2.1.252 build -- see
docs/compact-architecture.md in the cc-patcher repo for the source).

Not a real tokenizer: chars.length/4 rounded, per text/thinking/tool-use-
input(JSON-stringified)/redacted-thinking block, flat 2000 for image/
document blocks, recursing through tool_result content. This is what drives
Claude Code's own retry cadence in the summarize/preserve loop, so matching
it (rather than using a real tokenizer) keeps this port's backoff behavior
aligned with the real app's.

`estimate_transcript_tokens` additionally ports `xh`: anchor on the most
recent assistant line carrying a real API usage block and trust it,
estimating only what comes after with the heuristic above. The source JS
also excludes a background/utility model (`rd`) and a sentinel short-
response set (`yce`) from counting as a usage anchor; neither was resolved
to a concrete value during reverse engineering, so `excluded_model` here
defaults to no exclusion -- pass a value if a specific model id needs
excluding.
"""

import json


def count_chars_over_n(text: str | None, chars_per_token: int = 4) -> int:
    """Port of Nc."""
    if not isinstance(text, str):
        return 0
    return round(len(text) / chars_per_token)


def estimate_block(block, chars_per_token: int = 4) -> int:
    """Port of nt: token estimate for one content block."""
    if isinstance(block, str):
        return count_chars_over_n(block, chars_per_token)
    if not isinstance(block, dict):
        return count_chars_over_n(json.dumps(block), chars_per_token)

    block_type = block.get("type")
    if block_type == "text":
        return count_chars_over_n(block.get("text"), chars_per_token)
    if block_type in ("image", "document"):
        return 2000
    if block_type == "tool_result":
        return estimate_content(block.get("content"), chars_per_token)
    if block_type == "tool_use":
        return count_chars_over_n(
            (block.get("name") or "") + json.dumps(block.get("input") or {}),
            chars_per_token,
        )
    if block_type == "thinking":
        # DEVIATION from nt: the real function only counts `.thinking`
        # (the visible text), never `.signature`. Confirmed against a real
        # extended-thinking session: 117 assistant turns had thinking:""
        # (visible text redacted/summarized away) with a substantial
        # signature blob -- 518,620 signature chars (~129,655 estimated
        # tokens) completely invisible to the unmodified heuristic across
        # one 488K-token transcript, enough to make a "residual fits"
        # convergence claim in multipass.py meaningfully wrong. The real
        # app can afford to ignore this (Claude's own context margin is
        # large); this tool's whole purpose is fitting a much smaller
        # target window reliably, so undercounting real payload data by
        # this much is a correctness problem here that it isn't there.
        # The signature is verbatim payload if this message is ever
        # resubmitted, so it counts the same way redacted_thinking's
        # `.data` already does below.
        return count_chars_over_n(block.get("thinking"), chars_per_token) + count_chars_over_n(
            block.get("signature"), chars_per_token,
        )
    if block_type == "redacted_thinking":
        return count_chars_over_n(block.get("data"), chars_per_token)
    return count_chars_over_n(json.dumps(block), chars_per_token)


def estimate_content(content, chars_per_token: int = 4) -> int:
    """Port of DHe: token estimate for a message's whole content field,
    which may be a plain string or a list of content blocks."""
    if not content:
        return 0
    if isinstance(content, str):
        return count_chars_over_n(content, chars_per_token)
    return sum(estimate_block(block, chars_per_token) for block in content)


def _estimate_attachment(line: dict, chars_per_token: int = 4) -> int:
    """Port of ljn's attachment branch. The source function `yie` (which
    turns a raw `attachment` field into a list of {message: {content}}
    items) wasn't resolved during reverse engineering, so this accepts
    either that shape directly or a bare {"message": {"content": ...}}
    dict, and returns 0 for anything else rather than guessing further."""
    attachment = line.get("attachment")
    if not attachment:
        return 0
    if isinstance(attachment, dict) and "message" in attachment:
        items = [attachment]
    elif isinstance(attachment, list):
        items = attachment
    else:
        return 0
    total = 0
    for item in items:
        message = (item or {}).get("message") or {}
        total += estimate_content(message.get("content"), chars_per_token)
    return total


def estimate_line(line: dict, chars_per_token: int = 4) -> int:
    """Port of ljn: token estimate for one transcript line."""
    line_type = line.get("type")
    if line_type in ("assistant", "user", "api_system"):
        message = line.get("message") or {}
        if message.get("content"):
            return estimate_content(message.get("content"), chars_per_token)
        return 0
    if line_type == "attachment":
        return _estimate_attachment(line, chars_per_token)
    return 0


def estimate_group_tokens(lines: list[dict], chars_per_token: int = 4) -> int:
    """Port of th: token estimate for a list of transcript lines (one
    group, or any other slice)."""
    return sum(estimate_line(line, chars_per_token) for line in lines)


def _has_real_usage(line: dict, excluded_model: str | None) -> dict | None:
    """Port of kh."""
    if line.get("type") != "assistant":
        return None
    message = line.get("message") or {}
    usage = message.get("usage")
    if usage is None:
        return None
    if excluded_model is not None and message.get("model") == excluded_model:
        return None
    return usage


def _real_message_id(line: dict, excluded_model: str | None) -> str | None:
    """Port of the."""
    if line.get("type") != "assistant":
        return None
    message = line.get("message") or {}
    if "id" not in message:
        return None
    if excluded_model is not None and message.get("model") == excluded_model:
        return None
    return message["id"]


def _sum_usage(usage: dict) -> int:
    """Port of MG."""
    return (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("output_tokens", 0)
    )


def _find_usage_anchor(
    lines: list[dict], excluded_model: str | None,
) -> tuple[dict, int] | None:
    """Port of aYe: scan backward for the most recent assistant line
    carrying a real usage block, extending the anchor index across earlier
    lines that share the same message.id (a multi-block streamed response)."""
    index = len(lines) - 1
    while index >= 0:
        usage = _has_real_usage(lines[index], excluded_model)
        if usage is not None:
            message_id = _real_message_id(lines[index], excluded_model)
            if message_id:
                walk = index - 1
                while walk >= 0:
                    other_id = _real_message_id(lines[walk], excluded_model)
                    if other_id == message_id:
                        index = walk
                    elif other_id is not None:
                        break
                    walk -= 1
            return usage, index
        index -= 1
    return None


def estimate_transcript_tokens(
    lines: list[dict], chars_per_token: int = 4, excluded_model: str | None = None,
) -> int:
    """Port of xh: trust the most recent real API usage total, estimate
    only what comes after it with the chars/4 heuristic."""
    anchor = _find_usage_anchor(lines, excluded_model)
    if anchor is None:
        return estimate_group_tokens(lines, chars_per_token)
    usage, anchor_index = anchor
    return _sum_usage(usage) + estimate_group_tokens(
        lines[anchor_index + 1:], chars_per_token,
    )
