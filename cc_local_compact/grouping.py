"""Message grouping, ported from Claude Code's Mq/n$ (chunk-4scc8rka.js,
module 356 of the 2.1.252 build; see docs/compact-architecture.md in the
cc-patcher repo for the source).

A "group" is one assistant turn, keyed on the Anthropic API's message.id,
not wall-clock time, not a fixed message count. progress-type lines are
dropped before grouping. isVirtual user/assistant lines always join the
current group. A new group starts only when an assistant line's message.id
differs from the previous assistant line's, and it isn't a
resumedFromIncompleteThinking continuation.
"""


def group_messages(lines: list[dict]) -> list[list[dict]]:
    """Port of Mq: drop progress-type lines, then group by assistant turn."""
    filtered = [line for line in lines if line.get("type") != "progress"]
    return _group_by_assistant_turn(filtered)


def _group_by_assistant_turn(lines: list[dict]) -> list[list[dict]]:
    """Port of n$."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    last_message_id: str | None = None

    for line in lines:
        if line.get("type") in ("user", "assistant") and line.get("isVirtual") is True:
            current.append(line)
            continue

        message_id = (line.get("message") or {}).get("id")
        if (
            line.get("type") == "assistant"
            and message_id != last_message_id
            and not line.get("resumedFromIncompleteThinking")
            and current
        ):
            groups.append(current)
            current = [line]
        else:
            current.append(line)

        if line.get("type") == "assistant":
            last_message_id = message_id

    if current:
        groups.append(current)
    return groups
