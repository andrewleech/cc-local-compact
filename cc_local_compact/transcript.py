"""JSONL session-transcript parsing.

Claude Code session files are append-only and can interleave lines that
don't belong to the main conversation thread: isSidechain:true lines are a
Task-tool subagent's own turns, and a rewind or a prior compaction can leave
abandoned branches behind. File order alone isn't reliable causal order, so
the main thread is reconstructed by walking parentUuid backward from the
most recent on-thread line to the root, then reversing.
"""

import json
from pathlib import Path


KEPT_TYPES = {"user", "assistant", "system", "attachment"}
"""Line types relevant to grouping/summarization. Real transcripts also
contain "mode", "permission-mode", "queue-operation", "file-history-snapshot",
"ai-title", and "last-prompt" lines, which are control/UI bookkeeping and
never enter a compaction group.

Note "system" here is the line-level type used for control lines like
compact_boundary -- distinct from the "api_system" content-block type
tokens.py checks for inside an assistant/user message's own content array.
"""


def _read_lines(path: Path) -> list[dict]:
    lines: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            lines.append(json.loads(raw))
    return lines


def load_transcript(path: Path) -> list[dict]:
    """Parse a session .jsonl file and return its main-thread messages
    (kept types only) in causal order."""
    all_lines = _read_lines(path)
    uuid_map = {line["uuid"]: line for line in all_lines if "uuid" in line}

    candidates = [
        line for line in all_lines
        if line.get("type") in KEPT_TYPES and not line.get("isSidechain")
    ]
    if not candidates:
        return []

    chain: list[dict] = []
    seen: set[str] = set()
    node: dict | None = candidates[-1]
    while node is not None:
        uuid = node.get("uuid")
        if uuid is not None:
            if uuid in seen:
                break
            seen.add(uuid)
        if not node.get("isSidechain") and node.get("type") in KEPT_TYPES:
            chain.append(node)
        parent_uuid = node.get("parentUuid")
        node = uuid_map.get(parent_uuid) if parent_uuid else None

    chain.reverse()
    return chain
