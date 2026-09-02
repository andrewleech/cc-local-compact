"""Appends a compact_boundary + isCompactSummary + re-chained preserved-
tail sequence into a session's own JSONL, matching the exact on-disk shape
Claude Code's real /compact produces (see docs/compact-architecture.md in
the cc-patcher repo, "Confirmed on-disk JSONL schema").

CONFIRMED SAFE, CONFIRMED INERT for reducing future cost: appending this
sequence to a session's JSONL does not corrupt the file or crash the
client -- tested directly against a live throwaway session (write the
sequence, then `claude --resume` it, repeated across multiple trials with
no errors). But it does NOT reduce what Claude Code sends to the remote
model on the next resumed turn: the same experiment showed a resumed
session with an injected boundary reprocessing the FULL original
conversation, token-for-token indistinguishable from a normal resume with
no boundary at all. Separately confirmed via Claude Code's own
documentation (cross-session-messaging.md): slash commands arriving
through any channel other than direct interactive terminal input --
including a session's own JSONL, MCP tool/prompt output, Remote Control,
and the local cross-session messaging socket -- are never executed; there
is no external mechanism to trigger the real client-side /compact or
/clear.

So: this module exists for **on-disk record consistency**, not as a
cache-reload workaround. It makes a session's own transcript reflect that
a compaction happened, in the same shape the real feature produces (for
`/resume` picker display, transcript tooling, or anything else that parses
this schema) -- not to make the *next* remote-model turn any cheaper.
Callers (server.py, cli.py) should present this as an explicit opt-in with
that caveat stated, not as a default or an implied performance win.
"""

import dataclasses
import datetime
import json
import uuid as uuid_mod
from pathlib import Path

from . import response


SESSION_METADATA_KEYS = ("sessionId", "cwd", "version", "gitBranch", "userType", "entrypoint")


@dataclasses.dataclass(frozen=True)
class AppendResult:
    boundary_uuid: str
    anchor_uuid: str
    logical_parent_uuid: str | None
    lines_appended: int


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _read_raw_lines(session_path: Path) -> list[dict]:
    return [json.loads(line) for line in session_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _session_metadata_from(sample_line: dict) -> dict:
    """Pull the session-identity fields (sessionId, cwd, version,
    gitBranch, userType, entrypoint) from an existing line in the target
    session, so appended lines match the session's own conventions rather
    than guessing at values that could drift from what the real client
    writes."""
    return {key: sample_line[key] for key in SESSION_METADATA_KEYS if key in sample_line}


def build_boundary_line(
    trigger: str,
    pre_tokens: int,
    post_tokens: int,
    duration_ms: int,
    preserved_uuids: list[str],
    anchor_uuid: str,
    logical_parent_uuid: str | None,
    session_metadata: dict,
    boundary_uuid: str | None = None,
    discovered_tools: list[str] | None = None,
) -> dict:
    """Port of Xz + GTe's combined output shape (see compact-architecture.md)."""
    compact_metadata = {
        "trigger": trigger,
        "preTokens": pre_tokens,
        "postTokens": post_tokens,
        "cumulativeDroppedTokens": max(0, pre_tokens - post_tokens),
        "durationMs": duration_ms,
    }
    if preserved_uuids:
        compact_metadata["preservedSegment"] = {
            "headUuid": preserved_uuids[0],
            "anchorUuid": anchor_uuid,
            "tailUuid": preserved_uuids[-1],
        }
        # NOTE: the real xPe filters progress-type/empty-hook entries out of
        # "uuids" while "allUuids" keeps the unfiltered set. Our preserved
        # tail is already filtered to KEPT_TYPES by transcript.py, so there's
        # no further distinction to draw here -- both lists are identical, a
        # known simplification relative to the source.
        compact_metadata["preservedMessages"] = {
            "anchorUuid": anchor_uuid,
            "uuids": preserved_uuids,
            "allUuids": preserved_uuids,
        }
    if discovered_tools:
        compact_metadata["preCompactDiscoveredTools"] = sorted(discovered_tools)

    line = {
        "parentUuid": None,
        "isSidechain": False,
        "type": "system",
        "subtype": "compact_boundary",
        "content": "Conversation compacted",
        "isMeta": False,
        "level": "info",
        "compactMetadata": compact_metadata,
        "uuid": boundary_uuid or str(uuid_mod.uuid4()),
        "timestamp": _now_iso(),
        **session_metadata,
    }
    if logical_parent_uuid:
        line["logicalParentUuid"] = logical_parent_uuid
    return line


def build_summary_line(summary_content: str, boundary_uuid: str, anchor_uuid: str, session_metadata: dict) -> dict:
    """Port of the isCompactSummary message shape. `summary_content` should
    already be the full Vq-wrapped text (response.build_resume_preamble),
    not the bare cleaned summary."""
    return {
        "parentUuid": boundary_uuid,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": summary_content},
        "isVisibleInTranscriptOnly": True,
        "isCompactSummary": True,
        "uuid": anchor_uuid,
        "timestamp": _now_iso(),
        **session_metadata,
    }


def rechain_preserved_tail(preserved_tail: list[dict], anchor_uuid: str) -> list[dict]:
    """Re-append copies of the preserved-tail lines with parentUuid
    updated to form a fresh chain starting from the summary message
    (anchor_uuid). Each line keeps its own original uuid -- these are the
    same logical messages, just re-parented into the new active chain;
    their original (now-orphaned) copies stay untouched earlier in the
    file, matching append-only semantics (nothing in this module ever
    rewrites an existing line)."""
    rechained = []
    parent = anchor_uuid
    for line in preserved_tail:
        rechained.append({**line, "parentUuid": parent})
        if line.get("uuid"):
            parent = line["uuid"]
    return rechained


def append_compaction(
    session_path: Path,
    summary_text: str,
    preserved_tail: list[dict],
    trigger: str,
    pre_tokens: int,
    post_tokens: int,
    duration_ms: int,
    transcript_path: str | None = None,
) -> AppendResult:
    """Appends a compact_boundary + isCompactSummary + re-chained
    preserved-tail sequence to session_path's own JSONL, in the shape the
    real /compact produces. `summary_text` is the raw (uncleaned) summary
    -- this wraps it with response.build_resume_preamble itself, matching
    how the real app's Vq wrapper is applied to the raw NBn/rw output, not
    a pre-cleaned string. See the module docstring for what this does and
    does not achieve."""
    raw_lines = _read_raw_lines(session_path)
    if not raw_lines:
        raise ValueError(f"{session_path} has no existing lines to anchor a boundary to")

    # The raw file's literal last line is often a control-type entry with
    # no uuid at all (e.g. "last-prompt", "mode", "queue-operation",
    # "atis-latch") -- scan backward for the last line that actually has
    # one, matching a real conversational/system entry, not just take
    # raw_lines[-1] blindly.
    last_line = next((line for line in reversed(raw_lines) if line.get("uuid")), None)
    if last_line is None:
        raise ValueError(f"{session_path} has no line with a uuid to anchor a boundary to")
    logical_parent_uuid = last_line.get("uuid")
    session_metadata = _session_metadata_from(last_line)

    anchor_uuid = str(uuid_mod.uuid4())
    preserved_uuids = [line["uuid"] for line in preserved_tail if line.get("uuid")]

    boundary = build_boundary_line(
        trigger=trigger, pre_tokens=pre_tokens, post_tokens=post_tokens,
        duration_ms=duration_ms, preserved_uuids=preserved_uuids,
        anchor_uuid=anchor_uuid, logical_parent_uuid=logical_parent_uuid,
        session_metadata=session_metadata,
    )
    summary_content = response.build_resume_preamble(
        summary_text, transcript_path=transcript_path or str(session_path),
    )
    summary = build_summary_line(summary_content, boundary["uuid"], anchor_uuid, session_metadata)
    rechained_tail = rechain_preserved_tail(preserved_tail, anchor_uuid)

    new_lines = [boundary, summary, *rechained_tail]
    with session_path.open("a", encoding="utf-8") as f:
        for line in new_lines:
            f.write(json.dumps(line) + "\n")

    return AppendResult(
        boundary_uuid=boundary["uuid"], anchor_uuid=anchor_uuid,
        logical_parent_uuid=logical_parent_uuid, lines_appended=len(new_lines),
    )
