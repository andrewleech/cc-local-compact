"""Markdown output writer.

Writes a standalone summary file rather than appending to the live session
.jsonl -- see docs/compact-architecture.md in the cc-patcher repo,
"Decision for the first implementation". The metadata block is kept as
parseable `key: value` lines rather than prose, so a later JSONL-append
phase can pull the preserved-tail uuids and token counts back out
programmatically without re-parsing the summary body.
"""

import dataclasses
import datetime
import json
from pathlib import Path

from . import response
from .multipass import PassRecord


@dataclasses.dataclass(frozen=True)
class PreservedSegment:
    head_uuid: str | None
    anchor_uuid: str | None
    tail_uuid: str | None
    uuids: list[str]


@dataclasses.dataclass(frozen=True)
class CompactionOutput:
    source_session: Path
    trigger: str
    model: str
    backend_base_url: str
    context_budget: int
    passes: tuple[PassRecord, ...]
    multi_pass_reason: str | None
    """Non-fatal caveat from a multi-pass run ("max_passes_reached" or
    "later_pass_failed"), or None if it converged cleanly."""
    pre_tokens_estimate: int
    """Estimate over the FULL original transcript, before any pass."""
    post_tokens_estimate: int
    """Estimate over the FINAL residual (cleaned summary + preserved
    tail combined) -- the same quantity real Claude Code's own
    compactMetadata.postTokens measures, for direct comparability."""
    custom_instructions: str | None
    summary_text: str
    preserved_tail: list[dict]
    preserved_segment: PreservedSegment
    generated_at: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


def default_output_path(session_path: Path) -> Path:
    """~/.claude/projects/<project-dir-slug>/compact-summaries/<session-
    uuid>-compact-<UTC-timestamp>.md. session_path's own parent directory
    IS the project-dir slug already, so there's no need to recompute it
    from cwd here."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = session_path.parent / "compact-summaries"
    return out_dir / f"{session_path.stem}-compact-{timestamp}.md"


def _preview_line(line: dict, max_len: int = 100) -> str:
    message = line.get("message") or {}
    role = message.get("role", line.get("type", "?"))
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                parts.append(f"[tool_use: {block.get('name')}]")
            elif block.get("type") == "tool_result":
                parts.append("[tool_result]")
        text = " ".join(p for p in parts if p)
    else:
        text = ""
    text = text.strip().replace("\n", " ")
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return f"- [{role}] {text}"


def _pass_table(passes: tuple[PassRecord, ...]) -> list[str]:
    header = "| pass | attempts | total_groups | groups_preserved | pre_tokens | post_tokens | fallback |"
    sep = "|---|---|---|---|---|---|---|"
    rows = [
        f"| {p.pass_number} | {p.attempts} | {p.total_groups} | {p.groups_preserved} "
        f"| {p.pre_tokens} | {p.post_tokens} | {'yes' if p.used_fallback else 'no'} |"
        for p in passes
    ]
    return [header, sep, *rows]


def render_markdown(output: CompactionOutput) -> str:
    total_attempts = sum(p.attempts for p in output.passes)
    metadata_lines = [
        f"- source_session: {output.source_session}",
        f"- generated_at: {output.generated_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- trigger: {output.trigger}",
        f"- model: {output.model}",
        f"- backend_base_url: {output.backend_base_url}",
        f"- context_budget: {output.context_budget}",
        f"- pass_count: {len(output.passes)}",
        f"- total_attempts: {total_attempts}",
        f"- multi_pass_reason: {output.multi_pass_reason or '(converged)'}",
        f"- pre_tokens_estimate: {output.pre_tokens_estimate}",
        f"- post_tokens_estimate: {output.post_tokens_estimate}",
        f"- custom_instructions: {output.custom_instructions or '(none)'}",
        f"- preserved_segment.head_uuid: {output.preserved_segment.head_uuid}",
        f"- preserved_segment.anchor_uuid: {output.preserved_segment.anchor_uuid}",
        f"- preserved_segment.tail_uuid: {output.preserved_segment.tail_uuid}",
        f"- preserved_segment.uuid_count: {len(output.preserved_segment.uuids)}",
    ]

    resume_preamble = response.build_resume_preamble(
        output.summary_text,
        transcript_path=str(output.source_session),
    )

    preview_lines = [_preview_line(line) for line in output.preserved_tail[:20]]
    preview_note = (
        f"{len(output.preserved_tail)} preserved messages in the final "
        f"pass' residual. Not inlined in full (see source transcript); "
        f"preview of the first {len(preview_lines)} below:"
    )

    return "\n".join([
        "# Compact Summary",
        "",
        "## Metadata",
        *metadata_lines,
        "",
        "### Passes",
        "",
        *_pass_table(output.passes),
        "",
        "### Preserved Segment UUIDs",
        "",
        "Full ordered uuid list of the preserved tail, for a future JSONL-append phase to consume programmatically:",
        "",
        "```json",
        json.dumps(output.preserved_segment.uuids),
        "```",
        "",
        "## Summary",
        "",
        response.clean_summary(output.summary_text),
        "",
        "## Resume Preamble",
        "",
        "Verbatim Vq-wrapper text, for pasting into a fresh session or "
        "reconstructing a future JSONL isCompactSummary append:",
        "",
        "```",
        resume_preamble,
        "```",
        "",
        "## Preserved Tail",
        "",
        preview_note,
        *preview_lines,
        "",
    ])


def write_markdown(output: CompactionOutput, output_path: Path | None = None) -> Path:
    path = output_path or default_output_path(output.source_session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(output), encoding="utf-8")
    return path
