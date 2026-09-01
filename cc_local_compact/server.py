"""FastMCP stdio server exposing compact_session and list_sessions."""

from pathlib import Path

import anthropic
from fastmcp import FastMCP

from . import client, config as config_mod, discovery, loop, markdown_out, response, tokens, transcript

mcp = FastMCP("cc-local-compact")


def _build_client(cfg: config_mod.Config) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=cfg.base_url, api_key=cfg.api_key)


def _run_compaction(
    session_path: Path,
    custom_instructions: str | None,
    context_budget: int | None,
    model: str | None,
    output_path: Path | None,
) -> dict:
    cfg = config_mod.load_config()
    resolved_model = model or cfg.model
    resolved_budget = context_budget or cfg.context_budget
    api_client = _build_client(cfg)

    lines = transcript.load_transcript(session_path)
    pre_tokens = tokens.estimate_transcript_tokens(lines)

    loop_config = loop.LoopConfig(
        context_budget=resolved_budget,
        response_max_tokens=cfg.response_max_tokens,
    )

    def summarize_fn(summarize_set, instructions, strip_media_flag):
        return client.summarize_group(
            api_client, resolved_model, summarize_set, instructions,
            strip_media_flag, cfg.response_max_tokens,
        )

    result = loop.run_loop(lines, summarize_fn, loop_config, custom_instructions)

    if not result.ok:
        return {
            "ok": False,
            "reason": result.reason,
            "attempts": result.attempts,
            "total_groups": result.total_groups,
            "detail": result.detail,
        }

    preserved = result.messages_to_preserve or []
    preserved_uuids = [line["uuid"] for line in preserved if "uuid" in line]
    segment = markdown_out.PreservedSegment(
        head_uuid=preserved_uuids[0] if preserved_uuids else None,
        anchor_uuid=preserved_uuids[0] if preserved_uuids else None,
        tail_uuid=preserved_uuids[-1] if preserved_uuids else None,
        uuids=preserved_uuids,
    )
    cleaned_summary = response.clean_summary(result.summary_text)
    post_tokens = tokens.count_chars_over_n(cleaned_summary)

    output = markdown_out.CompactionOutput(
        source_session=session_path,
        trigger="manual",
        model=resolved_model,
        backend_base_url=cfg.base_url,
        total_groups=result.total_groups,
        groups_summarized=result.total_groups - result.groups_preserved,
        groups_preserved=result.groups_preserved,
        attempts=result.attempts,
        context_budget=resolved_budget,
        pre_tokens_estimate=pre_tokens,
        post_tokens_estimate=post_tokens,
        custom_instructions=custom_instructions,
        summary_text=result.summary_text,
        preserved_tail=preserved,
        preserved_segment=segment,
    )
    written_path = markdown_out.write_markdown(output, output_path)

    return {
        "ok": True,
        "output_path": str(written_path),
        "summary": cleaned_summary,
        "groups_summarized": output.groups_summarized,
        "groups_preserved": output.groups_preserved,
        "attempts": result.attempts,
        "pre_tokens_estimate": pre_tokens,
        "post_tokens_estimate": post_tokens,
    }


@mcp.tool
def compact_session(
    session_path: str | None = None,
    custom_instructions: str | None = None,
    context_budget: int | None = None,
    model: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Summarize a Claude Code session transcript against the local model,
    writing a markdown summary file. If session_path is omitted, discovers
    the most recently modified .jsonl for the current working directory's
    project."""
    resolved_session = (
        Path(session_path) if session_path else discovery.most_recent_session(Path.cwd())
    )
    if resolved_session is None:
        return {
            "ok": False,
            "reason": "no_session_found",
            "detail": f"no session found for cwd {Path.cwd()}",
        }
    return _run_compaction(
        resolved_session,
        custom_instructions,
        context_budget,
        model,
        Path(output_path) if output_path else None,
    )


@mcp.tool
def list_sessions(project_cwd: str | None = None) -> list[dict]:
    """List available session transcripts (path, mtime, size) for the
    given or current working directory's project."""
    cwd = Path(project_cwd) if project_cwd else Path.cwd()
    return discovery.list_sessions(cwd)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
