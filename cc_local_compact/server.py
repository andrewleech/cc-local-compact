"""FastMCP stdio server exposing compact_session and list_sessions."""

import time
from pathlib import Path

import anthropic
from fastmcp import FastMCP

from . import (
    client, config as config_mod, discovery, fallback, jsonl_append, loop,
    markdown_out, multipass, response, tokens, transcript,
)

mcp = FastMCP("cc-local-compact")


def _build_client(cfg: config_mod.Config) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=cfg.base_url, api_key=cfg.api_key)


def _run_compaction(
    session_path: Path,
    custom_instructions: str | None,
    context_budget: int | None,
    model: str | None,
    output_path: Path | None,
    fallback_model: str | None = None,
    append_to_jsonl: bool = False,
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

    if fallback_model:
        def fallback_summarize_fn(summarize_set, instructions, strip_media_flag):
            return client.summarize_group(
                api_client, fallback_model, summarize_set, instructions,
                strip_media_flag, cfg.response_max_tokens,
            )
        summarize_fn = fallback.with_fallback(summarize_fn, fallback_summarize_fn)

    start_time = time.monotonic()
    result = multipass.run_multi_pass(lines, summarize_fn, loop_config, custom_instructions)
    duration_ms = int((time.monotonic() - start_time) * 1000)

    if not result.ok:
        return {
            "ok": False,
            "reason": result.reason,
            "detail": result.detail,
            "passes": len(result.passes),
        }

    preserved = result.final_preserved_tail or []
    preserved_uuids = [line["uuid"] for line in preserved if "uuid" in line]
    segment = markdown_out.PreservedSegment(
        head_uuid=preserved_uuids[0] if preserved_uuids else None,
        anchor_uuid=preserved_uuids[0] if preserved_uuids else None,
        tail_uuid=preserved_uuids[-1] if preserved_uuids else None,
        uuids=preserved_uuids,
    )
    cleaned_summary = response.clean_summary(result.final_summary_text)
    post_tokens = tokens.count_chars_over_n(cleaned_summary) + tokens.estimate_group_tokens(preserved)

    output = markdown_out.CompactionOutput(
        source_session=session_path,
        trigger="manual",
        model=resolved_model,
        backend_base_url=cfg.base_url,
        context_budget=resolved_budget,
        passes=result.passes,
        multi_pass_reason=result.reason,
        pre_tokens_estimate=pre_tokens,
        post_tokens_estimate=post_tokens,
        custom_instructions=custom_instructions,
        summary_text=result.final_summary_text,
        preserved_tail=preserved,
        preserved_segment=segment,
    )
    written_path = markdown_out.write_markdown(output, output_path)

    response_dict = {
        "ok": True,
        "output_path": str(written_path),
        "summary": cleaned_summary,
        "passes": len(result.passes),
        "fallback_passes": sum(1 for p in result.passes if p.used_fallback),
        "multi_pass_reason": result.reason,
        "pre_tokens_estimate": pre_tokens,
        "post_tokens_estimate": post_tokens,
    }

    if append_to_jsonl:
        try:
            append_result = jsonl_append.append_compaction(
                session_path=session_path,
                summary_text=result.final_summary_text,
                preserved_tail=preserved,
                trigger="manual",
                pre_tokens=pre_tokens,
                post_tokens=post_tokens,
                duration_ms=duration_ms,
            )
            response_dict["jsonl_appended"] = True
            response_dict["jsonl_boundary_uuid"] = append_result.boundary_uuid
            response_dict["jsonl_note"] = (
                "Boundary appended to the session's own JSONL in the same shape "
                "the real /compact produces, for on-disk record consistency. "
                "This does NOT reduce what gets sent to the remote model on the "
                "next resumed turn -- confirmed an externally-appended boundary "
                "is not respected by the client. See jsonl_append.py."
            )
        except Exception as error:
            response_dict["jsonl_appended"] = False
            response_dict["jsonl_append_error"] = str(error)

    return response_dict


@mcp.tool
def compact_session(
    session_path: str | None = None,
    custom_instructions: str | None = None,
    context_budget: int | None = None,
    model: str | None = None,
    output_path: str | None = None,
    fallback_model: str | None = None,
    append_to_jsonl: bool = False,
) -> dict:
    """Summarize a Claude Code session transcript against the local model,
    writing a markdown summary file. If session_path is omitted, discovers
    the most recently modified .jsonl for the current working directory's
    project. If fallback_model is given, any pass whose response fails a
    structural sanity check (see validate.py) is retried against that
    model instead -- lets a fast/small primary model handle most passes
    while a slower/larger model only gets used where actually needed.

    If append_to_jsonl is True, also appends a compact_boundary +
    isCompactSummary sequence directly to session_path's own JSONL, in the
    same shape the real /compact produces (see jsonl_append.py). This is
    for on-disk record consistency only -- confirmed it does NOT reduce
    what gets sent to the remote model on the next resumed turn (an
    externally-appended boundary is not respected by the client). Not
    recommended on a session the real Claude Code client currently has
    open; safe to use on a session that's closed or won't be resumed
    imminently."""
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
        fallback_model,
        append_to_jsonl,
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
