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
    resolution_meta: dict | None = None,
    lines: list[dict] | None = None,
    trigger: str = "manual",
    logical_parent_uuid_override: str | None = None,
) -> dict:
    """Shared compaction pipeline for compact_session and
    continue_after_clear. By default operates on session_path's full
    transcript (trigger "manual", boundary anchored to the file's literal
    last line) -- compact_session and the CLI rely on exactly this
    default and pass none of the three trailing params. `lines`,
    `trigger`, and `logical_parent_uuid_override` exist solely for
    continue_after_clear, which compacts only a pre-/clear span rather
    than the whole file and must anchor any appended boundary to that
    span's own last line, not the file's true end (see
    jsonl_append.append_compaction's docstring)."""
    cfg = config_mod.load_config()
    resolved_model = model or cfg.model
    resolved_budget = context_budget or cfg.context_budget
    api_client = _build_client(cfg)

    lines = lines if lines is not None else transcript.load_transcript(session_path)
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

    # By the time _run_compaction runs, session_path is always a definite
    # choice -- never a guess. compact_session returns early on an
    # "ambiguous" resolution instead of calling this with one (see below);
    # the CLI resolves ambiguity interactively before calling this too.
    resolution_fields = {"session_path": str(session_path), **(resolution_meta or {"source": "explicit"})}

    start_time = time.monotonic()
    result = multipass.run_multi_pass(lines, summarize_fn, loop_config, custom_instructions)
    duration_ms = int((time.monotonic() - start_time) * 1000)

    if not result.ok:
        return {
            "ok": False,
            "reason": result.reason,
            "detail": result.detail,
            "passes": len(result.passes),
            **resolution_fields,
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
        trigger=trigger,
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
        **resolution_fields,
    }

    if append_to_jsonl:
        try:
            append_result = jsonl_append.append_compaction(
                session_path=session_path,
                summary_text=result.final_summary_text,
                preserved_tail=preserved,
                trigger=trigger,
                pre_tokens=pre_tokens,
                post_tokens=post_tokens,
                duration_ms=duration_ms,
                logical_parent_uuid_override=logical_parent_uuid_override,
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
    writing a markdown summary file.

    session_path identifies WHICH session gets compacted. If omitted, this
    resolves it from the CLAUDE_CODE_SESSION_ID environment variable that
    Claude Code injects into every stdio MCP server's own environment (see
    discovery.py's module docstring for how this was confirmed) -- that
    identifies the calling session directly, not a guess, even with more
    than one Claude Code window open on the same project. If that env var
    is unset/stale and exactly one session exists for the project, that
    one is used unambiguously by elimination.

    If neither applies and more than one session transcript exists for the
    project, this does NOT guess (no mtime-based fallback) -- it returns
    {"ok": false, "reason": "ambiguous_session", "candidates": [...]}
    without compacting anything. Each candidate carries path/mtime/size
    plus a display_name (the session's own `/rename` title if it has one,
    otherwise a condensed snippet of its last visible message -- see
    discovery.describe_session) so the candidates can actually be told
    apart; re-call with session_path set to the right one's path. Calling
    list_sessions first gets the same candidate list without attempting a
    compaction.

    If fallback_model is given, any pass whose response fails a
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
    resolved_cwd = discovery.resolve_cwd()
    resolved_session, resolution_meta = discovery.resolve_session(session_path, resolved_cwd)
    if resolved_session is None:
        if resolution_meta.get("source") == "ambiguous":
            return {
                "ok": False,
                "reason": "ambiguous_session",
                "detail": (
                    f"{resolution_meta['candidate_count']} session transcripts exist for this project "
                    "and none could be identified as the caller automatically -- pick one from "
                    "candidates and re-call with session_path set to its path."
                ),
                "candidates": resolution_meta["candidates"],
            }
        return {
            "ok": False,
            "reason": "no_session_found",
            "detail": f"no session found for project directory {resolved_cwd}",
        }
    return _run_compaction(
        resolved_session,
        custom_instructions,
        context_budget,
        model,
        Path(output_path) if output_path else None,
        fallback_model,
        append_to_jsonl,
        resolution_meta,
    )


@mcp.tool
def list_sessions(project_cwd: str | None = None) -> list[dict]:
    """List available session transcripts (path, mtime, size, display_name,
    display_name_source, newest first for display only) for the given
    project directory, or the resolved current one (CLAUDE_PROJECT_DIR if
    set, else this process' own cwd). display_name is the session's own
    `/rename` title if it has one, otherwise a condensed snippet of its
    last visible message (see discovery.describe_session) -- use it to
    tell candidates apart. Call this before compact_session when more than
    one session might exist for a project, to pick session_path explicitly
    -- compact_session refuses to guess when it can't identify the caller
    and more than one candidate exists."""
    cwd = Path(project_cwd) if project_cwd else discovery.resolve_cwd()
    return discovery.list_sessions(cwd)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
