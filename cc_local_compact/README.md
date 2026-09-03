# cc-local-compact

Replicates Claude Code's `/compact` context-summarization against an external, locally-hosted Anthropic-protocol-compatible model, operating on a session's own `.jsonl` transcript instead of live in-app state.

Written from a full reverse-engineering pass on Claude Code 2.1.252's `/compact`, documented in `docs/compact-architecture.md` in the `cc-patcher` repo. Every module here is a deliberately mechanical port of the corresponding function(s) documented there, so the mapping stays auditable against the original minified source.

## Why a local model needs this

The real `/compact` discovers Claude's context limit reactively, via live API errors, because Claude's window is large enough that this rarely happens more than once or twice. A local model (Qwen3.8, Gemma-4, ...) has a much smaller window, so this tool proactively sizes each summarization batch against a configured `context_budget` before ever calling the model, and keeps the real app's reactive retry/backoff as a safety net for when the token estimate is wrong.

## Multi-pass compaction

A single pass (`loop.run_loop`, the port of the real app's `ZZ` loop) groups the input oldest-to-newest, estimates each group's token size, and proactively decides a split: how many of the *newest* groups to preserve untouched so that everything older than the split fits inside `context_budget`. It starts from "preserve almost nothing" and only grows the preserved count as far as needed to fit, correcting itself reactively if the estimate was wrong (a real `prompt_too_long`/`exceed_context_size_error` from the backend). One pass sends the oldest portion to the model once and returns a summary of that portion plus the newest portion completely untouched (`messages_to_preserve`).

That untouched tail can still be huge. On this project's own 488K-token benchmark session, a single Gemma-4-12b pass (128K window) summarized only the oldest 22 of 211 groups, leaving 106K+ tokens of the most recent conversation completely unprocessed. A session that large needs more than one pass no matter how generous `context_budget` is, unless the model's window happens to comfortably exceed the whole session (which is exactly why real Claude Code's own `/compact` usually gets away with a single pass: its context window is often large enough to hold nearly the whole conversation at once).

`multipass.run_multi_pass` chains passes to close that gap. After a pass completes, it checks whether that pass' entire preserved tail would itself fit as input to one more pass (`estimate_group_tokens(preserved) <= config.usable_budget()`). If yes, it's done. If not, it builds the next pass' input as `[synthetic summary-of-this-pass message, *the whole preserved tail]` (`multipass._residual_lines`) and runs the identical single-pass logic again on that combined input: re-group it, re-estimate it, and let the same proactive-seeding math decide how much of *this* residual's oldest portion (which now starts with the prior summary) fits in one more call. Repeat until the leftover tail fits in a hypothetical next pass, a pass can't compact anything further (`too_few_groups`), or `max_passes` is hit.

The important distinction from a naive "append the next chunk of raw lines to a running summary" design: each pass is handed the *entire* remaining untouched tail, however large that still is, not a fixed-size next slice. How far the boundary actually creeps forward is decided fresh by that pass' own budget-seeding math, so passes don't shrink by even steps. On the same 488K-token benchmark, Gemma-4-12b's corrected 10-pass run left this many groups still preserved (untouched) after each pass: 189, 181, 174, 162, 152, 141, 124, 102, 65, 35 -- converging, not stepping down by a constant amount, because each pass re-evaluates the actual token weight of what's left rather than assuming a fixed batch size.

The synthetic summary message injected at the front of each residual deliberately does *not* use the real app's `Vq`/`build_resume_preamble` "this session is being continued... continue the conversation from where it left off" framing, even though that's the literal wording Claude Code's own client injects for a real `isCompactSummary` message. That framing tells an assistant to *resume a live task*, which is the wrong implication for something being fed back in as input to another summarization pass, so `_residual_lines` uses a plain "the following is a summary of earlier conversation history, already compacted in a prior pass" label instead, on principle. This does **not**, however, explain the real `qwen3.5-9b` failures described below: the same failure reproduced even feeding the exact same raw content through a single standalone call with no synthetic preamble and no multi-pass framing at all, proving the cause is content-specific, not this framing choice. See "Model reliability and fallback" below.

`custom_instructions` (the equivalent of `/compact <text>`) only apply to the first pass. Later passes are summarizing this tool's own prior output, not fresh user-requested content, so repeating the same instructions across every pass would misapply them to context the user never saw when they wrote them.

If a later pass fails outright (the backend errors, or the model returns something unusable), `run_multi_pass` doesn't discard the work already done: it returns the prior pass' result with `reason: "later_pass_failed"` rather than failing the whole operation. Only a first-pass failure with no prior progress returns `ok: False`.

## Model reliability and fallback

A smaller/faster model isn't uniformly reliable, and `validate.py`/`fallback.py` exist because this was found the hard way, not designed in speculatively. An earlier AWQ-int4 build served under the `qwen3.5-9b` id ran roughly 13x faster wall-clock than `qwen3.8-27b` on this project's own 488K-token benchmark and handled most content well, but four independently-reproduced real runs against a specific residual (the tail of that same benchmark, dense with self-referential/meta-technical content -- a conversation about building a compaction tool, containing raw `tool_result` JSON about compaction itself) produced four different garbled non-summaries: cut off mid-word with no tags, empty, prose that tried to continue the *original* task instead of compacting, and a hallucinated "this data is fake" preamble. `qwen3.8-27b` never showed this failure on the same content. Isolating the content from the multi-pass framing (feeding it through a single standalone call, no synthetic preamble at all) reproduced the same failure, confirming it was about *that specific content* defeating that model's instruction-following, not about multi-pass chaining or message framing.

A live A/B against a Q6 GGUF build on the identical failing input cleared it completely, root-causing the failures to the int4 quantization rather than the 9B parameter count. `qwen3.5-9b` was then cut over to that Q6 build (the AWQ deployment is fully retired, deleted from the backend catalog), and is now this tool's default model: on a full corrected benchmark run it matched `qwen3.8-27b`'s reliability and compaction quality (0 fallbacks needed, same group split both passes) at ~5.8x the speed (~2m48s vs ~16m14s for the same task). See `config.DEFAULT_MODEL`'s docstring for the numbers.

The validation/fallback mechanism stays in the codebase regardless, as general infrastructure for whichever model or backend hits this next: all four observed AWQ failures shared one structural signal -- none of them produced the `<summary>` tag wrapping the 9 numbered sections the prompt explicitly asks for (`prompts.INSTRUCTIONS_BODY`). `validate.is_well_formed_summary` checks for exactly that -- non-empty, contains `<summary>`, and has at least 5 of the 9 numbered section headers -- as a cheap, deterministic "did this even attempt the right shape" gate, not a semantic quality judgment.

`fallback.with_fallback(primary, fallback)` composes two `SummarizeFn`s into one: call `primary` once, and only if it fails outright or its output fails `is_well_formed_summary`, call `fallback` instead. `loop.py`/`multipass.py` never know two models are involved -- they see one `SummarizeFn` either way. This means a fast/cheap primary model can handle whichever passes it's actually capable of, and a slower/more reliable model only gets invoked on the specific batches that need it, rather than distrusting the fast model globally because of one bad case. Enable it with `--fallback-model` (CLI) or the `fallback_model` param (MCP `compact_session` tool); `AttemptResult.used_fallback` and each `PassRecord.used_fallback` record whether it actually fired, surfaced in the output markdown's pass table and the `fallback_passes` count in the tool/CLI result.

## Output: markdown by default, optional JSONL append

Always writes a standalone markdown summary file. Optionally (`--append-jsonl` / `append_to_jsonl=True`) also appends a `compact_boundary` + `isCompactSummary` + re-chained preserved-tail sequence directly into the session's own `.jsonl` (`jsonl_append.py`), in the exact shape real Claude Code's `/compact` produces (see `docs/compact-architecture.md`, "Confirmed on-disk JSONL schema").

**This does not reduce cost on the session's next resumed turn.** That was the original motivation for wanting it (avoid resending a session whose prompt cache has already expired), and it was tested directly and found not to work: a well-formed boundary injected into a live throwaway session's JSONL, *before that session was ever resumed*, had no effect -- the next turn reprocessed the entire original conversation, token-for-token indistinguishable from a normal resume with no boundary at all. Separately confirmed via Claude Code's own documentation (`cross-session-messaging.md`): slash commands arriving through any channel other than direct interactive terminal keystrokes -- a session's own JSONL, MCP tool/prompt output, Remote Control, the local cross-session messaging socket -- are never executed. There is no external mechanism to trigger the real client's `/compact` or `/clear`, or to make it reduce what it sends on a future turn.

What appending *is* confirmed to do: it's safe. Directly tested against a live throwaway session (write the sequence, then `claude --resume` it, repeated across several trials) with no corruption and no errors -- the client just ignores the boundary and continues normally. So this exists purely for **on-disk record consistency**: a session's own transcript reflects that a compaction happened, in the real schema, for `/resume` picker display or any other tooling that parses this format -- not as a way to make the next turn cheaper. Avoid using it on a session the real Claude Code client currently has open, since concurrent-write behavior against a live client process hasn't been tested (only against a closed/idle session file).

## Recovering after a manual `/clear`

Nothing external can trigger the real client's own `/clear`/`/compact`, or make it send less to the model on a future turn -- see "Output" above for the confirmed test of that. `/clear` itself is free (no model call) and doesn't rotate the session file or ID -- confirmed by inspecting a real transcript containing one: it's recorded in-place as an ordinary `type:"user"` line, and the `parentUuid` chain runs straight across it unbroken. So the only usable "compact and continue" pattern is: a human runs `/clear` themselves, then the freshly-cleared agent calls `continue_after_clear` to recover a summary of what came before, in the same session file.

`/clear` is detected differently from `/rename` (see "Which session gets compacted" above): it's a `type:"user"` line whose `message.content` is the literal `<command-name>/clear</command-name>...` tag string, not a `type:"system",subtype:"local_command"` line. `discovery.find_clear_indices` returns every such line's index on the transcript's live main thread (via `transcript.load_transcript`, so an abandoned branch's stale `/clear` -- e.g. orphaned by a later rewind -- is correctly ignored, same principle as `describe_session`'s rename handling; confirmed against two real sessions where exactly this happened).

Because `/clear` doesn't break the `parentUuid` chain, the full session history is still reachable no matter how many times it's been cleared -- summarizing everything before the *last* `/clear` would re-summarize spans an earlier `continue_after_clear` call already returned. So only the span since the *previous* `/clear` (or since session start, for the first one) gets summarized. No `/clear` on the live thread at all returns `{"ok": false, "reason": "no_clear_boundary_found", ...}`; an empty span (`/clear` was the first thing since the prior boundary) returns `{"ok": false, "reason": "empty_pre_clear_span", ...}` -- neither silently falls back to summarizing the whole transcript.

Unlike `compact_session`, whose `summary` field is the bare cleaned text (that tool's caller/purpose is heterogeneous -- archival, review, handoff to a different session), `continue_after_clear`'s `summary` field is wrapped in `response.build_resume_preamble`'s "resume directly, don't acknowledge, don't ask questions" framing. This tool is only ever called by an agent that was just cleared and must resume acting, and its `tool_result` lands in exactly the position a real injected `isCompactSummary` message would -- there's no other channel for that framing to reach the agent through. The bare cleaned text is still available as `summary_cleaned`.

The CLI's `compact --before-last-clear` flag exercises the same slicing outside a live session, for dry-running -- it does not apply the resume-framed wrapping, since there's no live agent there to read it.

## Backend

Talks directly to the local inference backend via the `anthropic` Python SDK with a custom `base_url` -- the backend (`llama-swap` on `titan:8080`, serving Qwen3.8/Gemma-4 GGUF models) speaks the Anthropic `/v1/messages` protocol, so grouped transcript messages go through close to verbatim with no lossy translation layer. Does not route through the `cc-local-router` proxy process, though it defaults to reading the same `CLAUDE_NET_PROXY_LOCAL_URL`/`CLAUDE_NET_PROXY_LOCAL_MODEL` environment variables that proxy already uses, so there's one place to keep the backend address in sync.

## Which session gets compacted

Claude Code sets `CLAUDE_CODE_SESSION_ID` (the calling session's own UUID) in the environment of every stdio MCP server it spawns, alongside `CLAUDE_PROJECT_DIR` (which `discovery.resolve_cwd()` uses in preference to this process' own cwd). This isn't documented anywhere public, but it's confirmed directly from the installed binary's own MCP stdio spawn code (Claude Code 2.1.258): `env:{...inherited,CLAUDE_PROJECT_DIR:mn(),CLAUDE_CODE_SESSION_ID:Q(),CLAUDECODE:"1",...serverEnv}`. `Q()` is the same current-session accessor used elsewhere in the app for `${CLAUDE_SESSION_ID}` prompt substitution. Neither `claude-net`'s nor `cc-local-router`'s patcher providers inject this -- it's stock Claude Code behaviour.

So when `compact_session`/`cc-local-compact compact` is called without an explicit `session_path`, `discovery.resolve_session` first checks `CLAUDE_CODE_SESSION_ID` and, if its `.jsonl` exists in the resolved project directory, uses it directly -- an authoritative identification of the caller, not a guess, even with several sessions open on the same project. If that env var is unset/stale and exactly one session exists for the project, that one is used unambiguously by elimination.

Since this env var is undocumented and internal, treat it as liable to change across Claude Code versions -- so there has to be a real fallback for whenever it's absent (e.g. running the CLI standalone outside an MCP session, or an older Claude Code build) and more than one session exists for the project. There is deliberately **no mtime-based guess** in that fallback: silently picking "most recently modified" can pick the wrong session, and a wrong pick is worse than making the caller choose. Instead, resolution stops and hands back every candidate, each with a `display_name` from `discovery.describe_session` so they can actually be told apart:

- the session's own `/rename` title, if it was ever renamed (Claude Code records `/rename` as a `type:"system",subtype:"local_command"` line with the new title in a `<command-args>` tag -- the most reliable signal short of an explicit path), the last one used if renamed more than once;
- failing that, a condensed, one-line, terminal-width-truncated snippet of the last visible message on the transcript's main thread (text content only -- no tool calls/results/thinking) -- what a person would actually see on screen at the end of that session.

`compact_session` (the MCP tool) returns `{"ok": false, "reason": "ambiguous_session", "candidates": [...]}` without compacting anything in this case -- the calling agent has to look at the candidates and re-call with `session_path` set explicitly (or call `list_sessions` first, which returns the same enriched candidate list). `cc-local-compact compact` (the CLI) instead prompts interactively: it prints the numbered, display-named candidate list and reads a selection from stdin, refusing to guess there either; if stdin isn't a terminal (e.g. run from a script) it prints the same list and exits non-zero instead of hanging on `input()`.

## Quick start

```bash
pip install -e .[dev]
cc-local-compact list      # session transcripts for the current project
cc-local-compact compact   # resolves the session automatically, or prompts if more than one candidate exists
```

## Configuration

Environment variables, first non-empty wins per pair:

- `CC_LOCAL_COMPACT_BASE_URL` / `CLAUDE_NET_PROXY_LOCAL_URL` (default `http://titan:8080`)
- `CC_LOCAL_COMPACT_MODEL` / `CLAUDE_NET_PROXY_LOCAL_MODEL` (default `qwen3.5-9b`)
- `CC_LOCAL_COMPACT_API_KEY` / `ANTHROPIC_AUTH_TOKEN` (default `local`)
- `CC_LOCAL_COMPACT_CONTEXT_BUDGET` (default: 75% of the resolved model's real context window, per `config.MODEL_CONTEXT_WINDOWS`; falls back to 75% of 32768 for an unlisted model)
- `CC_LOCAL_COMPACT_RESPONSE_MAX_TOKENS` (default: 30% of the resolved `context_budget`, floored at 8192)

## File layout

```
cc_local_compact/
  transcript.py     JSONL parsing, parentUuid-chain reconstruction
  grouping.py        Mq/n$ port: assistant-turn grouping
  tokens.py           Nc/nt/DHe/ljn/th/xh port: chars/4 token estimation
  prompts.py          OBn/hCt/Gse port: the verbatim summarization prompt
  client.py           NBn/rw port: model client, response classification
  loop.py             ZZ/_Ct/$Bn port: budget-seeded summarize/preserve loop
  multipass.py         chains loop.run_loop passes for sessions too large for one
  response.py          FBn/Vq port: post-processing, resume-preamble wrapper
  validate.py            structural sanity-check on a raw response, gates fallback
  fallback.py              composes a primary+fallback model pair into one SummarizeFn
  markdown_out.py            output file structure/writer
  jsonl_append.py              optional: append a compact_boundary to the session's own JSONL
  discovery.py                   cwd -> project-dir slug -> session (via CLAUDE_CODE_SESSION_ID or interactive/agent disambiguation), plus /clear-boundary detection for continue_after_clear
  server.py                         FastMCP tool surface (stdio)
  cli.py                               standalone CLI
config.py                                environment-variable resolution
```

## Known deviations from the source

- No `PreCompact`/`PostCompact` hook dispatch (a later phase).
- No precompute cache (no live session to speculate against).
- Overflow detection doesn't use Anthropic's own error shape (`actualTokens`/`limitTokens`) as the primary path -- confirmed against the live backend (llama-swap on titan:8080) that it emits its own `{"type": "exceed_context_size_error", "n_prompt_tokens", "n_ctx"}` shape instead, which `client.py`'s `_classify_error` now parses first. See `loop.py`'s module docstring.

## See also

- `docs/compact-architecture.md` in the `cc-patcher` repo -- the full reverse-engineering writeup this implementation is built from.
