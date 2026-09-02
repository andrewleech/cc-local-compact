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

A smaller/faster model isn't uniformly reliable. `qwen3.5-9b` (9B, AWQ-4bit) handles most content well and runs roughly 13x faster wall-clock than `qwen3.8-27b` on this project's own 488K-token benchmark, but four independently-reproduced real runs against a specific residual (the tail of that same benchmark, which happens to be dense with self-referential/meta-technical content -- a conversation about building a compaction tool, containing raw `tool_result` JSON about compaction itself) produced four different garbled non-summaries: cut off mid-word with no tags, empty, prose that tried to continue the *original* task instead of compacting, and a hallucinated "this data is fake" preamble. `qwen3.8-27b` never showed this failure on the same content. Isolating the content from the multi-pass framing (feeding it through a single standalone call, no synthetic preamble at all) reproduced the same failure, confirming it's about *this specific content* defeating a 9B model's instruction-following, not about multi-pass chaining or message framing.

All four observed failures shared one structural signal: none of them produced the `<summary>` tag wrapping the 9 numbered sections the prompt explicitly asks for (`prompts.INSTRUCTIONS_BODY`). `validate.is_well_formed_summary` checks for exactly that -- non-empty, contains `<summary>`, and has at least 5 of the 9 numbered section headers -- as a cheap, deterministic "did this even attempt the right shape" gate, not a semantic quality judgment.

`fallback.with_fallback(primary, fallback)` composes two `SummarizeFn`s into one: call `primary` once, and only if it fails outright or its output fails `is_well_formed_summary`, call `fallback` instead. `loop.py`/`multipass.py` never know two models are involved -- they see one `SummarizeFn` either way. This means a fast/cheap primary model can handle whichever passes it's actually capable of, and a slower/more reliable model only gets invoked on the specific batches that need it, rather than distrusting the fast model globally because of one bad case. Enable it with `--fallback-model` (CLI) or the `fallback_model` param (MCP `compact_session` tool); `AttemptResult.used_fallback` and each `PassRecord.used_fallback` record whether it actually fired, surfaced in the output markdown's pass table and the `fallback_passes` count in the tool/CLI result.

## v1 scope

Writes a standalone markdown summary file. Does **not** append a `compact_boundary` marker into the live session `.jsonl` -- mutating a session file the real Claude Code client may have open is untested and out of scope for this phase. See `docs/compact-architecture.md`, "Decision for the first implementation".

## Backend

Talks directly to the local inference backend via the `anthropic` Python SDK with a custom `base_url` -- the backend (`llama-swap` on `titan:8080`, serving Qwen3.8/Gemma-4 GGUF models) speaks the Anthropic `/v1/messages` protocol, so grouped transcript messages go through close to verbatim with no lossy translation layer. Does not route through the `cc-local-router` proxy process, though it defaults to reading the same `CLAUDE_NET_PROXY_LOCAL_URL`/`CLAUDE_NET_PROXY_LOCAL_MODEL` environment variables that proxy already uses, so there's one place to keep the backend address in sync.

## Quick start

```bash
pip install -e .[dev]
cc-local-compact list      # session transcripts for the current project
cc-local-compact compact   # compact the most recent session, budget derived from the model
```

## Configuration

Environment variables, first non-empty wins per pair:

- `CC_LOCAL_COMPACT_BASE_URL` / `CLAUDE_NET_PROXY_LOCAL_URL` (default `http://titan:8080`)
- `CC_LOCAL_COMPACT_MODEL` / `CLAUDE_NET_PROXY_LOCAL_MODEL` (default `qwen3.8-27b`)
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
  discovery.py                 cwd -> project-dir slug -> most recent session
  server.py                       FastMCP tool surface (stdio)
  cli.py                             standalone CLI
config.py                              environment-variable resolution
```

## Known deviations from the source

- No `PreCompact`/`PostCompact` hook dispatch (a later phase).
- No precompute cache (no live session to speculate against).
- Overflow detection doesn't use Anthropic's own error shape (`actualTokens`/`limitTokens`) as the primary path -- confirmed against the live backend (llama-swap on titan:8080) that it emits its own `{"type": "exceed_context_size_error", "n_prompt_tokens", "n_ctx"}` shape instead, which `client.py`'s `_classify_error` now parses first. See `loop.py`'s module docstring.

## See also

- `docs/compact-architecture.md` in the `cc-patcher` repo -- the full reverse-engineering writeup this implementation is built from.
