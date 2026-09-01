# cc-local-compact

Replicates Claude Code's `/compact` context-summarization against an external, locally-hosted Anthropic-protocol-compatible model, operating on a session's own `.jsonl` transcript instead of live in-app state.

Written from a full reverse-engineering pass on Claude Code 2.1.252's `/compact`, documented in `docs/compact-architecture.md` in the `cc-patcher` repo. Every module here is a deliberately mechanical port of the corresponding function(s) documented there, so the mapping stays auditable against the original minified source.

## Why a local model needs this

The real `/compact` discovers Claude's context limit reactively, via live API errors, because Claude's window is large enough that this rarely happens more than once or twice. A local model (Qwen3.8, Gemma-4, ...) has a much smaller window, so this tool proactively sizes each summarization batch against a configured `context_budget` before ever calling the model, and keeps the real app's reactive retry/backoff as a safety net for when the token estimate is wrong.

## v1 scope

Writes a standalone markdown summary file. Does **not** append a `compact_boundary` marker into the live session `.jsonl` -- mutating a session file the real Claude Code client may have open is untested and out of scope for this phase. See `docs/compact-architecture.md`, "Decision for the first implementation".

## Backend

Talks directly to the local inference backend via the `anthropic` Python SDK with a custom `base_url` -- the backend (`llama-swap` on `titan:8080`, serving Qwen3.8/Gemma-4 GGUF models) speaks the Anthropic `/v1/messages` protocol, so grouped transcript messages go through close to verbatim with no lossy translation layer. Does not route through the `cc-local-router` proxy process, though it defaults to reading the same `CLAUDE_NET_PROXY_LOCAL_URL`/`CLAUDE_NET_PROXY_LOCAL_MODEL` environment variables that proxy already uses, so there's one place to keep the backend address in sync.

## Quick start

```bash
pip install -e .[dev]
cc-local-compact list                          # session transcripts for the current project
cc-local-compact compact --context-budget 24000 # compact the most recent session
```

## Configuration

Environment variables, first non-empty wins per pair:

- `CC_LOCAL_COMPACT_BASE_URL` / `CLAUDE_NET_PROXY_LOCAL_URL` (default `http://titan:8080`)
- `CC_LOCAL_COMPACT_MODEL` / `CLAUDE_NET_PROXY_LOCAL_MODEL` (default `qwen3.8-27b`)
- `CC_LOCAL_COMPACT_API_KEY` / `ANTHROPIC_AUTH_TOKEN` (default `local`)
- `CC_LOCAL_COMPACT_CONTEXT_BUDGET` (default `32000`)
- `CC_LOCAL_COMPACT_RESPONSE_MAX_TOKENS` (default `4096`)

## File layout

```
cc_local_compact/
  transcript.py     JSONL parsing, parentUuid-chain reconstruction
  grouping.py        Mq/n$ port: assistant-turn grouping
  tokens.py           Nc/nt/DHe/ljn/th/xh port: chars/4 token estimation
  prompts.py          OBn/hCt/Gse port: the verbatim summarization prompt
  client.py           NBn/rw port: model client, response classification
  loop.py             ZZ/_Ct/$Bn port: budget-seeded summarize/preserve loop
  response.py          FBn/Vq port: post-processing, resume-preamble wrapper
  markdown_out.py       output file structure/writer
  discovery.py           cwd -> project-dir slug -> most recent session
  server.py               FastMCP tool surface (stdio)
  cli.py                   standalone CLI
config.py                  environment-variable resolution
```

## Known deviations from the source

- No `PreCompact`/`PostCompact` hook dispatch (a later phase).
- No precompute cache (no live session to speculate against).
- Overflow detection doesn't use Anthropic's own error shape (`actualTokens`/`limitTokens`) as the primary path -- confirmed against the live backend (llama-swap on titan:8080) that it emits its own `{"type": "exceed_context_size_error", "n_prompt_tokens", "n_ctx"}` shape instead, which `client.py`'s `_classify_error` now parses first. See `loop.py`'s module docstring.

## See also

- `docs/compact-architecture.md` in the `cc-patcher` repo -- the full reverse-engineering writeup this implementation is built from.
