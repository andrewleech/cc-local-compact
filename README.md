# cc-local-compact

Replicates Claude Code's `/compact` context-summarization against an external, locally-hosted model (Qwen, Gemma, ... anything speaking the Anthropic `/v1/messages` protocol), so a session can be compacted without sending its content to Claude. Works directly on a session's own `.jsonl` transcript, not live in-app state, so it runs as a standalone CLI, an MCP server, or a hook; no dependency on what a running Claude Code session currently has in context.

Built from a full reverse-engineering pass on Claude Code's own `/compact` implementation; every module is a deliberately mechanical port of the corresponding real function, kept auditable against the source. See `docs/compact-architecture.md` in the `cc-patcher` repo for that writeup, and `cc_local_compact/README.md` for this project's own architecture, configuration, and design rationale in full.

## Install

```bash
pip install -e .[dev]
# or, as an installed CLI tool:
uv tool install /path/to/cc-local-compact
```

## Quick start

```bash
cc-local-compact list                           # session transcripts for the current project
cc-local-compact compact                        # compact the current session (see "Which session gets compacted" in cc_local_compact/README.md)
cc-local-compact compact path/to/session.jsonl   # compact a specific session
cc-local-compact serve                           # run as an MCP stdio server
```

Register as an MCP server via `.mcp.json` (see the example in this repo) to expose `compact_session`/`list_sessions` as tools inside a Claude Code session.

### Recovering context after `/clear`

```bash
cc-local-compact register     # one-time: installs the bare /remind command + its hook into ~/.claude
```

Then, in any Claude Code session: type `/clear`, then `/remind`. `/remind` is a plain user-level slash command + hook this installs, deliberately not an MCP tool or a plugin (both were tried; see `cc_local_compact/README.md`, "Recovering after a manual `/clear`", for why neither works for this). `cc-local-compact unregister` removes it.

## Documentation

- `cc_local_compact/README.md`, full architecture: multi-pass compaction, model reliability/fallback, output formats, session resolution, the `/remind` mechanism, configuration, file layout.
- `docs/compact-architecture.md` in the `cc-patcher` repo, the reverse-engineering writeup this implementation is built from.
