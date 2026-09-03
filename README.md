# cc-local-compact

MCP server replicating Claude Code's `/compact` against an external, locally-hosted model.

See `cc_local_compact/README.md` for the full description, configuration, and file layout, and `docs/compact-architecture.md` in the `cc-patcher` repo for the reverse-engineering writeup this implementation is built from.

## Install

```bash
pip install -e .[dev]
# or, as an installed CLI tool:
uv tool install /path/to/cc-local-compact
```

## Use

```bash
cc-local-compact list                           # session transcripts for the current project
cc-local-compact compact                        # compact the current session (see "Which session gets compacted" below)
cc-local-compact compact path/to/session.jsonl   # compact a specific session
cc-local-compact serve                           # run as an MCP stdio server
cc-local-compact register                        # install the bare /remind command + hook into ~/.claude (see below)
```

Register as an MCP server via `.mcp.json` (see the example in this repo) to expose `compact_session`/`list_sessions` as tools inside a Claude Code session.

`/remind` (installed by `cc-local-compact register`) is the way to recover context after typing `/clear` in a live session -- it's a plain user-level slash command + hook, deliberately not an MCP tool or a plugin. See `cc_local_compact/README.md`, "Recovering after a manual `/clear`", for why.
