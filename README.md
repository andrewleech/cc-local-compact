# cc-local-compact

MCP server replicating Claude Code's `/compact` against an external, locally-hosted model.

See `cc_local_compact/README.md` for the full description, configuration, and file layout, and `docs/compact-architecture.md` in the `cc-patcher` repo for the reverse-engineering writeup this implementation is built from.

## Install

```bash
pip install -e .[dev]
```

## Use

```bash
cc-local-compact list                           # session transcripts for the current project
cc-local-compact compact                        # compact the most recent session
cc-local-compact compact path/to/session.jsonl   # compact a specific session
cc-local-compact serve                           # run as an MCP stdio server
```

Register as an MCP server via `.mcp.json` (see the example in this repo) to expose `compact_session`/`list_sessions` as tools inside a Claude Code session.
