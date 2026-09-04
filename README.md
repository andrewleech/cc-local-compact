# cc-local-compact

Summarise a Claude Code session using your own locally-hosted model, instead of sending the conversation to Anthropic.

Two things you get out of it:

- **`/remind`**, a slash command you type after `/clear` to pull back a summary of what you were just working on. Claude Code has no built-in equivalent; once you `/clear`, that context is gone for good.
- **Compaction of any session on demand**, from the command line or as an MCP tool, without needing the session open (or even needing Claude Code running).

Both run the summarisation against a model you host yourself, so the conversation being summarised never leaves your network.

## What you need first

An inference server that speaks the Anthropic `/v1/messages` API with a model loaded. I run llama-swap on a machine called `titan` serving Qwen3.5-9B, and those are the defaults compiled in, so unless you happen to have the same setup you'll need to point it at yours before anything works.

llama.cpp, llama-swap, vLLM behind an Anthropic-compatible shim, all fine. A real Anthropic endpoint works too if you don't care about the local part, though at that point you may as well use the built-in `/compact`.

## Install

```bash
uv tool install git+https://github.com/andrewleech/cc-local-compact
```

Pin it to a specific version if you'd rather not track the branch:

```bash
uv tool install git+https://github.com/andrewleech/cc-local-compact@<commit-or-tag>
```

Or for hacking on it:

```bash
git clone https://github.com/andrewleech/cc-local-compact
cd cc-local-compact
pip install -e .[dev]
```

## Point it at your model

Set a couple of environment variables somewhere they'll be picked up by your shell (and therefore by Claude Code when you launch it):

```bash
export CC_LOCAL_COMPACT_BASE_URL=http://your-box:8080
export CC_LOCAL_COMPACT_MODEL=your-model-name
```

That's usually all you need. The rest have sensible defaults:

| Variable | Default | What it does |
|---|---|---|
| `CC_LOCAL_COMPACT_BASE_URL` | `http://titan:8080` | Where your model server lives |
| `CC_LOCAL_COMPACT_MODEL` | `qwen3.5-9b` | Which model to ask |
| `CC_LOCAL_COMPACT_API_KEY` | `local` | Only matters if your server checks it |
| `CC_LOCAL_COMPACT_CONTEXT_BUDGET` | the model's full context window | Size of the window it plans against |
| `CC_LOCAL_COMPACT_RESPONSE_MAX_TOKENS` | 30% of the budget, minimum 8192 | How long a summary it's allowed to write |

Those are the only variables it reads. It won't pick up settings from any other tool, so there's nothing to untangle if you're running several against different backends.

The context budget is the window it plans against, not the amount of transcript it sends. It reserves room out of that for the summary the model has to write, plus a margin for its own token estimate being off, and feeds in what's left. The default works the window out from the model name where it recognises one, and drops to a conservative 32k where it doesn't; set it by hand if your server is serving a smaller window than the model nominally supports.

## Setting up `/remind`

This is the part most people actually want. One-time setup:

```bash
cc-local-compact register
```

**Then restart Claude Code.** Any session already open won't see the new command until it's genuinely restarted, and this trips people up constantly, including me. Opening a new tab is enough, `/clear` on a running session is not.

After that, in any project:

1. Work away until the context is getting full
2. `/clear`
3. Type `/remind`

You get back a summary of everything from before the `/clear`, injected as background context. Claude reads it and waits for your next instruction rather than charging off and continuing what you were doing, which is deliberate; `/clear` is something you chose to do, so it shouldn't be treated as "carry on where we left off".

You can steer the summary the same way you would with `/compact`:

```
/remind focus on the USB descriptor work and skip the test failures
```

`cc-local-compact unregister` takes it all back out again if you change your mind.

### Things worth knowing about `/remind`

It's slow, and it's silent. A real summarisation pass on a large session takes a couple of minutes on the default 9B model, and up to about 16 minutes if you've pointed it at something like a 27B. There's no spinner or progress bar while it runs, Claude Code doesn't draw one for this kind of thing, so it just sits there looking hung. It isn't.

It only works if something actually ran in that window before you cleared it. Fresh terminal, immediate `/remind`, nothing to recover, and it'll tell you so rather than making something up.

## Using it from the command line

Nothing needs to be running for this, it works on the session transcript files directly.

```bash
cc-local-compact list       # what sessions exist for this project
cc-local-compact compact    # summarise the current one
```

If there's more than one session for the project and it can't work out which one you mean, it asks rather than guessing:

```
2 sessions exist for this project, pick one:
  [1] 2026-09-04 14:22  fixing the i2c timeout
      /home/you/.claude/projects/my-project/6f2a....jsonl
  [2] 2026-09-03 09:15  README rewrite
      /home/you/.claude/projects/my-project/91cd....jsonl
Select session [1-2] (blank to cancel):
```

Each one is labelled with its `/rename` title if you gave it one, otherwise the last thing said in it, so they're actually tellable apart.

Or just name the file:

```bash
cc-local-compact compact ~/.claude/projects/<project>/<session>.jsonl
```

Useful flags:

```
--instructions TEXT     steer the summary, same as /compact <text>
--model NAME            use a different model just for this run
--fallback-model NAME   retry against a second model if the first produces garbage
--output PATH           write the summary somewhere specific
--context-budget N      override the per-pass token budget
```

`--fallback-model` is handy if you want a fast small model doing the bulk of the work with something larger picking up the passes it makes a mess of.

## Using it as an MCP tool

If you'd rather ask Claude to compact a session mid-conversation, register it as an MCP server:

```bash
claude mcp add cc-local-compact --scope user -- cc-local-compact serve
```

`--scope user` makes it available in every project, which is usually what you want for this. Drop it to add it to just the current project, or use `--scope project` to write it into a `.mcp.json` your team shares.

That exposes `compact_session` and `list_sessions`, so you can just say "compact this session" and it'll work out which one you're in and do it.

This is independent of `/remind`; set up either, or both.

## Where the summaries go

Every run writes a markdown file next to the session it came from:

```
~/.claude/projects/<project>/compact-summaries/<session-id>-compact-<timestamp>.md
```

The summary itself is at the bottom, under a header block recording which model produced it, how many passes it took, and rough before/after token counts. Handy for working out whether your model is doing a decent job before you rely on it.

## What it doesn't do

**It won't make your next resumed turn cheaper.** This was the original motivation and it turns out not to be possible; there's no way for an external tool to make Claude Code send less on a future turn, and it was tested directly rather than assumed. What you get is a summary you can read, and `/remind` to inject it after a deliberate `/clear`. Nothing more.

**It can't trigger the real `/clear` or `/compact` for you.** Those are yours to type.

**Summary quality is entirely down to your model.** A small quantised model will produce a worse summary than Claude does, sometimes a lot worse. Check the output a few times before trusting it with anything you care about. The `--fallback-model` flag exists because of exactly this.

## How it works

Short version: it reads the session's `.jsonl` transcript off disk, groups the messages oldest-first, and feeds as much as fits into your model's context in one go, keeping the newest messages untouched. If the session is too big for one pass it chains passes until it converges. `/remind` hangs off a couple of Claude Code hooks that keep track of which session belongs to which terminal window, so it can find the right pre-`/clear` conversation even with several windows open on the same project.

The whole thing is a port of Claude Code's own `/compact`, written from a reverse-engineering pass on the real implementation, so the summarisation behaves close to the original rather than being a fresh guess at how to do it.

If you want the long version, `cc_local_compact/README.md` covers the architecture, the multi-pass logic, session resolution, and a fair bit of detail on why `/remind` is built the way it is. `docs/compact-architecture.md` in the `cc-patcher` repo has the reverse-engineering writeup itself.

## Licence

MIT.
