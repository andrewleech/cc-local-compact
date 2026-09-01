"""Runtime configuration, resolved from environment variables.

Resolution order for base URL and model deliberately falls back to the
CLAUDE_NET_PROXY_LOCAL_* vars already used by cc-local-router's proxy: this
tool never imports or invokes that proxy process, it only reuses the same
known-good values so there is one place the user keeps them in sync.
"""

import dataclasses
import os


DEFAULT_BASE_URL = "http://titan:8080"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_API_KEY = "local"
DEFAULT_CONTEXT_BUDGET = 48000
"""Deliberately well under either backend model's real context window
tested so far (qwen3.8-27b: 262144; gemma-4-12b/gemma-4-e4b: 64k) -- this
sizes one summarization batch, a working-set choice, not a claim about
what the model can hold. Override per model/host via
CC_LOCAL_COMPACT_CONTEXT_BUDGET if a model's real window is smaller."""

RESPONSE_TOKENS_FRACTION = 0.3
RESPONSE_TOKENS_FLOOR = 8192
"""Claude Code's own /compact call (NBn -> rw) passes no maxOutputTokens
at all -- it just uses the model's normal ceiling for an ordinary turn,
not a small artificial cap. The Anthropic Messages API requires an
explicit max_tokens, so this can't literally be "unconstrained," but it
should not be a small constant independent of context_budget either: a
fixed 4096 was observed truncating a real qwen3.8-27b summary mid-sentence
on a large transcript (no closing tag reached at all), and tokens.py's
chars/4 heuristic underestimates code-heavy content, so a tight estimate-
matched value isn't safe either. Scaling as a fraction of context_budget
(floored, so a small context_budget doesn't starve the response entirely)
keeps this from becoming the tightest constraint in the pipeline
regardless of what context_budget is configured."""


def _default_response_max_tokens(context_budget: int) -> int:
    return max(RESPONSE_TOKENS_FLOOR, round(context_budget * RESPONSE_TOKENS_FRACTION))


@dataclasses.dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    api_key: str
    context_budget: int
    response_max_tokens: int


def _first(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def load_config() -> Config:
    context_budget = int(
        os.environ.get("CC_LOCAL_COMPACT_CONTEXT_BUDGET", DEFAULT_CONTEXT_BUDGET)
    )
    response_max_tokens_env = os.environ.get("CC_LOCAL_COMPACT_RESPONSE_MAX_TOKENS")
    response_max_tokens = (
        int(response_max_tokens_env) if response_max_tokens_env
        else _default_response_max_tokens(context_budget)
    )
    return Config(
        base_url=_first(
            "CC_LOCAL_COMPACT_BASE_URL", "CLAUDE_NET_PROXY_LOCAL_URL",
            default=DEFAULT_BASE_URL,
        ),
        model=_first(
            "CC_LOCAL_COMPACT_MODEL", "CLAUDE_NET_PROXY_LOCAL_MODEL",
            default=DEFAULT_MODEL,
        ),
        api_key=_first(
            "CC_LOCAL_COMPACT_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            default=DEFAULT_API_KEY,
        ),
        context_budget=context_budget,
        response_max_tokens=response_max_tokens,
    )
