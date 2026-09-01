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
DEFAULT_CONTEXT_BUDGET = 32000
DEFAULT_RESPONSE_MAX_TOKENS = 4096


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
        context_budget=int(
            os.environ.get("CC_LOCAL_COMPACT_CONTEXT_BUDGET", DEFAULT_CONTEXT_BUDGET)
        ),
        response_max_tokens=int(
            os.environ.get(
                "CC_LOCAL_COMPACT_RESPONSE_MAX_TOKENS", DEFAULT_RESPONSE_MAX_TOKENS,
            )
        ),
    )
