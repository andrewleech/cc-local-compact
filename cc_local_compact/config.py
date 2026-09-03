"""Runtime configuration, resolved from environment variables.

Resolution order for base URL and model deliberately falls back to the
CLAUDE_NET_PROXY_LOCAL_* vars already used by cc-local-router's proxy: this
tool never imports or invokes that proxy process, it only reuses the same
known-good values so there is one place the user keeps them in sync.
"""

import dataclasses
import os


DEFAULT_BASE_URL = "http://titan:8080"
DEFAULT_MODEL = "qwen3.5-9b"
"""As of 2026-09-02, qwen3.5-9b is the Q6 GGUF build (unsloth UD-Q6_K_XL,
llama.cpp, full 262144 context); confirmed the best default for this
tool's workload: on this project's own 488K-token benchmark it matched
qwen3.8-27b's compaction quality and reliability (0 fallbacks needed, 0
malformed responses, same group split both passes) at ~5.8x the speed
(~2m48s vs ~16m14s for the same task). It replaced an AWQ-int4 build
previously served under this same id, which showed reproducible coherence
failures (garbled/empty/off-task/hallucinated output) on dense self-
referential content, root-caused to the int4 quantization via a live
A/B against this Q6 build on the identical failing input, and the AWQ
deployment has since been fully retired (catalog entry and wrapper
deleted, unreachable under any id). Also answers to the alias
`summariser`."""
DEFAULT_API_KEY = "local"

MODEL_CONTEXT_WINDOWS = {
    "qwen3.8-27b": 262144,
    "qwen3.8-27b-fast": 131072,
    "qwen3.5-9b": 262144,
    "gemma-4-12b": 131072,
    "gemma-4-e4b": 65536,
}
"""Real windows per llama-swap's own /v1/models listing on titan:8080, not
a nominal/marketing figure; gemma-4-12b was reconfigured from 64k to
128k ("Gemma 4 native max") after this constant was first written, and
qwen3.5-9b itself was cut over from an AWQ-int4 build to a Q6 GGUF build
under the same id (see DEFAULT_MODEL); re-check the live listing before
trusting any of these if the deployment might have changed again.
qwen3.8-27b-fast is the same weights as qwen3.8-27b with ~2x faster decode
via speculative decoding, but its MTP draft buffers only fit at half the
main model's window (128K, not 262K). qwen3.5-9b is single-stream only
(one shared KV pool, not N concurrent slots), which matches how this tool
already calls it (multi-pass runs sequentially, never fans out concurrent
in-flight requests), so no retuning needed on this side."""

MODELS_WITH_THINKING_TOGGLE = {"qwen3.8-27b", "qwen3.8-27b-fast", "qwen3.8-27b-vllm"}
"""Models where client.py should send extra_body.chat_template_kwargs.
enable_thinking=False, a deliberate per-model allowlist, not a blanket
default, because sending it to a model that doesn't need it isn't free.
qwen3.8-27b itself is directly confirmed: reasoning on by default, and the
override measurably suppresses it (content blocks drop from
['thinking','text'] to ['text']) at normal latency. qwen3.8-27b-fast
shares those weights so is assumed to behave the same, not independently
retested. qwen3.8-27b-vllm is included on the strength of its own
llama-swap model description, which explicitly documents this exact
mechanism ("disable per-request with chat_template_kwargs.enable_thinking
=false"); not directly load-tested here either.

qwen3.5-9b (now the Q6 GGUF build, see DEFAULT_MODEL) has reasoning off by
default per its own model description and is deliberately left off this
set, no evidence it needs toggling. The now-retired AWQ-int4 build
previously served under this id was confirmed harmful to send the
override to (never emitted a thinking block either way, but the override
cost ~43x latency for identical output in a real side-by-side test:
147.4s vs 3.4s for the same input_tokens=1416/output_tokens=400); worth
remembering as a general lesson (an unneeded chat-template override is not
free on every backend) even though the specific model it applied to is
gone. gemma-4-12b/gemma-4-e4b are untested for this latency effect and are
deliberately left off this set until checked, rather than assumed safe."""

FALLBACK_CONTEXT_WINDOW = 32768
CONTEXT_BUDGET_FRACTION = 0.75
"""A single flat context_budget default can't serve every backend model
well; windows above range from 64k to 262k, 4x apart. Deriving the
default per resolved model, as a fraction of its real window, gets each
model close to matching what it can actually do in one pass (real Claude
Code /compact observed compacting a 488K-token session to ~15K in a single
pass, only possible because its own model's context comfortably held the
whole batch) while still leaving margin for the response-token reservation
and seeding-estimate error. Unknown models fall back to a conservative
32768 window. This governs per-pass batch size, not a hard ceiling on what
the tool can handle overall; multipass.py chains passes for sessions too
large even for this budget. Override with CC_LOCAL_COMPACT_CONTEXT_BUDGET
to bypass model-based derivation entirely."""

RESPONSE_TOKENS_FRACTION = 0.3
RESPONSE_TOKENS_FLOOR = 8192
"""Claude Code's own /compact call (NBn -> rw) passes no maxOutputTokens
at all; it just uses the model's normal ceiling for an ordinary turn,
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


def _default_context_budget(model: str) -> int:
    window = MODEL_CONTEXT_WINDOWS.get(model, FALLBACK_CONTEXT_WINDOW)
    return round(window * CONTEXT_BUDGET_FRACTION)


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
    model = _first(
        "CC_LOCAL_COMPACT_MODEL", "CLAUDE_NET_PROXY_LOCAL_MODEL",
        default=DEFAULT_MODEL,
    )
    context_budget_env = os.environ.get("CC_LOCAL_COMPACT_CONTEXT_BUDGET")
    context_budget = (
        int(context_budget_env) if context_budget_env
        else _default_context_budget(model)
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
        model=model,
        api_key=_first(
            "CC_LOCAL_COMPACT_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            default=DEFAULT_API_KEY,
        ),
        context_budget=context_budget,
        response_max_tokens=response_max_tokens,
    )
