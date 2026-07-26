"""Provider abstraction — the agent's brain is pluggable.

The orchestrator never touches provider-specific message formats. A provider
owns its native conversation history format; the orchestrator only:

    provider.step(system, history, tools)  -> StepResult
    provider.append_assistant(history, step)          # add model turn
    provider.append_tool_results(history, results)    # add tool outputs

`history` is an opaque list the orchestrator threads through unchanged, so any
backend (Anthropic, OpenAI, local vLLM...) can be dropped in by implementing
this interface.
"""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class StepResult:
    text: str                       # concatenated visible text this turn
    tool_calls: list[ToolCall]      # requested tool invocations (may be empty)
    stop_reason: str                # provider-normalized: "tool_use" | "end" | other
    usage: dict                     # {input_tokens, output_tokens}
    raw: Any = field(repr=False, default=None)  # provider-native assistant content


@dataclass
class ToolResult:
    call_id: str
    output: str
    is_error: bool = False


class Provider:
    """Interface. Implement all methods for a new backend."""

    name = "base"
    model = "?"

    def step(self, system: str, history: list, tools: list[dict],
             max_tokens: int = 8000) -> StepResult:
        raise NotImplementedError

    def append_assistant(self, history: list, step: StepResult) -> None:
        raise NotImplementedError

    def append_tool_results(self, history: list, results: list[ToolResult]) -> None:
        raise NotImplementedError

    def append_user(self, history: list, text: str) -> None:
        raise NotImplementedError

    def cost_usd(self, usage: dict) -> float:
        return 0.0


def _construct(klass, kwargs: dict) -> Provider:
    """Instantiate a provider, dropping kwargs its constructor does not accept.

    Adapters grow parameters at different rates (``base_url`` only means
    something for an OpenAI-compatible gateway). Dropping is preferable to
    crashing on competition day, but a dropped argument is a silent config
    error, so it is warned about.
    """
    accepted = inspect.signature(klass.__init__).parameters
    kept, dropped = {}, []
    for k, v in kwargs.items():
        if k in accepted:
            kept[k] = v
        elif v is not None:
            dropped.append(k)
    if dropped:
        warnings.warn(
            f"{klass.__name__} ignores {dropped}: not supported by this backend",
            RuntimeWarning,
            stacklevel=3,
        )
    return klass(**kept)


def get_provider(
    backend: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> Provider:
    """Build the adapter for ``backend``.

    ``base_url``/``api_key_env`` exist so one OpenAI-compatible adapter can
    serve any internal gateway (this is how Gemini 3.1 Pro is reached).
    ``api_key_env`` is the *name* of the environment variable holding the key,
    never the key itself — keys must not travel through config files.
    Positional calls ``get_provider(backend)`` / ``get_provider(backend, model)``
    keep working.
    """
    kwargs = {"model": model, "base_url": base_url, "api_key_env": api_key_env}
    if backend in ("claude", "anthropic"):
        from .anthropic_p import AnthropicProvider
        return _construct(AnthropicProvider, kwargs)
    if backend in ("openai", "gpt", "gemini", "compat"):
        from .openai_p import OpenAIProvider
        return _construct(OpenAIProvider, kwargs)
    raise ValueError(f"unknown backend: {backend!r} (use claude|openai)")


# Tool schema is declared provider-neutrally as:
#   {"name": ..., "description": ..., "input_schema": {json-schema}}
# (This happens to match Anthropic's wire format; adapters translate as needed.)
