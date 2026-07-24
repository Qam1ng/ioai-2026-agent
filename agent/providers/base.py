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


def get_provider(backend: str, model: str | None = None) -> Provider:
    if backend in ("claude", "anthropic"):
        from .anthropic_p import AnthropicProvider
        return AnthropicProvider(model=model)
    if backend in ("openai", "gpt"):
        from .openai_p import OpenAIProvider
        return OpenAIProvider(model=model)
    raise ValueError(f"unknown backend: {backend!r} (use claude|openai)")


# Tool schema is declared provider-neutrally as:
#   {"name": ..., "description": ..., "input_schema": {json-schema}}
# (This happens to match Anthropic's wire format; adapters translate as needed.)
