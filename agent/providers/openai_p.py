"""OpenAI provider adapter — skeleton.

The system is model-agnostic by design (IOAI² allows any model; awards favor
reproducible ones). This adapter maps the neutral interface onto OpenAI's
chat-completions tool-calling. Finish + test when an OpenAI key is available.
"""
from __future__ import annotations

import json
import os

from .base import Provider, StepResult, ToolCall, ToolResult


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5")
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pip install openai to use --backend openai") from e
        from openai import OpenAI
        self._client = OpenAI()

    def _to_openai_tools(self, tools):
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"],
                              "parameters": t["input_schema"]}} for t in tools]

    def step(self, system, history, tools, max_tokens=8000) -> StepResult:
        msgs = [{"role": "system", "content": system}] + history
        resp = self._client.chat.completions.create(
            model=self.model, messages=msgs,
            tools=self._to_openai_tools(tools), max_completion_tokens=max_tokens)
        m = resp.choices[0].message
        calls = [ToolCall(id=tc.id, name=tc.function.name,
                          input=json.loads(tc.function.arguments or "{}"))
                 for tc in (m.tool_calls or [])]
        stop = "tool_use" if calls else "end"
        usage = {"input_tokens": resp.usage.prompt_tokens,
                 "output_tokens": resp.usage.completion_tokens}
        return StepResult(text=m.content or "", tool_calls=calls,
                          stop_reason=stop, usage=usage, raw=m)

    def append_assistant(self, history, step) -> None:
        m = step.raw
        entry = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            entry["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
        history.append(entry)

    def append_tool_results(self, history, results: list[ToolResult]) -> None:
        for r in results:
            history.append({"role": "tool", "tool_call_id": r.call_id,
                            "content": r.output})

    def append_user(self, history, text: str) -> None:
        history.append({"role": "user", "content": text})
