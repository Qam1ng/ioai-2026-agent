"""Anthropic (Claude) provider adapter. Default backend."""
from __future__ import annotations

import os
import time
from pathlib import Path

from .base import Provider, StepResult, ToolCall, ToolResult

# $/MTok for claude-opus-4-8
#: USD per million tokens, (input, output). Longest matching substring wins.
#: The efficiency figures the rules require us to publish come straight from
#: this table, so a single hardcoded price would misreport every cheap run.
_PRICES = {
    "haiku": (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus": (5.0, 25.0),
    "fable": (5.0, 25.0),
}
_PRICE_DEFAULT = (5.0, 25.0)


def price_for(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    best, price = 0, _PRICE_DEFAULT
    for key, val in _PRICES.items():
        if key in m and len(key) > best:
            best, price = len(key), val
    return price


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        _load_dotenv()
        import anthropic

        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
        self._client = anthropic.Anthropic()
        self._anthropic = anthropic
        # Assume extended thinking until a 400 proves otherwise (see step()).
        self._extended = True

    # -- interface ---------------------------------------------------------
    def step(self, system, history, tools, max_tokens=8000) -> StepResult:
        last_err = None
        for attempt in range(5):
            kwargs = dict(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=history,
                tools=tools,
            )
            # Adaptive thinking and effort control are frontier-model features.
            # Cheap models (Haiku) reject them with a 400, which used to kill
            # the whole run on the first call; probe once and remember.
            if self._extended:
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {"effort": "high"}
            try:
                resp = self._client.messages.create(**kwargs)
                break
            except (self._anthropic.APIStatusError,
                    self._anthropic.APIConnectionError) as e:
                status = getattr(e, "status_code", None)
                msg = str(e).lower()
                if self._extended and status == 400 and (
                    "thinking" in msg or "output_config" in msg or "effort" in msg
                ):
                    # This model does not do extended thinking. Downgrade for
                    # the rest of the run rather than failing.
                    self._extended = False
                    continue
                if status is not None and status < 500 and status != 429:
                    raise  # 4xx (except 429): our bug — don't retry
                last_err = e
                time.sleep(3 * (attempt + 1))
        else:
            raise last_err

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, input=dict(b.input))
                 for b in resp.content if b.type == "tool_use"]
        stop = "tool_use" if resp.stop_reason == "tool_use" else (
            "end" if resp.stop_reason == "end_turn" else resp.stop_reason)
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}
        return StepResult(text=text, tool_calls=calls, stop_reason=stop,
                          usage=usage, raw=resp.content)

    def append_assistant(self, history, step) -> None:
        # pass provider-native blocks back verbatim (incl. thinking blocks)
        history.append({"role": "assistant", "content": step.raw})

    def append_tool_results(self, history, results: list[ToolResult]) -> None:
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call_id,
             "content": r.output, "is_error": r.is_error}
            for r in results
        ]})

    def append_user(self, history, text: str) -> None:
        history.append({"role": "user", "content": text})

    def cost_usd(self, usage) -> float:
        pin, pout = price_for(self.model)
        return (usage.get("input_tokens", 0) * pin
                + usage.get("output_tokens", 0) * pout) / 1e6
