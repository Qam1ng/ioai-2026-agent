"""OpenAI-compatible provider adapter.

This is not only "the OpenAI backend". It is the adapter for *any* endpoint
that speaks the chat-completions wire format, which on competition day is how
we reach **Gemini 3.1 Pro through an internal gateway**: same code, different
``base_url`` + ``model``.

Four things the previous skeleton got wrong, and why each mattered:

* ``.env`` was never loaded, so the adapter only worked if the key happened to
  be exported in the shell — the Anthropic adapter loads it, so the failure was
  invisible until the backend was switched.
* No ``base_url``, which made every internal gateway unreachable.
* No retries. A single 429 killed a role mid-run.
* ``cost_usd`` inherited the base class's ``0.0``, which silently *disabled*
  the cost budget: ``PodBudget.exhausted()`` can never fire on cost if every
  call is free. The fallback price here is deliberately non-zero.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

from .base import Provider, StepResult, ToolCall, ToolResult

#: $/MTok, matched by substring against the model name, longest match wins.
#: Approximate list prices — the point is that the cost budget *moves*, not
#: that accounting is exact to the cent.
_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3-pro": (2.00, 12.00),
    "gemini-3": (2.00, 12.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini": (1.25, 10.00),
    "gpt-5.5": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "o4-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "deepseek": (0.27, 1.10),
    "qwen": (0.40, 1.20),
}

#: Used when the model name matches nothing above. Never 0.0 — a zero here
#: turns the cost budget into a no-op.
_FALLBACK_PRICE = (1.00, 5.00)

#: Gemini draws reasoning tokens from ``max_tokens`` and rejects anything above
#: this (memory: gemini-crawl-reasoning-maxtokens). Asking for more is a hard
#: 400, not a clamp, so we clamp on the way out.
_GEMINI_MAX_TOKENS_CAP = 65536

_RETRYABLE_STATUS = re.compile(r"\b(429|5\d\d)\b")


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def price_for(model: str) -> tuple[float, float]:
    """($/MTok in, $/MTok out) for ``model``, with a non-zero fallback."""
    name = (model or "").lower()
    best: tuple[str, tuple[float, float]] | None = None
    for key, price in _PRICES.items():
        if key in name and (best is None or len(key) > len(best[0])):
            best = (key, price)
    if best is not None:
        return best[1]
    return (
        float(os.environ.get("SWARM_PRICE_IN", _FALLBACK_PRICE[0])),
        float(os.environ.get("SWARM_PRICE_OUT", _FALLBACK_PRICE[1])),
    )


def _status_of(exc: Exception) -> int | None:
    """Best-effort HTTP status from an SDK exception or its message."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        return status
    m = _RETRYABLE_STATUS.search(str(exc))
    return int(m.group(1)) if m else None


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ):
        _load_dotenv()
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - environment problem
            raise RuntimeError("pip install openai to use --backend openai") from e

        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        # api_key_env names a variable; the value never appears in config.
        key = os.environ.get(api_key_env or "", "") if api_key_env else ""
        self.api_key = key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "no API key: set OPENAI_API_KEY (or point api_key_env at the "
                "variable that holds it) in the environment or in .env"
            )

        kwargs: dict = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

        self.max_attempts = int(os.environ.get("SWARM_MAX_ATTEMPTS", "5"))
        # Real OpenAI wants max_completion_tokens; most compat gateways only
        # know max_tokens. Start with the likely-correct one and self-correct
        # on the first 400 rather than failing the whole run.
        self._token_param = os.environ.get("OPENAI_MAX_TOKENS_PARAM") or (
            "max_tokens" if self.base_url else "max_completion_tokens"
        )
        self._token_param_pinned = bool(os.environ.get("OPENAI_MAX_TOKENS_PARAM"))

    # ------------------------------------------------------------- helpers
    @property
    def _is_gemini(self) -> bool:
        return "gemini" in (self.model or "").lower()

    def _cap_tokens(self, max_tokens: int) -> int:
        if self._is_gemini:
            return min(int(max_tokens), _GEMINI_MAX_TOKENS_CAP)
        return int(max_tokens)

    def _to_openai_tools(self, tools):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def _create(self, msgs, tools, max_tokens: int):
        payload: dict = {
            "model": self.model,
            "messages": msgs,
            self._token_param: max_tokens,
        }
        if tools:
            payload["tools"] = self._to_openai_tools(tools)
        # Anything beyond the common subset (reasoning_effort, response_format,
        # parallel_tool_calls, ...) is a 400 on most gateways, and effort is a
        # no-op on the Gemini gateway anyway, so nothing extra is sent.
        return self._client.chat.completions.create(**payload)

    # -- interface ---------------------------------------------------------
    def step(self, system, history, tools, max_tokens=8000) -> StepResult:
        msgs = [{"role": "system", "content": system}] + history
        max_tokens = self._cap_tokens(max_tokens)
        last_err: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                resp = self._create(msgs, tools, max_tokens)
                break
            except Exception as e:  # noqa: BLE001 - normalised below
                status = _status_of(e)
                msg = str(e).lower()
                blob = f"{type(e).__name__.lower()} {msg}"
                is_conn = any(
                    w in blob for w in ("connection", "timeout", "timed out", "unavailable")
                )
                if status == 400 and self._token_param in msg and not self._token_param_pinned:
                    # The endpoint disagrees about the token parameter name.
                    # Flip once and retry immediately; this is not a backoff case.
                    self._token_param = (
                        "max_tokens"
                        if self._token_param == "max_completion_tokens"
                        else "max_completion_tokens"
                    )
                    self._token_param_pinned = True
                    last_err = e
                    continue
                retryable = is_conn if status is None else (status == 429 or status >= 500)
                if not retryable:
                    raise  # other 4xx: our bug, retrying just burns the clock
                last_err = e
                # Exponential backoff with jitter so parallel roles do not
                # re-collide on the same rate limit window.
                time.sleep(min(30.0, 2.0 * (2**attempt)) * (0.5 + random.random()))
        else:
            raise last_err  # type: ignore[misc]

        m = resp.choices[0].message
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=self._parse_args(tc.function.arguments),
            )
            for tc in (m.tool_calls or [])
        ]
        stop = "tool_use" if calls else "end"
        usage = {
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
        }
        return StepResult(
            text=m.content or "", tool_calls=calls, stop_reason=stop, usage=usage, raw=m
        )

    @staticmethod
    def _parse_args(arguments: str | None) -> dict:
        """Tool arguments arrive as a JSON string; malformed JSON is a tool
        error, not a crash — the role sees it and can retry."""
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"__malformed_arguments__": arguments or ""}
        return parsed if isinstance(parsed, dict) else {"value": parsed}

    def append_assistant(self, history, step) -> None:
        m = step.raw
        entry = {"role": "assistant", "content": m.content or ""}
        if getattr(m, "tool_calls", None):
            entry["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
        history.append(entry)

    def append_tool_results(self, history, results: list[ToolResult]) -> None:
        for r in results:
            history.append(
                {"role": "tool", "tool_call_id": r.call_id, "content": r.output}
            )

    def append_user(self, history, text: str) -> None:
        history.append({"role": "user", "content": text})

    def cost_usd(self, usage) -> float:
        p_in, p_out = price_for(self.model)
        return (
            usage.get("input_tokens", 0) * p_in + usage.get("output_tokens", 0) * p_out
        ) / 1e6
