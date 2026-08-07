"""Provider for the ByteDance ModelHub `crawl` endpoint (Gemini et al.).

This is not a standard OpenAI server: the full URL (including the ``ak=``
credential in the query string) is the address, errors arrive as HTTP 200
bodies with an ``error.code``, and the response is OpenAI-shaped with Gemini
extras. Measured behaviour this provider is built around (probed 2026-07-27):

* Tool calling works natively (``finish_reason: "tool_calls"``, OpenAI shape)
  — but ``tool_calls[].id`` comes back as an **empty string**, so we mint local
  ids and keep an id→name map for the tool-result turn.
* Assistant messages carry a ``signature`` (Gemini thought signature). The
  message dict is replayed verbatim on later turns, signature included —
  stripping it risks the same class of failure as dropping Anthropic thinking
  blocks mid-conversation.
* ``usage.reasoning_tokens`` is real and billed on top of completion tokens,
  and reasoning is drawn from ``max_tokens`` (cap 65536), so small caps yield
  empty answers: the effective cap gets a floor.
* Rate limiting is the dominant failure: ``-2001`` (qpm) and ``-2004``
  (capacity) appeared on 2 of 3 probe calls. Both are retryable; a module-wide
  semaphore keeps parallel roles from stampeding the same quota.

The credential never enters the repo: ``CRAWL_URL`` lives in ``.env``
(gitignored) or the environment.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from pathlib import Path

import requests

from .base import Provider, StepResult, ToolCall, ToolResult

_ROOT = Path(__file__).resolve().parents[2]

#: Error codes that mean "try again later", not "your request is wrong".
_RETRYABLE_CODES = {"-2001", "-2004", "-4003"}
_MAX_ATTEMPTS = 8
_BACKOFF_BASE_S = 15.0  # qpm windows are minute-scale; short backoffs just burn attempts

#: One shared gate for every CrawlProvider in the process: ten roles hammering
#: a qpm-limited endpoint concurrently would turn the whole swarm into retries.
_GATE = threading.Semaphore(int(os.environ.get("CRAWL_CONCURRENCY", "2")))

_MAX_TOKENS_CAP = 65536
_MAX_TOKENS_FLOOR = 16384  # reasoning eats the budget; below this answers truncate


def _load_dotenv() -> None:
    env = _ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Anthropic schema shape ({name, description, input_schema}) → OpenAI."""
    out = []
    for t in tools or []:
        if "function" in t:  # already OpenAI-shaped
            out.append(t)
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return out


class CrawlError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"crawl error {code}: {message}")
        self.code = code


class CrawlProvider(Provider):
    """OpenAI-ish chat over the ModelHub crawl endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ):
        _load_dotenv()
        # The whole URL (with ?ak=...) is the credential-bearing address. An
        # api_key_env naming a var that holds the URL is accepted for symmetry
        # with the other providers.
        self.url = base_url or os.environ.get("CRAWL_URL", "")
        if not self.url and api_key_env:
            self.url = os.environ.get(api_key_env, "")
        if not self.url:
            raise RuntimeError(
                "CrawlProvider needs the full endpoint URL: set CRAWL_URL in .env "
                "(includes the ak= query parameter; never commit it)"
            )
        self.model = model or os.environ.get("CRAWL_MODEL", "gemini-3.1-p")
        self._call_names: dict[str, str] = {}  # local tool-call id -> tool name

    # ------------------------------------------------------------------ http
    def _post(self, body: dict) -> dict:
        last: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with _GATE:
                    resp = requests.post(
                        self.url,
                        json=body,
                        headers={
                            "Content-Type": "application/json",
                            "X-TT-LOGID": f"ioai-swarm-{uuid.uuid4().hex[:16]}",
                        },
                        timeout=600,
                    )
                data = resp.json()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(min(120.0, _BACKOFF_BASE_S * (attempt + 1)) + random.uniform(0, 5))
                continue

            err = data.get("error")
            if err:
                code = str(err.get("code", ""))
                if code in _RETRYABLE_CODES and attempt < _MAX_ATTEMPTS - 1:
                    last = CrawlError(code, err.get("message", ""))
                    time.sleep(min(120.0, _BACKOFF_BASE_S * (attempt + 1)) + random.uniform(0, 5))
                    continue
                raise CrawlError(code, err.get("message", ""))
            return data
        raise RuntimeError(f"crawl endpoint unreachable after {_MAX_ATTEMPTS} attempts: {last}")

    # ------------------------------------------------------------------ step
    def step(
        self, system: str, history: list, tools: list[dict], max_tokens: int = 8000
    ) -> StepResult:
        eff_max = min(_MAX_TOKENS_CAP, max(int(max_tokens), _MAX_TOKENS_FLOOR))
        body: dict = {
            "stream": False,
            "model": self.model,
            "max_tokens": eff_max,
            "messages": [{"role": "system", "content": system}] + list(history),
        }
        oai_tools = _to_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools

        data = self._post(body)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}

        calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            # Measured: the endpoint returns id="". Mint a stable local id so
            # the tool-result turn can reference something.
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            tc["id"] = call_id
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments", "")}
            name = fn.get("name", "")
            self._call_names[call_id] = name
            calls.append(ToolCall(id=call_id, name=name, input=args))

        usage_raw = data.get("usage") or {}
        usage = {
            "input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            # reasoning_tokens are billed and budgeted on top of completion.
            "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0)
            + int(usage_raw.get("reasoning_tokens", 0) or 0),
        }
        finish = choice.get("finish_reason", "")
        stop = "tool_use" if calls else ("end" if finish in ("stop", "") else finish)
        return StepResult(
            text=msg.get("content", "") or "",
            tool_calls=calls,
            stop_reason=stop,
            usage=usage,
            raw=msg,
        )

    # --------------------------------------------------------------- history
    def append_assistant(self, history: list, step: StepResult) -> None:
        # Replay the message verbatim: signature and tool_calls included.
        history.append(step.raw if isinstance(step.raw, dict) else {"role": "assistant", "content": step.text})

    def append_tool_results(self, history: list, results: list[ToolResult]) -> None:
        for r in results:
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": r.call_id,
                    "name": self._call_names.get(r.call_id, ""),
                    "content": r.output if not r.is_error else f"[error] {r.output}",
                }
            )

    def append_user(self, history: list, text: str) -> None:
        history.append({"role": "user", "content": text})

    # ------------------------------------------------------------------ cost
    def cost_usd(self, usage: dict) -> float:
        from .openai_p import price_for

        pin, pout = price_for(self.model)
        return (usage.get("input_tokens", 0) * pin + usage.get("output_tokens", 0) * pout) / 1e6
