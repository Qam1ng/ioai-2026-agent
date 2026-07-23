"""Thin Anthropic client wrapper for the IOAI 2026 agent.

The API key is read from the environment (ANTHROPIC_API_KEY), loaded from a
gitignored .env file. Never hardcode the key.

Model defaults to Opus 4.8 (claude-opus-4-8) with adaptive thinking, per the
Anthropic API guidance for agentic / coding work.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency on python-dotenv).

    Only sets keys that are not already present in the environment, so a real
    exported env var always wins.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        os.environ.setdefault(key, val)


_load_dotenv()

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")


def get_client():
    """Return an Anthropic client. Import is local so the module loads even
    before the package is installed (e.g. for --check-key)."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Put it in .env (gitignored) or export it."
        )
    return anthropic.Anthropic(api_key=api_key)


def complete(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16000,
    effort: str = "high",
    thinking: bool = True,
    stream: bool = True,
) -> str:
    """One-shot completion. Returns the concatenated text of the response.

    - Adaptive thinking is on by default (recommended for reasoning/coding).
    - Streams by default so large max_tokens doesn't hit HTTP timeouts.
    """
    client = get_client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort},
    }
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    if stream:
        parts: list[str] = []
        with client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                parts.append(text)
        return "".join(parts)

    resp = client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if b.type == "text")


if __name__ == "__main__":
    # Smoke test: confirm the dedicated key + Opus 4.8 work end to end.
    import sys

    print(f"Model: {DEFAULT_MODEL}")
    print("Sending a tiny request to verify the API key works...\n")
    try:
        out = complete(
            system="You are a terse assistant.",
            user="Reply with exactly: IOAI-2026 agent online.",
            max_tokens=64,
            effort="low",
            thinking=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    print("Response:", out.strip())
    print("\nOK — key valid, Opus 4.8 reachable.")
