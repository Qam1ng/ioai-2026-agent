"""Run a role as a real headless Claude Code process.

For roles whose job is *writing and debugging code* (Coder, Tuner, Aggregator),
a full coding agent beats a hand-rolled tool loop: it already owns file
editing, shell access, its own test-and-fix cycle, and context management. So
those roles are spawned as `claude -p` subprocesses, one per candidate, each
confined to its own candidate directory.

Roles whose job is *judgement* (Manager, Verifier, Compliance, Designer) stay
on the plain API path in ``swarm.roles.base.Role`` — they need a controlled,
auditable context, not a filesystem.

The two paths return the same ``RoleResult``, so the pod does not care which
one a role uses; it is a per-role configuration choice.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


class ClaudeCodeUnavailable(RuntimeError):
    """Raised when the `claude` CLI is not installed or not on PATH."""


_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv() -> None:
    """Make ``.env`` visible to the spawned CLI.

    The API-path providers read ``.env`` themselves, but a subprocess only sees
    the real environment — and a spawned ``claude -p`` does not inherit an
    interactive login either. Without this, a repo with a perfectly good key in
    ``.env`` still produces "Not logged in" from every coding role.
    """
    env = _ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def claude_available() -> bool:
    return shutil.which("claude") is not None


#: Substrings the CLI returns when it has a binary but no credentials. A
#: spawned `claude -p` does not inherit an interactive session's login, so this
#: is the failure a run hits first if nobody checked beforehand.
_AUTH_MARKERS = ("not logged in", "please run /login", "invalid api key", "authentication_error")


def is_auth_failure(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _AUTH_MARKERS)


def check_auth(timeout_s: float = 60.0) -> tuple[bool, str]:
    """Verify the CLI can actually answer. Run this before a competition window.

    Returns ``(ok, detail)``. Silent degradation here is expensive: every
    coding role would return empty and the pod would fall back to the trivial
    kernel while looking, from the outside, like it was working.
    """
    load_dotenv()
    if not claude_available():
        return False, "`claude` CLI not found on PATH"
    try:
        proc = subprocess.run(
            ["claude", "-p", "reply with the single word: ok", "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False, f"claude CLI did not respond within {timeout_s:.0f}s"
    out = (proc.stdout or "") + (proc.stderr or "")
    if is_auth_failure(out):
        return False, "claude CLI is not authenticated (run `claude` and /login, or set ANTHROPIC_API_KEY)"
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return proc.returncode == 0, out[-300:]
    if payload.get("is_error"):
        return False, str(payload.get("result", ""))[-300:]
    return True, str(payload.get("result", ""))[:120]


@dataclass
class ClaudeCodeResult:
    ok: bool
    text: str = ""
    session_id: str = ""
    num_turns: int = 0
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)
    duration_s: float = 0.0
    error: str = ""
    raw: dict = field(default_factory=dict)


#: Tools a coding role is allowed to use. Deliberately excludes WebFetch and
#: WebSearch: the submitted kernel runs without internet, so a solution that
#: quietly depends on a downloaded artifact cannot be reproduced on Kaggle.
CODER_TOOLS = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")


def run_claude_code(
    prompt: str,
    cwd: Path,
    *,
    system_append: str = "",
    allowed_tools: tuple[str, ...] = CODER_TOOLS,
    model: str = "",
    max_turns: int = 60,
    timeout_s: float = 1800,
    add_dirs: tuple[Path, ...] = (),
    resume_session: str = "",
    env_extra: dict | None = None,
) -> ClaudeCodeResult:
    """Run one headless Claude Code turn to completion.

    ``cwd`` is the sandbox: the agent is pointed at this directory and only
    ``add_dirs`` are additionally readable. Returns structured output rather
    than raising, because a failed coding attempt is information the Manager
    must see, not an exception that kills the pod.
    """
    load_dotenv()
    if not claude_available():
        raise ClaudeCodeUnavailable("`claude` CLI not found on PATH")

    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = ["claude", "-p", prompt, "--output-format", "json"]
    if allowed_tools:
        cmd += ["--allowedTools", ",".join(allowed_tools)]
    if system_append:
        cmd += ["--append-system-prompt", system_append]
    if model:
        cmd += ["--model", model]
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]
    for d in add_dirs:
        cmd += ["--add-dir", str(d)]
    if resume_session:
        cmd += ["--resume", resume_session]

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ClaudeCodeResult(
            ok=False,
            duration_s=time.time() - t0,
            error=f"claude code timed out after {timeout_s:.0f}s",
        )

    elapsed = time.time() - t0
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    payload: dict = {}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        # Older CLI versions, or a crash before the JSON envelope was written.
        return ClaudeCodeResult(
            ok=proc.returncode == 0,
            text=stdout[-8000:],
            duration_s=elapsed,
            error="" if proc.returncode == 0 else f"exit={proc.returncode}: {stderr[-2000:]}",
        )

    usage = payload.get("usage") or {}
    is_error = bool(payload.get("is_error")) or proc.returncode != 0
    if is_auth_failure(payload.get("result", "")):
        return ClaudeCodeResult(
            ok=False,
            text=payload.get("result", ""),
            duration_s=elapsed,
            error=(
                "claude CLI is not authenticated. A spawned `claude -p` does not inherit an "
                "interactive login. Run `claude` and /login, or export ANTHROPIC_API_KEY."
            ),
            raw=payload,
        )
    return ClaudeCodeResult(
        ok=not is_error,
        text=payload.get("result", "") or "",
        session_id=payload.get("session_id", "") or "",
        num_turns=int(payload.get("num_turns", 0) or 0),
        cost_usd=float(payload.get("total_cost_usd", 0.0) or 0.0),
        usage={
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        },
        duration_s=elapsed,
        error="" if not is_error else str(payload.get("result", ""))[-2000:],
        raw=payload,
    )
