"""Both CLIs, both pointed at one Azure resource.

Claude Code speaks the Anthropic Messages API and Codex speaks the OpenAI
Responses API. The Azure resource serves both shapes off one host and one key,
so no translating proxy is needed -- only two base URLs that differ in how much
of the path they already carry.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_ANTHROPIC_BASE, DEFAULT_OPENAI_BASE

#: Only these are handed to an agent process. Everything else -- above all any
#: KAGGLE_* variable -- stays with the controller, which is the only thing
#: allowed to spend the submission budget.
_ENV_ALLOWLIST = {
    "PATH", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "NO_PROXY", "CUDA_VISIBLE_DEVICES",
}


@dataclass
class RunResult:
    agent_id: str
    exit_code: int | None
    timed_out: bool
    duration_s: float
    result_text: str = ""
    session_id: str | None = None
    resolved_model: str | None = None
    errors: list[str] = field(default_factory=list)


def base_env(api_key_env: str, *, require_key: bool) -> tuple[dict[str, str], str]:
    """The env an agent process gets, plus the OpenRouter key to inject into it."""
    env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    key = os.environ.get(api_key_env, "")
    if not key and require_key:
        raise RuntimeError(
            f"{api_key_env} is not set; export your OpenRouter key before --live"
        )
    env.update({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    return env, key


class _Proc:
    def __init__(self) -> None:
        self._active: dict[str, asyncio.subprocess.Process] = {}

    async def _kill(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=8)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                await process.wait()

    async def terminate_all(self) -> None:
        await asyncio.gather(
            *(self._kill(p) for p in list(self._active.values())),
            return_exceptions=True,
        )

    async def _spawn(
        self, *, agent_id: str, argv: list[str], cwd: Path, env: dict[str, str],
        prompt: str, trace_path: Path, timeout_s: float, on_event,
    ) -> tuple[asyncio.subprocess.Process, bool, float]:
        start = time.monotonic()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True,
            limit=64 * 1024 * 1024,
        )
        self._active[agent_id] = process
        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        async def read_stdout() -> None:
            assert process.stdout is not None
            with trace_path.open("w", encoding="utf-8") as trace:
                while line := await process.stdout.readline():
                    text = line.decode("utf-8", errors="replace").rstrip("\n")
                    trace.write(text + "\n")
                    trace.flush()
                    try:
                        on_event(json.loads(text))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue

        async def read_stderr() -> None:
            assert process.stderr is not None
            path = trace_path.with_suffix(".stderr.log")
            with path.open("w", encoding="utf-8") as stream:
                while line := await process.stderr.readline():
                    stream.write(line.decode("utf-8", errors="replace"))
                    stream.flush()

        readers = [asyncio.create_task(read_stdout()),
                   asyncio.create_task(read_stderr())]
        timed_out = False
        try:
            async with asyncio.timeout(max(1.0, timeout_s)):
                await process.wait()
                await asyncio.gather(*readers)
        except TimeoutError:
            timed_out = True
            await self._kill(process)
            await asyncio.gather(*readers, return_exceptions=True)
        except asyncio.CancelledError:
            await self._kill(process)
            await asyncio.gather(*readers, return_exceptions=True)
            raise
        finally:
            self._active.pop(agent_id, None)
        return process, timed_out, time.monotonic() - start


class ClaudeRunner(_Proc):
    """Claude Code driven through OpenRouter's Anthropic-compatible endpoint."""

    #: Measured, not assumed: `--bare` hands the agent only Bash/Edit/Read, and
    #: neither --tools nor --allowedTools can add anything back. A solver needs
    #: Write to publish a candidate, so solvers must NOT run bare.
    SOLVER_TOOLS = "Bash,Read,Edit,Write,Glob,Grep,WebSearch,WebFetch"

    def __init__(self, *, binary: Path, model: str, effort: str,
                 api_key_env: str, base_url: str = DEFAULT_ANTHROPIC_BASE,
                 tools: str | None = None):
        super().__init__()
        self.binary = Path(binary)
        self.model = model
        self.effort = effort
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        #: None -> the solver set. "" -> no tools at all, which is what the
        #: manager wants: its whole input is already in the prompt, and a
        #: manager that can run Bash is a manager that can find a way to submit.
        self.tools = self.SOLVER_TOOLS if tools is None else tools

    def argv(self, *, resume: str | None) -> list[str]:
        argv = [
            str(self.binary), "-p",
            "--model", self.model,
            "--effort", self.effort,
            "--output-format", "stream-json", "--verbose",
            "--tools", self.tools,
            "--prompt-suggestions", "false",
        ]
        if self.tools:
            argv.extend(["--permission-mode", "bypassPermissions",
                         "--allowedTools", self.tools])
        if resume:
            argv.extend(["--resume", resume])
        return argv

    def environment(self, workdir: Path, *, require_key: bool) -> dict[str, str]:
        env, key = base_env(self.api_key_env, require_key=require_key)
        home = workdir / ".claude_home"
        home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["CLAUDE_CONFIG_DIR"] = str(home / "config")
        # The SDK appends "/v1/messages", so the base must NOT already carry a
        # version segment.
        env["ANTHROPIC_BASE_URL"] = self.base_url
        env["ANTHROPIC_AUTH_TOKEN"] = key
        env["ANTHROPIC_API_KEY"] = key
        env["ANTHROPIC_MODEL"] = self.model
        env["ANTHROPIC_SMALL_FAST_MODEL"] = self.model
        env["CLAUDE_CODE_BG_CLASSIFIER_MODEL"] = self.model
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = self.model
        # Load-bearing, not hygiene: without it Claude Code adds
        # `anthropic-beta: advisor-tool-2026-03-01`, and the Azure endpoint
        # rejects the whole request with 400 "Unexpected value(s)".
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env

    async def run(self, *, agent_id: str, prompt: str, workdir: Path,
                  trace_dir: Path, timeout_s: float,
                  resume: str | None = None, require_key: bool = True) -> RunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        state: dict[str, object] = {"text": "", "session": None, "model": None,
                                    "errors": []}

        def on_event(event: dict) -> None:
            if not isinstance(event, dict):
                return
            if event.get("type") == "system" and event.get("subtype") == "init":
                state["model"] = event.get("model")
                state["session"] = event.get("session_id")
            elif event.get("type") == "result":
                state["text"] = str(event.get("result", ""))
                if event.get("is_error"):
                    state["errors"] = [str(x) for x in (event.get("errors") or [])]

        process, timed_out, duration = await self._spawn(
            agent_id=agent_id, argv=self.argv(resume=resume), cwd=workdir,
            env=self.environment(workdir, require_key=require_key), prompt=prompt,
            trace_path=trace_dir / f"{agent_id}.jsonl", timeout_s=timeout_s,
            on_event=on_event,
        )
        return RunResult(
            agent_id=agent_id, exit_code=process.returncode, timed_out=timed_out,
            duration_s=duration, result_text=str(state["text"]),
            session_id=state["session"], resolved_model=state["model"],
            errors=list(state["errors"]),
        )


class CodexRunner(_Proc):
    """Codex CLI driven through OpenRouter's Responses endpoint."""

    PROVIDER = "ioai"

    def __init__(self, *, binary: Path, model: str, effort: str, api_key_env: str,
                 base_url: str = DEFAULT_OPENAI_BASE):
        super().__init__()
        self.binary = Path(binary)
        self.model = model
        self.effort = effort
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")

    def argv(self, *, workdir: Path, last_message: Path,
             resume: str | None) -> list[str]:
        p = self.PROVIDER
        argv = [
            str(self.binary), "--search", "--model", self.model,
            "--sandbox", "workspace-write", "--ask-for-approval", "never",
            "--cd", str(workdir),
            "--config", f'model_reasoning_effort="{self.effort}"',
            "--config", f'model_provider="{p}"',
            "--config", f'model_providers.{p}.name="IOAI"',
            "--config", f'model_providers.{p}.base_url="{self.base_url}"',
            "--config", f'model_providers.{p}.env_key="{self.api_key_env}"',
            "--config", f'model_providers.{p}.wire_api="responses"',
            "--config", f"model_providers.{p}.supports_standalone_web_search=true",
            # The provider key authenticates Codex itself; it must never reach a
            # shell the agent spawns.
            "--config", 'shell_environment_policy.inherit="core"',
            "--config", "shell_environment_policy.ignore_default_excludes=false",
            "--config", f'shell_environment_policy.filters.{self.api_key_env}="exclude"',
            "exec",
        ]
        tail = ["--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
                "--json", "--output-last-message", str(last_message)]
        if resume:
            argv.extend(["resume", *tail, resume, "-"])
        else:
            argv.extend([*tail, "-"])
        return argv

    def environment(self, workdir: Path, *, require_key: bool) -> dict[str, str]:
        env, key = base_env(self.api_key_env, require_key=require_key)
        home = workdir / ".codex_agent_home"
        codex_home = workdir / ".codex_home"
        home.mkdir(parents=True, exist_ok=True)
        codex_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["CODEX_HOME"] = str(codex_home)
        env[self.api_key_env] = key
        return env

    async def run(self, *, agent_id: str, prompt: str, workdir: Path,
                  trace_dir: Path, timeout_s: float,
                  resume: str | None = None, require_key: bool = True) -> RunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        last_message = trace_dir / f"{agent_id}.last_message.md"
        state: dict[str, object] = {"text": "", "session": None, "model": None}

        def on_event(event: dict) -> None:
            if not isinstance(event, dict):
                return
            if event.get("type") == "thread.started":
                state["session"] = str(event.get("thread_id") or "") or None
            model = event.get("model") or event.get("model_name")
            if isinstance(model, str):
                state["model"] = model
            item = event.get("item")
            if event.get("type") == "item.completed" and isinstance(item, dict) \
                    and item.get("type") == "agent_message":
                state["text"] = str(item.get("text") or "")

        process, timed_out, duration = await self._spawn(
            agent_id=agent_id,
            argv=self.argv(workdir=workdir, last_message=last_message, resume=resume),
            cwd=workdir, env=self.environment(workdir, require_key=require_key),
            prompt=prompt, trace_path=trace_dir / f"{agent_id}.jsonl",
            timeout_s=timeout_s, on_event=on_event,
        )
        text = str(state["text"])
        if last_message.is_file():
            text = last_message.read_text(encoding="utf-8", errors="replace")
        return RunResult(
            agent_id=agent_id, exit_code=process.returncode, timed_out=timed_out,
            duration_s=duration, result_text=text, session_id=state["session"],
            resolved_model=state["model"],
        )
