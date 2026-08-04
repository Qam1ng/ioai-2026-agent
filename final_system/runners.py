from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from pathlib import Path

from search_system.ioai_agent_system.models import AgentResult
from search_system.ioai_agent_system.redaction import redact_text

from .io import atomic_json
from .security import AgentSandbox


def _safe_base_env() -> dict[str, str]:
    allowed = {
        "PATH", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM",
        "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY",
        "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


class _ProcessRunner:
    def __init__(self) -> None:
        self._active: dict[str, asyncio.subprocess.Process] = {}

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
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

    async def terminate_all(self, *, grace_s: float = 8.0) -> None:
        del grace_s
        await asyncio.gather(
            *(self._terminate(process) for process in list(self._active.values())),
            return_exceptions=True,
        )


class ClaudeSubscriptionRunner(_ProcessRunner):
    """Claude Code using a selected Max-account profile, with an isolated HOME."""

    def __init__(
        self, *, binary: Path, model: str, effort: str, profile_dir: Path,
        allowed_tools: str = "Bash,Read,Edit,Write,Glob,Grep,WebSearch,WebFetch",
        sandbox: AgentSandbox | None = None,
    ):
        super().__init__()
        self.binary = Path(binary)
        self.model = model
        self.effort = effort
        self.profile_dir = Path(profile_dir)
        self.allowed_tools = allowed_tools
        self.sandbox = sandbox

    def argv(self, *, resume_session_id: str | None, persist_session: bool) -> list[str]:
        command = [
            str(self.binary), "--bare", "-p", "--model", self.model,
            "--effort", self.effort,
            "--output-format", "stream-json", "--verbose",
            "--permission-mode", "bypassPermissions",
            "--allowedTools", self.allowed_tools,
            "--prompt-suggestions", "false",
        ]
        if resume_session_id:
            command.extend(["--resume", resume_session_id])
        elif not persist_session:
            command.append("--no-session-persistence")
        return self.sandbox.wrap(command) if self.sandbox else command

    def environment(self, workdir: Path) -> dict[str, str]:
        env = _safe_base_env()
        home = workdir / ".agent_home"
        home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["CLAUDE_CONFIG_DIR"] = str(self.profile_dir)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION"] = "0"
        env["CLAUDE_CODE_BG_CLASSIFIER_MODEL"] = self.model
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = self.model
        return env

    async def run(
        self, *, agent_id: str, role: str, prompt: str, workdir: Path,
        trace_dir: Path, timeout_s: float, persist_session: bool = False,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        workdir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{agent_id}.jsonl"
        stderr_path = trace_dir / f"{agent_id}.stderr.log"
        start = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *self.argv(
                resume_session_id=resume_session_id,
                persist_session=persist_session,
            ),
            cwd=str(workdir), env=self.environment(workdir),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True,
            limit=64 * 1024 * 1024,
        )
        self._active[agent_id] = process
        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8")); await process.stdin.drain()
        process.stdin.close()
        resolved_model = session_id = None
        result_text = ""
        cost = 0.0
        errors: list[str] = []

        async def stdout_reader() -> None:
            nonlocal resolved_model, session_id, result_text, cost
            assert process.stdout is not None
            with trace_path.open("w", encoding="utf-8") as trace:
                while line := await process.stdout.readline():
                    text = redact_text(
                        line.decode("utf-8", errors="replace").rstrip("\n")
                    )
                    trace.write(text + "\n"); trace.flush()
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "system" and event.get("subtype") == "init":
                        resolved_model = event.get("model")
                        session_id = event.get("session_id")
                    elif event.get("type") == "result":
                        result_text = str(event.get("result", ""))
                        cost = float(event.get("total_cost_usd") or 0.0)
                        if event.get("is_error"):
                            errors.extend(str(item) for item in event.get("errors", []))

        async def stderr_reader() -> None:
            assert process.stderr is not None
            with stderr_path.open("w", encoding="utf-8") as stream:
                while line := await process.stderr.readline():
                    stream.write(redact_text(line.decode("utf-8", errors="replace")))
                    stream.flush()

        readers = [asyncio.create_task(stdout_reader()), asyncio.create_task(stderr_reader())]
        timed_out = False
        try:
            async with asyncio.timeout(max(0.1, timeout_s)):
                await process.wait(); await asyncio.gather(*readers)
        except TimeoutError:
            timed_out = True
            await self._terminate(process)
            await asyncio.gather(*readers, return_exceptions=True)
        except asyncio.CancelledError:
            await self._terminate(process)
            await asyncio.gather(*readers, return_exceptions=True)
            raise
        finally:
            self._active.pop(agent_id, None)
        if resolved_model and resolved_model != self.model:
            errors.append(f"resolved model mismatch: {resolved_model} != {self.model}")
        return AgentResult(
            agent_id=agent_id, role=role, workdir=workdir, trace_path=trace_path,
            exit_code=process.returncode, timed_out=timed_out,
            duration_s=time.monotonic() - start, resolved_model=resolved_model,
            session_id=session_id, cost_usd=cost, result_text=result_text,
            errors=errors,
        )


class OpenRouterCodexRunner(_ProcessRunner):
    """Codex CLI on OpenRouter Responses API, without operator HOME/Kaggle auth."""

    def __init__(
        self, *, binary: Path, model: str, effort: str, provider_id: str,
        base_url: str, api_key_env: str, supports_web_search: bool,
        sandbox: AgentSandbox | None = None,
    ):
        super().__init__()
        self.binary = Path(binary)
        self.model = model
        self.effort = effort
        self.provider_id = provider_id
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.supports_web_search = supports_web_search
        self.sandbox = sandbox

    def argv(
        self, *, workdir: Path, last_message: Path,
        resume_session_id: str | None, persist_session: bool,
    ) -> list[str]:
        provider = self.provider_id
        command = [
            str(self.binary), "--search", "--model", self.model,
            "--sandbox", "workspace-write", "--ask-for-approval", "never",
            "--cd", str(workdir),
            "--config", f'model_reasoning_effort="{self.effort}"',
            "--config", f'model_provider="{provider}"',
            "--config", f'model_providers.{provider}.name="OpenRouter"',
            "--config", f'model_providers.{provider}.base_url="{self.base_url}"',
            "--config", f'model_providers.{provider}.env_key="{self.api_key_env}"',
            "--config", f'model_providers.{provider}.wire_api="responses"',
            "--config", (
                f"model_providers.{provider}.supports_standalone_web_search="
                f"{'true' if self.supports_web_search else 'false'}"
            ),
            # provider key 只供 Codex 主进程鉴权，绝不能传给 Agent 启动的 Bash。
            "--config", 'shell_environment_policy.inherit="core"',
            "--config", "shell_environment_policy.ignore_default_excludes=false",
            "--config", (
                f'shell_environment_policy.filters.{self.api_key_env}="exclude"'
            ),
            "exec",
        ]
        tail = [
            "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
            "--json",
            "--output-last-message", str(last_message),
        ]
        if resume_session_id:
            command.extend(["resume", *tail, resume_session_id, "-"])
        else:
            if not persist_session:
                tail.append("--ephemeral")
            command.extend([*tail, "-"])
        return self.sandbox.wrap(command) if self.sandbox else command

    def environment(self, workdir: Path) -> dict[str, str]:
        env = _safe_base_env()
        if self.api_key_env not in os.environ:
            raise RuntimeError(f"required environment variable is missing: {self.api_key_env}")
        env[self.api_key_env] = os.environ[self.api_key_env]
        home = workdir / ".agent_home"
        codex_home = workdir / ".codex_home"
        home.mkdir(parents=True, exist_ok=True); codex_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["CODEX_HOME"] = str(codex_home)
        return env

    async def run(
        self, *, agent_id: str, role: str, prompt: str, workdir: Path,
        trace_dir: Path, timeout_s: float, persist_session: bool = False,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        workdir.mkdir(parents=True, exist_ok=True); trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{agent_id}.jsonl"
        stderr_path = trace_dir / f"{agent_id}.stderr.log"
        last_message = trace_dir / f"{agent_id}.last_message.md"
        start = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *self.argv(
                workdir=workdir, last_message=last_message,
                resume_session_id=resume_session_id,
                persist_session=persist_session,
            ), cwd=str(workdir), env=self.environment(workdir),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True,
            limit=64 * 1024 * 1024,
        )
        self._active[agent_id] = process
        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8")); await process.stdin.drain(); process.stdin.close()
        session_id = resolved_model = None
        result_text = ""
        errors: list[str] = []

        async def stdout_reader() -> None:
            nonlocal session_id, resolved_model, result_text
            assert process.stdout is not None
            with trace_path.open("w", encoding="utf-8") as trace:
                while line := await process.stdout.readline():
                    text = redact_text(
                        line.decode("utf-8", errors="replace").rstrip("\n")
                    )
                    trace.write(text + "\n"); trace.flush()
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "thread.started":
                        session_id = str(event.get("thread_id") or "") or None
                    model = event.get("model") or event.get("model_name")
                    if isinstance(model, str):
                        resolved_model = model
                    item = event.get("item")
                    if event.get("type") == "item.completed" and isinstance(item, dict) \
                            and item.get("type") == "agent_message":
                        result_text = str(item.get("text") or "")

        async def stderr_reader() -> None:
            assert process.stderr is not None
            with stderr_path.open("w", encoding="utf-8") as stream:
                while line := await process.stderr.readline():
                    stream.write(redact_text(line.decode("utf-8", errors="replace")))
                    stream.flush()

        readers = [asyncio.create_task(stdout_reader()), asyncio.create_task(stderr_reader())]
        timed_out = False
        try:
            async with asyncio.timeout(max(0.1, timeout_s)):
                await process.wait(); await asyncio.gather(*readers)
        except TimeoutError:
            timed_out = True; await self._terminate(process)
            await asyncio.gather(*readers, return_exceptions=True)
        except asyncio.CancelledError:
            await self._terminate(process)
            await asyncio.gather(*readers, return_exceptions=True); raise
        finally:
            self._active.pop(agent_id, None)
        if last_message.is_file():
            result_text = last_message.read_text(encoding="utf-8", errors="replace")
        if session_id is None:
            errors.append("Codex JSONL did not expose a thread_id")
        return AgentResult(
            agent_id=agent_id, role=role, workdir=workdir, trace_path=trace_path,
            exit_code=process.returncode, timed_out=timed_out,
            duration_s=time.monotonic() - start, resolved_model=resolved_model,
            session_id=session_id, result_text=result_text, errors=errors,
        )


async def run_lane_loop(
    *, lane: str, runner, prompt: str, continuation: str, workdir: Path,
    trace_dir: Path, deadline_monotonic: float, turn_seconds: float,
    max_rounds: int,
) -> list[AgentResult]:
    results: list[AgentResult] = []
    session_id: str | None = None
    for round_id in range(1, max_rounds + 1):
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 1:
            break
        result = await runner.run(
            agent_id=f"{lane}-round-{round_id:02d}", role=f"direct_{lane}",
            prompt=prompt if round_id == 1 else continuation,
            workdir=workdir, trace_dir=trace_dir,
            timeout_s=min(turn_seconds, remaining), persist_session=True,
            resume_session_id=session_id,
        )
        results.append(result)
        session_id = result.session_id or None
        atomic_json(
            workdir / "LANE_STATUS.json",
            {
                "lane": lane, "round": round_id, "session_id": session_id,
                "last_exit_code": result.exit_code,
                "last_timed_out": result.timed_out,
                "errors": result.errors,
                "updated_at": time.time(),
            },
        )
        if deadline_monotonic - time.monotonic() > 2:
            await asyncio.sleep(2)
    return results
