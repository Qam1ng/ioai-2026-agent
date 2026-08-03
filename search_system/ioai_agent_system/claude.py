from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import stat
import time
from pathlib import Path
from typing import Any

from .models import AgentResult
from .redaction import redact_text as redact


class SecretBroker:
    """Serve one credential from memory over a mode-0600 Unix socket."""

    def __init__(self, secret: str, socket_path: Path):
        if not secret or "\n" in secret:
            raise ValueError("invalid API credential")
        self._secret = secret
        self.socket_path = socket_path
        self._server: asyncio.AbstractServer | None = None

    async def __aenter__(self) -> "SecretBroker":
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(self._handle, path=str(self.socket_path))
        self.socket_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        self._secret = ""

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await asyncio.wait_for(reader.readline(), timeout=3)
            if request == b"GET\n":
                writer.write(self._secret.encode("utf-8") + b"\n")
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


class ClaudeRunner:
    def __init__(
        self,
        *,
        claude_bin: Path,
        python_bin: Path,
        package_root: Path,
        model: str,
        effort: str,
        secret_socket: Path,
        max_budget_usd: float = 20.0,
        permission_mode: str = "bypassPermissions",
        tools: str = "Bash,Read,Edit,Write,Glob,Grep,WebSearch,WebFetch",
    ):
        self.claude_bin = claude_bin
        self.python_bin = python_bin
        self.package_root = package_root
        self.model = model
        self.effort = effort
        self.secret_socket = secret_socket
        self.max_budget_usd = max_budget_usd
        self.permission_mode = permission_mode
        self.tools = tools
        self._active: dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()

    def _settings(self) -> str:
        helper = self.package_root / "ioai_agent_system" / "secret_helper.py"
        command = f'"{self.python_bin}" "{helper}"'
        return json.dumps({"apiKeyHelper": command}, separators=(",", ":"))

    def argv(
        self,
        *,
        persist_session: bool = False,
        resume_session_id: str | None = None,
    ) -> list[str]:
        argv = [
            str(self.claude_bin),
            "--bare",
            "-p",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            self.permission_mode,
            "--tools",
            self.tools,
            "--max-budget-usd",
            str(self.max_budget_usd),
            "--prompt-suggestions",
            "false",
            "--settings",
            self._settings(),
        ]
        if resume_session_id is not None:
            argv.extend(["--resume", resume_session_id])
        elif not persist_session:
            argv.append("--no-session-persistence")
        return argv

    def environment(self, workdir: Path | None = None) -> dict[str, str]:
        # Do not give an autonomous Builder the operator's full login
        # environment or HOME (which may contain Kaggle and other credentials).
        # Claude itself authenticates only through the short-lived helper socket.
        passthrough = {
            "PATH",
            "USER",
            "LOGNAME",
            "SHELL",
            "LANG",
            "LC_ALL",
            "TERM",
            "TMPDIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
        }
        env = {key: value for key, value in os.environ.items() if key in passthrough}
        if workdir is not None:
            isolated_home = workdir / ".agent_home"
            config_dir = workdir / ".claude_config"
            isolated_home.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)
            env["HOME"] = str(isolated_home)
            env["CLAUDE_CONFIG_DIR"] = str(config_dir)
        env["IOAI_SECRET_SOCKET"] = str(self.secret_socket)
        env["CLAUDE_CODE_API_KEY_HELPER_TTL_MS"] = "300000"
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["VECLIB_MAXIMUM_THREADS"] = "2"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION"] = "0"
        env["CLAUDE_CODE_BG_CLASSIFIER_MODEL"] = self.model
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = self.model
        env["CLAUDE_CODE_AUTO_MODE_MODEL"] = self.model
        return env

    async def run(
        self,
        *,
        agent_id: str,
        role: str,
        prompt: str,
        workdir: Path,
        trace_dir: Path,
        timeout_s: float,
        persist_session: bool = False,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        workdir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{agent_id}.jsonl"
        stderr_path = trace_dir / f"{agent_id}.stderr.log"
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *self.argv(
                persist_session=persist_session,
                resume_session_id=resume_session_id,
            ),
            cwd=str(workdir),
            env=self.environment(workdir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=64 * 1024 * 1024,
        )
        async with self._lock:
            self._active[agent_id] = process

        if process.stdin is not None:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

        init_model: str | None = None
        session_id: str | None = None
        result_text = ""
        cost_usd = 0.0
        errors: list[str] = []
        timed_out = False

        async def consume_stdout() -> None:
            nonlocal init_model, session_id, result_text, cost_usd
            assert process.stdout is not None
            with trace_path.open("w", encoding="utf-8") as trace:
                while line := await process.stdout.readline():
                    safe_line = redact(line.decode("utf-8", errors="replace").rstrip("\n"))
                    trace.write(safe_line + "\n")
                    trace.flush()
                    try:
                        event = json.loads(safe_line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "system" and event.get("subtype") == "init":
                        init_model = event.get("model")
                        session_id = event.get("session_id")
                    elif event.get("type") == "result":
                        result_text = str(event.get("result", ""))
                        cost_usd = float(event.get("total_cost_usd") or 0.0)
                        if event.get("is_error"):
                            errors.extend(str(value) for value in event.get("errors", []))

        async def consume_stderr() -> None:
            assert process.stderr is not None
            with stderr_path.open("w", encoding="utf-8") as stream:
                while line := await process.stderr.readline():
                    stream.write(redact(line.decode("utf-8", errors="replace")))
                    stream.flush()

        readers = [asyncio.create_task(consume_stdout()), asyncio.create_task(consume_stderr())]
        try:
            async with asyncio.timeout(max(0.1, timeout_s)):
                await process.wait()
                await asyncio.gather(*readers)
        except TimeoutError:
            timed_out = True
            await self._terminate_process(process)
            await asyncio.gather(*readers, return_exceptions=True)
        except asyncio.CancelledError:
            timed_out = True
            await self._terminate_process(process)
            await asyncio.gather(*readers, return_exceptions=True)
            raise
        finally:
            async with self._lock:
                self._active.pop(agent_id, None)

        if init_model != self.model:
            errors.append(f"resolved model mismatch: expected={self.model!r}, got={init_model!r}")
        if resume_session_id is not None and session_id != resume_session_id:
            errors.append(
                "resumed session mismatch: "
                f"expected={resume_session_id!r}, got={session_id!r}"
            )

        return AgentResult(
            agent_id=agent_id,
            role=role,
            workdir=workdir,
            trace_path=trace_path,
            exit_code=process.returncode,
            timed_out=timed_out,
            duration_s=time.monotonic() - started,
            resolved_model=init_model,
            session_id=session_id,
            cost_usd=cost_usd,
            result_text=result_text,
            errors=errors,
        )

    async def preflight(self, workdir: Path, trace_dir: Path, timeout_s: float = 45) -> AgentResult:
        result = await self.run(
            agent_id="preflight",
            role="preflight",
            prompt="Reply with exactly OK.",
            workdir=workdir,
            trace_dir=trace_dir,
            timeout_s=timeout_s,
        )
        if result.exit_code != 0 or result.resolved_model != self.model or result.result_text.strip() != "OK":
            raise RuntimeError(
                f"Claude preflight failed: exit={result.exit_code}, "
                f"model={result.resolved_model!r}, errors={result.errors}"
            )
        return result

    async def terminate_all(self, *, grace_s: float = 8.0) -> None:
        async with self._lock:
            processes = list(self._active.values())
        await asyncio.gather(
            *(self._terminate_process(proc, grace_s=grace_s) for proc in processes),
            return_exceptions=True,
        )

    @staticmethod
    async def _terminate_process(
        process: asyncio.subprocess.Process,
        *,
        grace_s: float = 8.0,
    ) -> None:
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=max(0.0, grace_s))
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                await process.wait()
