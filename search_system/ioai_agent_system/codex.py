from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from pathlib import Path

from .models import AgentResult
from .redaction import redact_text as redact


class CodexRunner:
    """Run isolated Codex CLI turns with native live web search enabled."""

    def __init__(
        self,
        *,
        codex_bin: Path,
        model: str,
        effort: str,
    ) -> None:
        self.codex_bin = codex_bin
        self.model = model
        self.effort = effort
        self._active: dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()

    def argv(
        self,
        *,
        workdir: Path,
        last_message_path: Path,
        persist_session: bool = False,
        resume_session_id: str | None = None,
    ) -> list[str]:
        # --search 使用 Codex 原生 web_search，不需要第三方 Search API。
        argv = [
            str(self.codex_bin),
            "--search",
            "--model",
            self.model,
            "--sandbox",
            "workspace-write",
            "--ask-for-approval",
            "never",
            "--cd",
            str(workdir),
            "--config",
            f'model_reasoning_effort="{self.effort}"',
            "exec",
        ]
        if resume_session_id is not None:
            argv.extend(
                [
                    "resume",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--json",
                    "--output-last-message",
                    str(last_message_path),
                    resume_session_id,
                    "-",
                ]
            )
        else:
            initial_args = [
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--json",
                "--output-last-message",
                str(last_message_path),
            ]
            if not persist_session:
                initial_args.append("--ephemeral")
            initial_args.append("-")
            argv.extend(initial_args)
        return argv

    @staticmethod
    def environment() -> dict[str, str]:
        # Codex 认证仍由本机 CLI 自己读取；工具 shell 由 Codex sandbox 管理。
        passthrough = {
            "PATH",
            "HOME",
            "CODEX_HOME",
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
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["TOKENIZERS_PARALLELISM"] = "false"
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
        last_message_path = trace_dir / f"{agent_id}.last_message.md"
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *self.argv(
                workdir=workdir,
                last_message_path=last_message_path,
                persist_session=persist_session,
                resume_session_id=resume_session_id,
            ),
            cwd=str(workdir),
            env=self.environment(),
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

        session_id: str | None = None
        resolved_model: str | None = None
        result_text = ""
        errors: list[str] = []
        timed_out = False

        async def consume_stdout() -> None:
            nonlocal session_id, resolved_model, result_text
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
                    if event.get("type") == "thread.started":
                        session_id = str(event.get("thread_id") or "") or None
                    model_value = event.get("model") or event.get("model_name")
                    if isinstance(model_value, str):
                        resolved_model = model_value
                    item = event.get("item")
                    if (
                        event.get("type") == "item.completed"
                        and isinstance(item, dict)
                        and item.get("type") == "agent_message"
                    ):
                        text_value = item.get("text")
                        if isinstance(text_value, str):
                            result_text = text_value

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

        if last_message_path.is_file():
            result_text = redact(last_message_path.read_text(encoding="utf-8"))
            last_message_path.write_text(result_text, encoding="utf-8")
        if session_id is None:
            errors.append("Codex JSONL did not expose a thread_id")
        if resume_session_id is not None and session_id != resume_session_id:
            errors.append(
                "resumed session mismatch: "
                f"expected={resume_session_id!r}, got={session_id!r}"
            )
        if resolved_model is not None and resolved_model != self.model:
            errors.append(
                f"resolved model mismatch: expected={self.model!r}, got={resolved_model!r}"
            )

        return AgentResult(
            agent_id=agent_id,
            role=role,
            workdir=workdir,
            trace_path=trace_path,
            exit_code=process.returncode,
            timed_out=timed_out,
            duration_s=time.monotonic() - started,
            resolved_model=resolved_model,
            session_id=session_id,
            result_text=result_text,
            errors=errors,
        )

    async def terminate_all(self, *, grace_s: float = 8.0) -> None:
        async with self._lock:
            processes = list(self._active.values())
        await asyncio.gather(
            *(self._terminate_process(process, grace_s=grace_s) for process in processes),
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


__all__ = ["CodexRunner"]
