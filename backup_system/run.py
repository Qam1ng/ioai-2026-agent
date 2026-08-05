"""Up to three tasks in parallel, in one process, sharing one Kaggle limit.

The operator's only lever is pasting each task's exact Starter prompt. From
there the system resolves the task itself, downloads the data itself, and never
asks a human for a value it could have looked up.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import BackupConfig
from .intake import IntakeError, StarterTask, build_task, extract_json
from .part import TaskPart, log
from .prompts import INTAKE_PROMPT
from .runners import ClaudeRunner
from .submit import Kaggle


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:80]


class BackupRun:
    def __init__(
        self, *, config: BackupConfig, live: bool, kaggle_user: str,
        run_id: str = "", fallback_minutes: float | None = None,
    ):
        if live and not kaggle_user:
            raise ValueError("--kaggle-user is required with --live")
        self.config = config
        self.live = live
        self.kaggle_user = kaggle_user
        self.fallback_minutes = fallback_minutes
        stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.session = config.run.workspace_root / safe_id(stamp)
        if self.session.exists():
            raise FileExistsError(f"session already exists: {self.session}")
        self.session.mkdir(parents=True)
        self.intake_dir = self.session / "intake"
        self.intake_dir.mkdir()

    # ------------------------------------------------------------- intake
    def _intake_runner(self) -> ClaudeRunner:
        manager = self.config.manager
        return ClaudeRunner(
            binary=manager.binary, model=manager.model, effort=manager.effort,
            api_key_env=self.config.run.api_key_env, base_url=manager.base_url,
            # Reading text and emitting JSON needs no tools, and a task-intake
            # agent holding Bash is one that could start solving the problem.
            tools="",
        )

    async def resolve(self, index: int, prompt_text: str) -> StarterTask:
        """Starter prompt -> the few typed facts the deterministic layer needs."""
        runner = self._intake_runner()
        work = self.intake_dir / f"{index:02d}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "STARTER_PROMPT.txt").write_text(prompt_text, encoding="utf-8")
        try:
            result = await runner.run(
                agent_id=f"intake-{index:02d}",
                prompt=INTAKE_PROMPT % {"starter_prompt": prompt_text},
                workdir=work, trace_dir=self.session / "traces",
                timeout_s=self.config.manager.timeout_seconds, require_key=True,
            )
        finally:
            await runner.terminate_all()
        parsed = extract_json(result.result_text)
        (work / "intake.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

        slug = str(parsed.get("slug", "")).strip().lower()
        if not slug:
            raise IntakeError("the intake agent did not find a competition slug")
        limits = {"num_allowed_now": self.config.run.max_submissions, "num_today": 0}
        if self.live:
            probe = Kaggle(
                slug=slug, root=work / "kaggle", user=self.kaggle_user,
                assets=work, kernel_timeout_s=60,
            )
            limits = probe.submission_limits()
            if not limits.get("ok"):
                # A wrong slug fails here rather than after a kernel has run.
                raise IntakeError(
                    f"Kaggle would not confirm competition {slug!r}: "
                    f"{limits.get('error', 'no submission limits returned')}. "
                    "Either the slug was misread or the account has not joined it"
                )
            limits["num_allowed_now"] = min(
                int(limits["num_allowed_now"]), self.config.run.max_submissions
            )
        task = build_task(
            parsed, raw_prompt=prompt_text, limits=limits,
            fallback_minutes=self.fallback_minutes,
        )
        (work / "resolved_task.json").write_text(
            json.dumps(task.summary(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        return task

    def fetch_data(self, task: StarterTask, root: Path) -> Path:
        """Download the competition data ourselves.

        The rules list this as step one for the agent, not something the
        operator prepares and hands over.
        """
        assets = root / "OFFICIAL_ASSETS"
        if not self.live:
            assets.mkdir(parents=True, exist_ok=True)
            return assets
        kaggle = Kaggle(
            slug=task.slug, root=root / "kaggle", user=self.kaggle_user,
            assets=assets, kernel_timeout_s=60,
        )
        ok, detail = kaggle.download_data(assets)
        if not ok:
            raise RuntimeError(f"could not download data for {task.slug}: {detail}")
        log(task.slug, f"downloaded competition data: {detail}")
        return assets

    # ---------------------------------------------------------------- run
    async def execute(
        self,
        prompts: list[str] | None = None,
        prepared: list[tuple[StarterTask, Path]] | None = None,
    ) -> dict:
        started = time.monotonic()
        status = {
            "status": "running", "session": str(self.session),
            "started_at": time.time(), "live": self.live,
        }
        self._write_status(status)

        resolved: list[tuple[StarterTask, Path]] = list(prepared or [])
        for index, text in enumerate(prompts or [], start=1):
            task = await self.resolve(index, text)
            log(task.slug, "resolved from Starter prompt: "
                           + json.dumps(task.summary(), ensure_ascii=False))
            root = self.session / safe_id(task.slug)
            root.mkdir(parents=True, exist_ok=True)
            resolved.append((task, self.fetch_data(task, root)))

        if not resolved:
            raise ValueError("no tasks were resolved")

        slots = asyncio.Semaphore(self.config.run.kaggle_concurrency)
        parts: list[TaskPart] = []
        for task, assets in resolved:
            root = self.session / safe_id(task.slug)
            root.mkdir(parents=True, exist_ok=True)
            part = TaskPart(
                task=task, assets=assets, root=root, config=self.config,
                kaggle_slots=slots,
                # Each task keeps its own deadline: they are separate
                # competitions and can close at different times.
                deadline=started + max(60.0, task.minutes_left() * 60),
                live=self.live, kaggle_user=self.kaggle_user,
                max_submissions=task.max_submissions,
            )
            parts.append(part)
            log(task.slug,
                f"ready | budget={task.max_submissions} "
                f"kernel_timeout={task.kernel_timeout_seconds}s "
                f"deadline_in={task.minutes_left():.0f}min "
                f"({task.deadline_source})")

        status["tasks"] = [task.summary() for task, _ in resolved]
        self._write_status(status)

        latest = max(part.deadline for part in parts)
        heartbeat = asyncio.create_task(self._heartbeat(parts, latest))
        results = await asyncio.gather(
            *(part.run() for part in parts), return_exceptions=True
        )
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)

        summaries = []
        for part, result in zip(parts, results):
            summaries.append(
                result if isinstance(result, dict) else
                {"slug": part.slug, "error": f"{type(result).__name__}: {result}"}
            )
        status.update({"status": "complete", "finished_at": time.time(),
                       "results": summaries})
        self._write_status(status)
        return status

    def _write_status(self, status: dict) -> None:
        (self.session / "RUN_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _heartbeat(self, parts: list[TaskPart], deadline: float) -> None:
        while time.monotonic() < deadline:
            await asyncio.sleep(60)
            for part in parts:
                best = part.board.best()
                log(part.slug,
                    f"T-{(part.deadline - time.monotonic()) / 60:5.1f}min | "
                    f"candidates={len(part.board.ready_candidates())} "
                    f"used={part.board.consumed()}/{part.board.max_submissions} "
                    f"best={best.lb_score if best else None}")
