"""Three tasks in parallel, in one process, sharing one Kaggle concurrency limit."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import BackupConfig
from .part import TaskPart, log
from .submit import Kaggle


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:80]


class BackupRun:
    def __init__(
        self, *, config: BackupConfig, tasks: list[tuple[str, Path]],
        minutes: float, live: bool, kaggle_user: str, run_id: str = "",
    ):
        if not tasks:
            raise ValueError("at least one --task is required")
        if minutes <= 0:
            raise ValueError("--minutes must be positive")
        if live and not kaggle_user:
            raise ValueError("--kaggle-user is required with --live")
        self.config = config
        self.tasks = tasks
        self.minutes = minutes
        self.live = live
        self.kaggle_user = kaggle_user
        stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.session = config.run.workspace_root / safe_id(stamp)
        if self.session.exists():
            raise FileExistsError(f"session already exists: {self.session}")
        self.session.mkdir(parents=True)

    def _budget(self, slug: str) -> int:
        """Never invent a budget: if Kaggle will not tell us, fail closed."""
        configured = self.config.run.max_submissions
        if not self.live:
            return configured
        probe = Kaggle(
            slug=slug, root=self.session / safe_id(slug) / "kaggle",
            user=self.kaggle_user, assets=Path("."),
            kernel_timeout_s=self.config.run.kernel_timeout_minutes * 60,
        )
        remaining = probe.remaining_today()
        if remaining is None:
            raise RuntimeError(
                f"could not read Kaggle's remaining-today quota for {slug}; "
                "refusing to guess a fresh budget in live mode"
            )
        if remaining <= 0:
            raise RuntimeError(f"Kaggle reports zero remaining submissions for {slug}")
        return min(configured, remaining)

    async def execute(self) -> dict:
        started = time.monotonic()
        deadline = started + self.minutes * 60
        slots = asyncio.Semaphore(self.config.run.kaggle_concurrency)
        status = {
            "status": "running",
            "session": str(self.session),
            "started_at": time.time(),
            "minutes": self.minutes,
            "live": self.live,
            "tasks": [slug for slug, _ in self.tasks],
        }
        (self.session / "RUN_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

        parts = []
        for slug, assets in self.tasks:
            part = TaskPart(
                slug=slug, assets=assets, root=self.session / safe_id(slug),
                config=self.config, kaggle_slots=slots, deadline=deadline,
                live=self.live, kaggle_user=self.kaggle_user,
                max_submissions=self._budget(slug),
            )
            parts.append(part)
            log(slug, f"ready | budget={part.board.max_submissions} "
                      f"assets={assets}")

        heartbeat = asyncio.create_task(self._heartbeat(parts, deadline))
        results = await asyncio.gather(
            *(part.run() for part in parts), return_exceptions=True
        )
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)

        summaries = []
        for part, result in zip(parts, results):
            if isinstance(result, BaseException):
                summaries.append({"slug": part.slug,
                                  "error": f"{type(result).__name__}: {result}"})
            else:
                summaries.append(result)
        status.update({"status": "complete", "finished_at": time.time(),
                       "results": summaries})
        (self.session / "RUN_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    async def _heartbeat(self, parts: list[TaskPart], deadline: float) -> None:
        while time.monotonic() < deadline:
            await asyncio.sleep(60)
            for part in parts:
                best = part.board.best()
                log(part.slug,
                    f"T-{(deadline - time.monotonic()) / 60:5.1f}min | "
                    f"candidates={len(part.board.ready_candidates())} "
                    f"used={part.board.consumed()}/{part.board.max_submissions} "
                    f"best={best.lb_score if best else None}")
