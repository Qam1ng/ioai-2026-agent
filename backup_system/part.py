"""One task: two solvers sharing a workspace, one manager, one submit loop."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .board import Board
from .config import BackupConfig
from .prompts import CONTINUATION, manager_prompt, solver_prompt
from .runners import ClaudeRunner, CodexRunner
from .submit import DryRunKaggle, Kaggle, floor_kernel


def log(slug: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {slug}: {message}", flush=True)


def _extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "decision" in value:
            return value
    raise ValueError("manager did not return a JSON decision object")


class TaskPart:
    """Everything that happens for one competition slug."""

    def __init__(
        self, *, slug: str, assets: Path, root: Path, config: BackupConfig,
        kaggle_slots: asyncio.Semaphore, deadline: float, live: bool,
        kaggle_user: str, max_submissions: int,
    ):
        self.slug = slug
        self.assets = Path(assets).resolve()
        self.config = config
        self.slots = kaggle_slots
        self.deadline = deadline
        self.live = live
        self.board = Board(root, slug=slug, max_submissions=max_submissions)
        self.traces = Path(root) / "traces"
        self.traces.mkdir(parents=True, exist_ok=True)
        self.kaggle = (
            Kaggle(slug=slug, root=Path(root) / "kaggle", user=kaggle_user,
                   assets=self.assets,
                   kernel_timeout_s=config.run.kernel_timeout_minutes * 60)
            if live else DryRunKaggle(slug, Path(root) / "kaggle", self.assets)
        )
        self._stop = asyncio.Event()
        self._runners: list = []
        # Agents read the assets through a stable name inside their workspace.
        link = self.board.work / "OFFICIAL_ASSETS"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(self.assets, target_is_directory=True)
        notes = self.board.work / "NOTES.md"
        if not notes.exists():
            notes.write_text(
                f"# {self.slug} · 共享笔记\n\n"
                "两个 agent 都往这里追加发现。不要重写别人的段落。\n",
                encoding="utf-8",
            )

    def _minutes_left(self) -> float:
        return max(0.0, (self.deadline - time.monotonic()) / 60)

    def _submission_cutoff(self) -> float:
        return self.deadline - self.config.run.submission_only_minutes_before_end * 60

    # ------------------------------------------------------------- solvers
    async def _solver_loop(self, agent: str, peer: str, runner, turn_minutes: float,
                           max_rounds: int) -> None:
        workdir = self.board.work
        session: str | None = None
        for round_id in range(1, max_rounds + 1):
            if self._stop.is_set():
                return
            remaining = self.deadline - time.monotonic()
            if remaining <= 90:
                return
            if session:
                prompt = (
                    f"{CONTINUATION}\n\n# 当前棋盘\n\n{self.board.digest()}\n\n"
                    f"剩余窗口约 {self._minutes_left():.0f} 分钟。"
                )
            else:
                prompt = solver_prompt(
                    agent=agent, peer=peer, slug=self.slug, workdir=workdir,
                    candidates_dir=self.board.candidates, assets_dir=self.assets,
                    feedback_path=self.board.feedback_path,
                    minutes_left=self._minutes_left(),
                    board=self.board.digest(),
                )
            try:
                result = await runner.run(
                    agent_id=f"{agent}-r{round_id:02d}", prompt=prompt,
                    workdir=workdir, trace_dir=self.traces,
                    timeout_s=min(turn_minutes * 60, remaining - 60),
                    resume=session, require_key=True,
                )
            except Exception as exc:  # noqa: BLE001
                self.board.event("solver_error", agent=agent, round=round_id,
                                 error=f"{type(exc).__name__}: {exc}")
                log(self.slug, f"{agent} round {round_id} crashed: {exc}")
                await asyncio.sleep(10)
                continue
            session = result.session_id or session
            self.board.event("solver_round", agent=agent, round=round_id,
                             exit_code=result.exit_code, timed_out=result.timed_out,
                             seconds=round(result.duration_s, 1),
                             errors=result.errors[:3])
            log(self.slug,
                f"{agent} round {round_id} done in {result.duration_s / 60:.1f}min "
                f"(exit={result.exit_code} timeout={result.timed_out})")
            await asyncio.sleep(3)

    # ------------------------------------------------------------- manager
    async def _choose(self, options: list[dict], runner) -> dict | None:
        listing = "\n".join(
            f"- `{item['candidate_id']}` — 作者 {item['author']}，"
            f"父版本 {item['parent'] or '无'}，加速器 {item['accelerator']}，"
            f"自报分 {item['self_reported']}\n    想法：{item['idea'] or '(未填写)'}"
            for item in options
        )
        prompt = manager_prompt(
            slug=self.slug, board=self.board.digest(), candidates=listing,
            minutes_left=self._minutes_left(), remaining=self.board.remaining(),
        )
        work = Path(self.board.root) / "manager"
        work.mkdir(parents=True, exist_ok=True)
        try:
            result = await runner.run(
                agent_id=f"manager-{int(time.time())}", prompt=prompt,
                workdir=work, trace_dir=self.traces,
                timeout_s=self.config.manager.timeout_seconds, require_key=True,
            )
            decision = _extract_json(result.result_text)
        except Exception as exc:  # noqa: BLE001
            self.board.event("manager_failed", error=f"{type(exc).__name__}: {exc}")
            return None
        if decision.get("decision") == "wait":
            self.board.event("manager_wait", reason=str(decision.get("reason", ""))[:300])
            return None
        chosen = decision.get("candidate_id")
        for option in options:
            if option["candidate_id"] == chosen:
                self.board.event("manager_pick", candidate_id=chosen,
                                 reason=str(decision.get("reason", ""))[:300])
                return option
        self.board.event("manager_invalid", candidate_id=str(chosen)[:100])
        return None

    # -------------------------------------------------------------- submit
    def _write_floor(self) -> dict | None:
        directory = self.board.candidates / "floor-insurance"
        if directory.exists():
            return None
        staging = self.board.candidates / ".staging-floor"
        if staging.exists():
            return None
        staging.mkdir(parents=True)
        (staging / "kernel.py").write_text(floor_kernel(), encoding="utf-8")
        (staging / "meta.json").write_text(json.dumps({
            "author": "watchdog", "accelerator": "cpu", "parent": "",
            "idea": "insurance: copy the official template through unchanged",
            "self_reported_score": None,
        }, ensure_ascii=False), encoding="utf-8")
        staging.rename(directory)
        (directory / "READY").touch()
        self.board.event("floor_queued")
        log(self.slug, "floor watchdog queued the insurance candidate")
        return None

    async def _spend(self, candidate: dict, kind: str) -> None:
        attempt = self.board.reserve(candidate, kind)
        if attempt is None:
            return
        message = (
            f"[bk:{attempt.submission_id}] {kind} {candidate['author']} "
            f"{candidate['candidate_id']}"
        )[:140]
        async with self.slots:
            if time.monotonic() >= self._submission_cutoff():
                self.board.update(attempt.submission_id, status="deferred",
                                  detail="submission window closed",
                                  retry_after=time.time() + 10 ** 6)
                return
            log(self.slug, f"submitting {candidate['candidate_id']} ({kind})")
            outcome = await asyncio.to_thread(
                self.kaggle.submit, candidate, attempt.submission_id, message
            )
        retry_after = (
            time.time() + self.config.run.retry_backoff_seconds
            if outcome.status in {"retryable", "deferred"} else 0.0
        )
        status = "deferred" if outcome.status == "retryable" else outcome.status
        self.board.update(
            attempt.submission_id, status=status, detail=outcome.detail[:800],
            kernel_ref=outcome.kernel_ref, kernel_version=outcome.kernel_version,
            retry_after=retry_after,
        )
        self.board.event("submit_result", submission_id=attempt.submission_id,
                         candidate_id=candidate["candidate_id"], status=status,
                         detail=outcome.detail[:300])
        self.board.feedback({
            "event": "submission_result",
            "candidate_id": candidate["candidate_id"],
            "author": candidate["author"], "kind": kind, "status": status,
            "detail": outcome.detail[:500],
            "note": ("这个候选已经交出去了，等分数。"
                     if status == "submitted"
                     else "这个候选没能提交成功，原因见 detail；修好再发一个新 id。"),
        })
        log(self.slug, f"  -> {status}: {outcome.detail[:120]}")

    async def _submit_loop(self, manager_runner) -> None:
        floor_at = time.monotonic() + self.config.run.floor_after_minutes * 60
        last_manager = 0.0
        while not self._stop.is_set() and time.monotonic() < self.deadline:
            await asyncio.sleep(self.config.run.poll_seconds)
            if time.monotonic() >= self._submission_cutoff():
                continue
            if self.board.remaining() <= 0:
                continue
            options = self.board.submittable(
                retry_attempts=self.config.run.retry_attempts
            )
            if not options:
                if time.monotonic() > floor_at and self.board.consumed() == 0:
                    self._write_floor()
                continue

            # The first submission always goes out immediately and without asking
            # anyone: proving the chain works is worth more than choosing well,
            # and there is nothing to choose between yet.
            if self.board.consumed() == 0 and self.board.inflight() == 0:
                await self._spend(options[0], "floor")
                continue

            if time.monotonic() - last_manager < self.config.manager.interval_seconds:
                continue
            last_manager = time.monotonic()
            chosen = await self._choose(options, manager_runner)
            if chosen is None:
                continue
            await self._spend(chosen, "milestone")

    # --------------------------------------------------------------- score
    async def _score_loop(self) -> None:
        while not self._stop.is_set() and time.monotonic() < self.deadline + 120:
            await asyncio.sleep(45)
            try:
                scores = await asyncio.to_thread(self.kaggle.scores)
            except Exception:  # noqa: BLE001
                continue
            for attempt in self.board.attempts():
                score = scores.get(attempt.submission_id)
                if score is None or attempt.lb_score == score:
                    continue
                self.board.update(attempt.submission_id, lb_score=score,
                                  status="scored")
                self.board.event("scored", submission_id=attempt.submission_id,
                                 candidate_id=attempt.candidate_id, lb_score=score)
                self.board.feedback({
                    "event": "leaderboard_score",
                    "candidate_id": attempt.candidate_id,
                    "author": attempt.author, "idea": attempt.idea,
                    "leaderboard_score": score,
                    "note": "这是真实排行榜分数，是本题唯一可信的评判。",
                })
                log(self.slug, f"  LB {attempt.candidate_id} = {score}")

    # ----------------------------------------------------------------- run
    def _solver_runner(self, agent):
        """Build whichever CLI this lane is configured to drive."""
        if agent.runner == "codex":
            return CodexRunner(
                binary=agent.binary, model=agent.model, effort=agent.effort,
                api_key_env=self.config.run.api_key_env, base_url=agent.base_url,
            )
        return ClaudeRunner(
            binary=agent.binary, model=agent.model, effort=agent.effort,
            api_key_env=self.config.run.api_key_env, base_url=agent.base_url,
        )

    async def run(self) -> dict:
        codex = self._solver_runner(self.config.codex)
        claude = self._solver_runner(self.config.claude)
        manager = ClaudeRunner(
            binary=self.config.manager.binary, model=self.config.manager.model,
            effort=self.config.manager.effort,
            api_key_env=self.config.run.api_key_env,
            base_url=self.config.manager.base_url,
            # No tools at all: the board and the candidate list are already in
            # the prompt, and a manager that can run Bash could find its own
            # way to Kaggle.
            tools="",
        )
        self._runners = [codex, claude, manager]
        tasks = [
            asyncio.create_task(self._solver_loop(
                "codex", "claude", codex, self.config.codex.turn_minutes,
                self.config.codex.max_rounds)),
            asyncio.create_task(self._solver_loop(
                "claude", "codex", claude, self.config.claude.turn_minutes,
                self.config.claude.max_rounds)),
            asyncio.create_task(self._submit_loop(manager)),
            asyncio.create_task(self._score_loop()),
        ]
        try:
            await asyncio.sleep(max(0.0, self.deadline - time.monotonic()))
        finally:
            self._stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*(r.terminate_all() for r in self._runners),
                                 return_exceptions=True)
        best = self.board.best()
        summary = {
            "slug": self.slug,
            "best_lb": best.lb_score if best else None,
            "best_candidate": best.candidate_id if best else None,
            "best_author": best.author if best else None,
            "submissions_consumed": self.board.consumed(),
            "candidates_produced": len(self.board.ready_candidates()),
        }
        (Path(self.board.root) / "SUMMARY.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
