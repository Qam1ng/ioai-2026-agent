from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from .models import AgentResult
from .redaction import contains_secret, redact_text, redact_value
from .search_assets import (
    AssetIntegrityError,
    AssetSnapshot,
    make_tree_read_only,
    require_disjoint_paths,
    snapshot_assets,
    verify_snapshot,
)
from .search_prompts import (
    PromptSpecError,
    ResearchPlan,
    SearchPromptSet,
    fallback_research_plan,
    load_search_prompts,
    parse_research_plan,
    render_prompt,
)


class SearchAgentRunner(Protocol):
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
    ) -> AgentResult: ...

    async def terminate_all(self, *, grace_s: float = 8.0) -> None: ...


@dataclass(frozen=True, slots=True)
class SearchRunConfig:
    assets_dir: Path
    output_root: Path
    prompt_spec: Path
    duration_s: int
    competition_mode: str
    analyst_backend: str
    research_backends: tuple[str, str, str, str]
    prior_lessons_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.competition_mode not in {"practice", "formal"}:
            raise ValueError("competition_mode must be practice or formal")
        if self.analyst_backend not in {"claude", "codex"}:
            raise ValueError("analyst_backend must be claude or codex")
        if len(self.research_backends) != 4 or any(
            value not in {"claude", "codex"} for value in self.research_backends
        ):
            raise ValueError("research_backends must assign R1..R4 to claude/codex")

    def backend_for_research(self, research_id: str) -> str:
        return self.research_backends[int(research_id[1]) - 1]


class ControllerAuditLog:
    """Small append-only, hash-chained controller trajectory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._previous = "0" * 64

    def append(self, event: str, payload: dict[str, Any] | None = None) -> None:
        body = {
            "sequence": self._sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": redact_value(payload or {}),
            "previous_hash": self._previous,
        }
        canonical = json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        body["event_hash"] = hashlib.sha256(canonical).hexdigest()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._previous = body["event_hash"]
        self._sequence += 1


@dataclass(slots=True)
class SearchRunPaths:
    session: Path
    analyst_workdir: Path
    stage_a_dir: Path
    bundle_dir: Path
    research_work_root: Path
    context_dir: Path
    prompts_dir: Path
    traces_dir: Path
    status_path: Path


# 关键调度比例：总时长由调用方决定；这里只划分阶段，不写死正式赛时长。
_STAGE_A_FRACTION = 0.25
_RESEARCH_END_FRACTION = 0.70
_FINALIZE_RESERVE_FRACTION = 0.10
_FINALIZE_RESERVE_MIN_S = 20.0
_REQUIRED_STAGE_A = ("ASSET_MAP.md", "TASK_ANALYSIS_V1.md", "RESEARCH_CHARTERS.md")
_REQUIRED_FINAL = (
    "SEARCH_OUTPUT.md",
    "ASSET_MAP.md",
    "TASK_ANALYSIS.md",
    "RESEARCH_SYNTHESIS.md",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(redact_value(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _agent_result_dict(result: AgentResult) -> dict[str, Any]:
    value = asdict(result)
    value["workdir"] = str(result.workdir)
    value["trace_path"] = str(result.trace_path)
    return redact_value(value)


def _copy_read_only(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".md", ".json", ".jsonl", ".txt"}:
        text = source.read_text(encoding="utf-8")
        safe = redact_text(text)
        if safe != text:
            source.write_text(safe, encoding="utf-8")
            destination.write_text(safe, encoding="utf-8")
            destination.chmod(0o444)
            raise RuntimeError(
                f"credential-like content was redacted from agent artifact: {source.name}"
            )
    if not destination.exists():
        shutil.copy2(source, destination)
    destination.chmod(0o444)


def _iso_at(start_wall: datetime, offset_s: float) -> str:
    return (start_wall + timedelta(seconds=max(0.0, offset_s))).isoformat()


class SearchOrchestrator:
    def __init__(
        self,
        config: SearchRunConfig,
        *,
        runners: dict[str, SearchAgentRunner],
    ) -> None:
        self.config = config
        self.runners = runners
        require_disjoint_paths(config.assets_dir, config.output_root)
        if config.prior_lessons_dir is not None:
            require_disjoint_paths(config.prior_lessons_dir, config.output_root)
        required_backends = {config.analyst_backend, *config.research_backends}
        missing = required_backends - set(runners)
        if missing:
            raise ValueError(f"missing agent runners: {sorted(missing)}")

    def _make_paths(self) -> SearchRunPaths:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        session = self.config.output_root.expanduser().resolve() / (
            f"search-{stamp}-{uuid.uuid4().hex[:8]}"
        )
        analyst = session / "work" / "analyst"
        paths = SearchRunPaths(
            session=session,
            analyst_workdir=analyst,
            stage_a_dir=analyst / "stage_a_output",
            bundle_dir=analyst / "SEARCH_BUNDLE",
            research_work_root=session / "work" / "research",
            context_dir=session / "context",
            prompts_dir=session / "prompts",
            traces_dir=session / "trajectories",
            status_path=session / "RUN_STATUS.json",
        )
        for directory in (
            paths.stage_a_dir,
            paths.bundle_dir,
            paths.research_work_root,
            paths.context_dir,
            paths.prompts_dir,
            paths.traces_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths

    async def execute(self) -> Path:
        start_mono = time.monotonic()
        start_wall = datetime.now(UTC)
        global_deadline = start_mono + self.config.duration_s
        stage_a_deadline = start_mono + self.config.duration_s * _STAGE_A_FRACTION
        research_deadline = start_mono + self.config.duration_s * _RESEARCH_END_FRACTION
        paths = self._make_paths()
        audit = ControllerAuditLog(paths.session / "controller_trajectory.jsonl")
        status: dict[str, Any] = {
            "status": "running",
            "started_at": start_wall.isoformat(),
            "duration_s": self.config.duration_s,
            "phase_policy": {
                "stage_a_deadline_fraction": _STAGE_A_FRACTION,
                "research_deadline_fraction": _RESEARCH_END_FRACTION,
                "finalization_reserve_fraction_cap": _FINALIZE_RESERVE_FRACTION,
            },
            "competition_mode": self.config.competition_mode,
            "analyst_backend": self.config.analyst_backend,
            "research_backends": list(self.config.research_backends),
            "bundle_dir": str(paths.bundle_dir),
            "agent_results": [],
            "warnings": [],
            "errors": [],
        }
        _atomic_json(paths.status_path, status)
        audit.append("run_started", status)

        asset_snapshot: AssetSnapshot | None = None
        protected_before: dict[str, str] = {}
        prompts: SearchPromptSet | None = None
        analyst_session_id: str | None = None
        stage_a_result: AgentResult | None = None
        research_results: list[AgentResult] = []
        stage_b_result: AgentResult | None = None
        plan: ResearchPlan | None = None
        planned_report_ids: tuple[str, ...] = ()
        asset_snapshot_duration_s = 0.0
        asset_verified_final = False

        try:
            async with asyncio.timeout(self.config.duration_s):
                prompts = load_search_prompts(self.config.prompt_spec.resolve())
                shutil.copy2(
                    self.config.prompt_spec.resolve(), paths.prompts_dir / "PROMPT_SPEC.md"
                )
                audit.append(
                    "prompt_spec_loaded",
                    {
                        "path": str(self.config.prompt_spec.resolve()),
                        "sha256": _sha256_file(self.config.prompt_spec.resolve()),
                    },
                )

                assets_source = self.config.assets_dir.expanduser().resolve()
                require_disjoint_paths(assets_source, paths.session)
                asset_snapshot_started = time.monotonic()
                asset_snapshot = snapshot_assets(
                    assets_source,
                    paths.bundle_dir / "ORIGINAL_ASSETS",
                    sha256sums_path=paths.bundle_dir / "SHA256SUMS",
                    manifest_path=paths.bundle_dir / "ASSET_MANIFEST.json",
                    deadline_monotonic=global_deadline,
                )
                asset_snapshot_duration_s = time.monotonic() - asset_snapshot_started
                audit.append(
                    "asset_snapshot_created",
                    {
                        "file_count": len(asset_snapshot.entries),
                        "sha256sums": str(asset_snapshot.sha256sums_path),
                        "duration_s": asset_snapshot_duration_s,
                    },
                )

                lessons_dir = self._prepare_lessons(
                    paths, deadline_monotonic=global_deadline
                )
                stage_a_prompt = self._render_stage_a(
                    prompts,
                    paths,
                    lessons_dir,
                    deadline_iso=_iso_at(
                        start_wall, stage_a_deadline - start_mono
                    ),
                )
                self._save_prompt(paths, "analyst_stage_a", stage_a_prompt)
                stage_a_result = await self.runners[self.config.analyst_backend].run(
                    agent_id="analyst_stage_a",
                    role="task_analyst_stage_a",
                    prompt=stage_a_prompt,
                    workdir=paths.analyst_workdir,
                    trace_dir=paths.traces_dir,
                    timeout_s=max(0.1, stage_a_deadline - time.monotonic()),
                    persist_session=True,
                )
                status["agent_results"].append(_agent_result_dict(stage_a_result))
                audit.append("stage_a_finished", _agent_result_dict(stage_a_result))
                analyst_session_id = stage_a_result.session_id
                stage_a_ok, stage_a_issues = self._validate_stage_a(paths)
                status["warnings"].extend(stage_a_issues)
                if not stage_a_ok:
                    raise RuntimeError("stage A core artifacts are incomplete")
                self._freeze_stage_a(paths)

                research_window_s = max(0.0, research_deadline - time.monotonic())
                try:
                    plan = parse_research_plan(
                        (paths.stage_a_dir / "RESEARCH_CHARTERS.md").read_text(
                            encoding="utf-8"
                        )
                    )
                except PromptSpecError as error:
                    plan = fallback_research_plan(
                        prompts,
                        research_window_s=research_window_s,
                        reason=str(error),
                    )
                    status["warnings"].append(
                        f"research_plan invalid; runner fallback used: {error}"
                    )
                    fallback_path = paths.bundle_dir / "RUNNER_FALLBACK_PLAN.md"
                    fallback_path.write_text(
                        "# Runner fallback research plan\n\n"
                        f"Reason: {error}\n\n"
                        f"Enabled: {list(plan.enabled_ids)}\n",
                        encoding="utf-8",
                    )
                    fallback_path.chmod(0o444)
                planned_report_ids = plan.enabled_ids
                audit.append(
                    "research_plan_selected",
                    {
                        "enabled_ids": list(plan.enabled_ids),
                        "used_fallback": plan.used_fallback,
                        "fallback_reason": plan.fallback_reason,
                    },
                )

                research_results = await self._run_research_agents(
                    prompts,
                    plan,
                    paths,
                    lessons_dir,
                    start_wall=start_wall,
                    start_mono=start_mono,
                    research_deadline=research_deadline,
                    audit=audit,
                )
                status["agent_results"].extend(
                    _agent_result_dict(result) for result in research_results
                )
                report_issues = self._collect_research_reports(plan, paths)
                status["warnings"].extend(report_issues)

                protected_before = {
                    **{
                        f"stage_a/{key}": value
                        for key, value in _tree_hashes(paths.bundle_dir / "stage_a").items()
                    },
                    **{
                        f"research_raw/{key}": value
                        for key, value in _tree_hashes(
                            paths.bundle_dir / "research_raw"
                        ).items()
                    },
                    "SHA256SUMS": _sha256_file(paths.bundle_dir / "SHA256SUMS"),
                    "ASSET_MANIFEST.json": _sha256_file(
                        paths.bundle_dir / "ASSET_MANIFEST.json"
                    ),
                }
                fallback_path = paths.bundle_dir / "RUNNER_FALLBACK_PLAN.md"
                if fallback_path.is_file():
                    protected_before["RUNNER_FALLBACK_PLAN.md"] = _sha256_file(
                        fallback_path
                    )

                if analyst_session_id is None:
                    raise RuntimeError("stage A did not expose a resumable session id")
                finalization_reserve = min(
                    self.config.duration_s * _FINALIZE_RESERVE_FRACTION,
                    max(_FINALIZE_RESERVE_MIN_S, asset_snapshot_duration_s),
                )
                stage_b_timeout = max(
                    0.1, global_deadline - time.monotonic() - finalization_reserve
                )
                stage_b_prompt = self._render_stage_b(
                    prompts,
                    paths,
                    lessons_dir,
                    deadline_iso=_iso_at(
                        start_wall,
                        global_deadline - start_mono - finalization_reserve,
                    ),
                    plan=plan,
                    stage_b_timeout_s=stage_b_timeout,
                )
                self._save_prompt(paths, "analyst_stage_b", stage_b_prompt)
                stage_b_result = await self.runners[self.config.analyst_backend].run(
                    agent_id="analyst_stage_b",
                    role="task_analyst_stage_b",
                    prompt=stage_b_prompt,
                    workdir=paths.analyst_workdir,
                    trace_dir=paths.traces_dir,
                    timeout_s=stage_b_timeout,
                    persist_session=True,
                    resume_session_id=analyst_session_id,
                )
                status["agent_results"].append(_agent_result_dict(stage_b_result))
                audit.append("stage_b_finished", _agent_result_dict(stage_b_result))

                verify_snapshot(
                    asset_snapshot,
                    verify_source=True,
                    deadline_monotonic=global_deadline,
                )
                asset_verified_final = True
                self._verify_protected_inputs(paths, protected_before)
                final_issues = self._validate_final_bundle(
                    paths,
                    planned_report_ids=planned_report_ids,
                    research_results=research_results,
                    stage_a_result=stage_a_result,
                    stage_b_result=stage_b_result,
                    analyst_session_id=analyst_session_id,
                )
                status["warnings"].extend(final_issues)
                status["status"] = (
                    "complete"
                    if not final_issues and not status["warnings"]
                    else "degraded"
                )
                make_tree_read_only(paths.bundle_dir)
        except TimeoutError:
            status["status"] = "timeout"
            status["errors"].append("global Search module deadline reached")
            audit.append("global_timeout")
        except BaseException as error:
            status["status"] = "failed"
            status["errors"].append(f"{type(error).__name__}: {error}")
            audit.append(
                "run_failed", {"type": type(error).__name__, "message": str(error)}
            )
        finally:
            unique_runners = {
                id(runner): runner for runner in self.runners.values()
            }.values()
            await asyncio.gather(
                *(runner.terminate_all() for runner in unique_runners),
                return_exceptions=True,
            )
            if asset_snapshot is not None and asset_verified_final:
                status["asset_integrity"] = "verified"
            elif asset_snapshot is not None and time.monotonic() < global_deadline:
                try:
                    verify_snapshot(
                        asset_snapshot,
                        verify_source=True,
                        deadline_monotonic=global_deadline,
                    )
                    status["asset_integrity"] = "verified"
                except AssetIntegrityError as error:
                    status["asset_integrity"] = "failed"
                    status["errors"].append(str(error))
                    status["status"] = "failed"
                except TimeoutError:
                    status["asset_integrity"] = "not_reverified_deadline"
                    if status["status"] in {"running", "complete"}:
                        status["status"] = "timeout"
            elif asset_snapshot is not None:
                status["asset_integrity"] = "not_reverified_deadline"
            status["finished_at"] = datetime.now(UTC).isoformat()
            status["elapsed_s"] = time.monotonic() - start_mono
            status["planned_research_ids"] = list(planned_report_ids)
            status["analyst_session_id"] = analyst_session_id
            status["bundle_exists"] = paths.bundle_dir.is_dir()
            _atomic_json(paths.status_path, status)
            audit.append(
                "run_finished",
                {
                    "status": status["status"],
                    "elapsed_s": status["elapsed_s"],
                    "bundle_dir": str(paths.bundle_dir),
                },
            )
        return paths.session

    def _prepare_lessons(
        self,
        paths: SearchRunPaths,
        *,
        deadline_monotonic: float,
    ) -> Path:
        destination = paths.context_dir / "prior_lessons"
        if self.config.prior_lessons_dir is None:
            destination.mkdir()
            (destination / "README.md").write_text(
                "# 历史教训\n\n本次运行没有提供历史教训目录。\n",
                encoding="utf-8",
            )
            make_tree_read_only(destination)
            return destination
        source = self.config.prior_lessons_dir.expanduser().resolve()
        snapshot_assets(
            source,
            destination,
            sha256sums_path=paths.context_dir / "PRIOR_LESSONS_SHA256SUMS",
            manifest_path=paths.context_dir / "PRIOR_LESSONS_MANIFEST.json",
            deadline_monotonic=deadline_monotonic,
        )
        return destination

    def _base_values(
        self,
        prompts: SearchPromptSet,
        paths: SearchRunPaths,
        lessons_dir: Path,
        *,
        deadline_iso: str,
    ) -> dict[str, object]:
        return {
            "ioai_context_preamble": prompts.preamble.rstrip(),
            "deadline_iso": deadline_iso,
            "task_assets_dir": paths.bundle_dir / "ORIGINAL_ASSETS",
            "analysis_stage_a_dir": paths.stage_a_dir,
            "research_raw_dir": paths.bundle_dir / "research_raw",
            "final_bundle_dir": paths.bundle_dir,
            "module_time_budget_minutes": f"{self.config.duration_s / 60:.2f}",
            "prior_lessons_dir": lessons_dir,
            "competition_mode": self.config.competition_mode,
        }

    def _render_stage_a(
        self,
        prompts: SearchPromptSet,
        paths: SearchRunPaths,
        lessons_dir: Path,
        *,
        deadline_iso: str,
    ) -> str:
        return render_prompt(
            prompts.stage_a,
            self._base_values(
                prompts, paths, lessons_dir, deadline_iso=deadline_iso
            ),
        )

    def _render_stage_b(
        self,
        prompts: SearchPromptSet,
        paths: SearchRunPaths,
        lessons_dir: Path,
        *,
        deadline_iso: str,
        plan: ResearchPlan,
        stage_b_timeout_s: float,
    ) -> str:
        prompt = render_prompt(
            prompts.stage_b,
            self._base_values(
                prompts, paths, lessons_dir, deadline_iso=deadline_iso
            ),
        )
        if plan.used_fallback:
            prompt += (
                "\n\n## 外层 Runner 降级通知\n\n"
                "阶段 A 的 research_plan 无法机器解析，Runner 已按默认优先级降级。"
                f"实际启动：{list(plan.enabled_ids)}；原因：{plan.fallback_reason}。"
                "请以 research_raw/ 中实际存在的报告为准，并在 SEARCH_OUTPUT.md 顶部声明。\n"
            )
        if stage_b_timeout_s <= self.config.duration_s * 0.25:
            prompt += (
                "\n\n## 剩余时间降级通知\n\n"
                f"本阶段最多剩余 {stage_b_timeout_s / 60:.2f} 分钟。立即执行 Prompt 中的降级阶梯，"
                "优先保证四个顶层文件和完整性状态，不扩展新研究。\n"
            )
        return prompt

    async def _run_research_agents(
        self,
        prompts: SearchPromptSet,
        plan: ResearchPlan,
        paths: SearchRunPaths,
        lessons_dir: Path,
        *,
        start_wall: datetime,
        start_mono: float,
        research_deadline: float,
        audit: ControllerAuditLog,
    ) -> list[AgentResult]:
        async def run_one(research_id: str) -> AgentResult:
            charter = plan.charters[research_id]
            workdir = paths.research_work_root / research_id
            output_path = workdir / f"research_{research_id}.md"
            timeout_s = min(
                charter.time_budget_minutes * 60,
                max(0.1, research_deadline - time.monotonic()),
            )
            values = self._base_values(
                prompts,
                paths,
                lessons_dir,
                deadline_iso=_iso_at(
                    start_wall,
                    min(
                        research_deadline - start_mono,
                        time.monotonic() - start_mono + timeout_s,
                    ),
                ),
            )
            values.update(
                {
                    "research_id": research_id,
                    "research_time_budget_minutes": f"{timeout_s / 60:.2f}",
                    "research_workdir": workdir,
                    "research_output_path": output_path,
                    "research_charter": charter.text.rstrip(),
                }
            )
            prompt = render_prompt(prompts.research, values)
            self._save_prompt(paths, f"research_{research_id}", prompt)
            backend = self.config.backend_for_research(research_id)
            audit.append(
                "research_started",
                {
                    "research_id": research_id,
                    "backend": backend,
                    "timeout_s": timeout_s,
                },
            )
            result = await self.runners[backend].run(
                agent_id=f"research_{research_id}",
                role="research_agent",
                prompt=prompt,
                workdir=workdir,
                trace_dir=paths.traces_dir,
                timeout_s=timeout_s,
            )
            audit.append("research_finished", _agent_result_dict(result))
            return result

        if not plan.enabled_ids:
            return []
        tasks = [asyncio.create_task(run_one(research_id)) for research_id in plan.enabled_ids]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[AgentResult] = []
        for research_id, value in zip(plan.enabled_ids, raw_results, strict=True):
            if isinstance(value, BaseException):
                workdir = paths.research_work_root / research_id
                trace_path = paths.traces_dir / f"research_{research_id}.jsonl"
                results.append(
                    AgentResult(
                        agent_id=f"research_{research_id}",
                        role="research_agent",
                        workdir=workdir,
                        trace_path=trace_path,
                        exit_code=None,
                        timed_out=isinstance(value, TimeoutError),
                        duration_s=0.0,
                        errors=[f"{type(value).__name__}: {value}"],
                    )
                )
            else:
                results.append(value)
        return results

    def _validate_stage_a(self, paths: SearchRunPaths) -> tuple[bool, list[str]]:
        issues: list[str] = []
        for name in _REQUIRED_STAGE_A:
            path = paths.stage_a_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(f"missing stage A artifact: {name}")
        return not issues, issues

    def _freeze_stage_a(self, paths: SearchRunPaths) -> None:
        destination = paths.bundle_dir / "stage_a"
        destination.mkdir(exist_ok=True)
        for name in _REQUIRED_STAGE_A:
            _copy_read_only(paths.stage_a_dir / name, destination / name)
        destination.chmod(0o555)

    def _collect_research_reports(
        self, plan: ResearchPlan, paths: SearchRunPaths
    ) -> list[str]:
        destination = paths.bundle_dir / "research_raw"
        destination.mkdir(exist_ok=True)
        issues: list[str] = []
        for research_id in plan.enabled_ids:
            source = (
                paths.research_work_root
                / research_id
                / f"research_{research_id}.md"
            )
            if not source.is_file() or source.stat().st_size == 0:
                issues.append(f"missing research report: {research_id}")
                continue
            if len(source.read_text(encoding="utf-8").splitlines()) > 400:
                issues.append(f"research report exceeds 400 lines: {research_id}")
            _copy_read_only(source, destination / f"research_{research_id}.md")
        destination.chmod(0o555)
        return issues

    @staticmethod
    def _verify_protected_inputs(
        paths: SearchRunPaths, expected: dict[str, str]
    ) -> None:
        actual = {
            **{
                f"stage_a/{key}": value
                for key, value in _tree_hashes(paths.bundle_dir / "stage_a").items()
            },
            **{
                f"research_raw/{key}": value
                for key, value in _tree_hashes(
                    paths.bundle_dir / "research_raw"
                ).items()
            },
            "SHA256SUMS": _sha256_file(paths.bundle_dir / "SHA256SUMS"),
            "ASSET_MANIFEST.json": _sha256_file(
                paths.bundle_dir / "ASSET_MANIFEST.json"
            ),
        }
        fallback_path = paths.bundle_dir / "RUNNER_FALLBACK_PLAN.md"
        if fallback_path.is_file():
            actual["RUNNER_FALLBACK_PLAN.md"] = _sha256_file(fallback_path)
        if actual != expected:
            raise AssetIntegrityError("stage B changed protected bundle inputs")

    def _validate_final_bundle(
        self,
        paths: SearchRunPaths,
        *,
        planned_report_ids: tuple[str, ...],
        research_results: list[AgentResult],
        stage_a_result: AgentResult,
        stage_b_result: AgentResult,
        analyst_session_id: str,
    ) -> list[str]:
        issues: list[str] = []
        for name in _REQUIRED_FINAL:
            path = paths.bundle_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(f"missing final artifact: {name}")
        search_output = paths.bundle_dir / "SEARCH_OUTPUT.md"
        if search_output.is_file() and len(
            search_output.read_text(encoding="utf-8").splitlines()
        ) > 150:
            issues.append("SEARCH_OUTPUT.md exceeds 150 lines")

        reports = {
            path.stem.removeprefix("research_")
            for path in (paths.bundle_dir / "research_raw").glob("research_*.md")
        }
        if reports != set(planned_report_ids):
            issues.append(
                "planned and materialized research reports differ: "
                f"planned={planned_report_ids}, actual={sorted(reports)}"
            )
        for result in (stage_a_result, *research_results, stage_b_result):
            if result.exit_code != 0 or result.timed_out or result.errors:
                issues.append(
                    f"agent did not finish cleanly: {result.agent_id} "
                    f"exit={result.exit_code} timeout={result.timed_out} errors={result.errors}"
                )
        if stage_b_result.session_id != analyst_session_id:
            issues.append("stage B did not resume the stage A analyst session")

        for path in paths.bundle_dir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.relative_to(paths.bundle_dir).parts[0] == "ORIGINAL_ASSETS":
                continue
            if path.suffix.lower() not in {".md", ".json", ".jsonl", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if contains_secret(text):
                safe = redact_text(text)
                mode = path.stat().st_mode & 0o777
                path.chmod(0o644)
                path.write_text(safe, encoding="utf-8")
                path.chmod(mode)
                issues.append(f"credential-like content in bundle: {path.name}")
        return issues

    @staticmethod
    def _save_prompt(paths: SearchRunPaths, name: str, prompt: str) -> None:
        safe = redact_text(prompt)
        path = paths.prompts_dir / f"{name}.md"
        path.write_text(safe, encoding="utf-8")


__all__ = [
    "ControllerAuditLog",
    "SearchAgentRunner",
    "SearchOrchestrator",
    "SearchRunConfig",
]
