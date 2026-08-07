from __future__ import annotations

import asyncio
import hashlib
import contextlib
import json
import os
import re
import signal
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from search_system.ioai_agent_system.search_assets import (
    AssetSnapshot,
    snapshot_assets,
    verify_snapshot,
)
from search_system.ioai_agent_system.search_orchestrator import (
    SearchOrchestrator,
    SearchRunConfig,
)

from agent.tools.registry import _kaggle_bin

from .anthropic_shim import AnthropicShim
from .broker import SubmissionBroker
from .calibration import FeedbackCalibrator
from .config import SystemConfig
from .day_gate import DayResourceGate
from .evaluation import freeze_contract

#: In unlimited mode a lane is told this much time per round; it delivers a
#: candidate, the loop restarts it with the continuation prompt, and that
#: repeats until someone stops the run. Telling it "unbounded" instead makes
#: agents sandbag — a rolling budget keeps candidates arriving.
LANE_ROLLING_MINUTES = 60.0
from .io import append_jsonl, atomic_json, read_json, sha256_file, tree_hash
from .kaggle import DryRunAdapter, KaggleAdapter
from .kaggle_gateway import ALLOWED as GATEWAY_ALLOWED
from .kaggle_gateway import KaggleReadOnlyGateway
from .prompts import continuation_prompt, direct_prompt
from .registry import CandidateRegistry, candidate_fingerprint
from .runners import (
    ClaudeSubscriptionRunner,
    OpenRouterCodexRunner,
    run_lane_loop,
)
from .selection import SelectionManager
from .security import AgentSandbox, build_agent_sandbox, verify_agent_sandbox

ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_SEARCH = (
    "SEARCH_OUTPUT.md", "ASSET_MAP.md", "TASK_ANALYSIS.md", "RESEARCH_SYNTHESIS.md"
)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:80]


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(Path(target).resolve(), target_is_directory=Path(target).is_dir())


def _verify_sha256sums(root: Path, sha_path: Path) -> None:
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        if sha256_file(Path(root) / relative) != digest:
            raise ValueError(f"asset hash mismatch: {relative}")


class FinalController:
    def __init__(
        self, *, config: SystemConfig, slug: str, assets_dir: Path,
        duration_minutes: float, competition_mode: str, kaggle_user: str,
        deadline_epoch: float = 0.0,
        live: bool, run_id: str = "", resource_pool_id: str = "",
        floor_group_id: str = "", day_slugs: tuple[str, ...] = (),
        starter_prompt: str = "", continuation_prompt_text: str = "",
    ):
        if duration_minutes < 0:
            raise ValueError("duration_minutes must not be negative")
        # 0 means "run until stopped": no window at all. The only hard clock is
        # then --deadline (the competition's own end), and the only other way
        # out is an explicit stop — control/STOP or SIGTERM/SIGINT.
        self.unlimited = duration_minutes <= 0
        if competition_mode not in {"practice", "formal"}:
            raise ValueError("competition_mode must be practice or formal")
        if live and not kaggle_user:
            raise ValueError("--kaggle-user is required with --live")
        if live and (
            not resource_pool_id or not floor_group_id or slug not in day_slugs
        ):
            raise ValueError(
                "--resource-pool-id, --floor-group-id and --day-slugs containing "
                "this slug are required with --live"
            )
        if live and competition_mode == "formal" and len(set(day_slugs)) != 3:
            raise ValueError("formal live mode requires exactly three --day-slugs")
        self.config = config
        self.slug = slug
        self.assets_dir = Path(assets_dir).expanduser().resolve()
        self.duration_minutes = duration_minutes
        # Wall-clock end of the *competition*, which is not the end of this
        # run's window. 0 means "unknown, fall back to the window".
        self.deadline_epoch = float(deadline_epoch or 0.0)
        self.competition_mode = competition_mode
        self.kaggle_user = kaggle_user
        self.live = live
        self.resource_pool_id = resource_pool_id
        self.floor_group_id = floor_group_id
        self.day_slugs = tuple(sorted(set(day_slugs)))
        stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.session = config.run.workspace_root / f"{_safe_id(slug)}-{_safe_id(stamp)}"
        self.search_root = self.session / "search"
        self.shared = self.session / "shared"
        self.eval_root = self.shared / "evaluation"
        self.control = self.session / "control"
        self._stop_event: asyncio.Event | None = None
        self.lanes = self.session / "lanes"
        self.hearsay_ws = self.lanes / "hearsay"
        self.fallback_evaluator_ws = self.lanes / "evaluator_fallback"
        self.events = self.control / "controller_events.jsonl"
        self.status_path = self.session / "RUN_STATUS.json"
        self._fallback_snapshot: AssetSnapshot | None = None
        self._hearsay_process: asyncio.subprocess.Process | None = None
        self._selection_task: asyncio.Task | None = None
        self._stable: dict[str, tuple[str, int]] = {}
        self._t_start: float = time.monotonic()
        self._registered_subs: set[str] = set()
        self._candidate_errors_seen: set[str] = set()
        self._contract_pending_seen: set[str] = set()
        self._agent_sandbox: AgentSandbox | None = None
        self._fallback_evaluator_task: asyncio.Task | None = None
        # The organisers' text, passed through verbatim. It is the task as the
        # competition states it; our own prompt is the system layer around it,
        # which the rules allow ("provide a system prompt on how the agent
        # should submit via Kaggle CLI"). Substituting our paraphrase for the
        # official wording would also drop the Report instruction it carries.
        self.starter_prompt = starter_prompt.strip()
        self.continuation_prompt_text = continuation_prompt_text.strip()
        self.gateway_root = self.session / "kaggle_gateway"
        self._gateway: KaggleReadOnlyGateway | None = None
        self._tool_bin_dir: Path | None = None
        self._anthropic_shim: AnthropicShim | None = None
        self._anthropic_base_url = ""
        self._hearsay_started_monotonic: float | None = None

    def _event(self, event: str, **payload: Any) -> None:
        append_jsonl(self.events, {"timestamp": time.time(), "event": event, **payload})

    def _claude_runner(
        self, profile_dir: Path, *, model: str | None = None,
        effort: str | None = None, allowed_tools: str | None = None,
        sandbox: AgentSandbox | None = None,
    ) -> ClaudeSubscriptionRunner:
        c = self.config.claude
        return ClaudeSubscriptionRunner(
            binary=c.binary, model=model or c.model, effort=effort or c.effort,
            profile_dir=profile_dir,
            sandbox=sandbox or self._agent_sandbox,
            tool_bin_dir=self._tool_bin_dir,
            anthropic_base_url=self._anthropic_base_url,
            api_key_env=c.api_key_env,
            **({"allowed_tools": allowed_tools} if allowed_tools is not None else {}),
        )

    def _codex_runner(
        self, *, sandbox: AgentSandbox | None = None
    ) -> OpenRouterCodexRunner:
        c = self.config.codex
        return OpenRouterCodexRunner(
            binary=c.binary, model=c.model, effort=c.effort,
            provider_id=c.provider_id, base_url=c.base_url,
            api_key_env=c.api_key_env,
            supports_web_search=c.supports_web_search,
            sandbox=sandbox or self._agent_sandbox,
            tool_bin_dir=self._tool_bin_dir,
        )

    def _selection_runner(self) -> OpenRouterCodexRunner:
        """Run the advisory manager through the isolated Codex backend."""
        # Bubblewrap only binds writable roots that exist when the sandbox is
        # constructed. SelectionManager creates this directory later, so make
        # it visible before building the Codex sandbox.
        agent_work = self.control / "selection_manager" / "agent_work"
        agent_work.mkdir(parents=True, exist_ok=True)
        return self._codex_runner(
            sandbox=self._sandbox_for(agent_work)
        )

    def _secondary_codex_runner(
        self, *, sandbox: AgentSandbox | None = None
    ) -> OpenRouterCodexRunner:
        """Create the independent Codex process kept under the legacy lane ID."""
        return self._codex_runner(sandbox=sandbox)

    def _sandbox_for(
        self, writable_root: Path, *, read_only_roots: tuple[Path, ...] = (),
        extra_writable: tuple[Path, ...] = (),
    ) -> AgentSandbox | None:
        if self._agent_sandbox is None:
            return None
        return build_agent_sandbox(
            probe_secret=self.control / "security" / "sandbox_probe_secret",
            protected_write_roots=(
                ROOT, self.session, self.config.resources.root, self.assets_dir,
            ),
            # The gateway dir holds the read-only Kaggle shim and its socket;
            # connecting to a unix socket needs write access to the node.
            writable_roots=(writable_root, self.gateway_root, *extra_writable),
            read_only_roots=read_only_roots,
            protected_read_roots=(self.config.run.workspace_root,),
            readable_roots=(writable_root, self.gateway_root, *read_only_roots),
        )

    def _lane_prompt(
        self, *, lane: str, workdir: Path, deadline_minutes: float,
        submission_mode: str,
    ) -> str:
        """Our operating contract, then the organisers' task text verbatim.

        The rules let us supply a system prompt covering how the agent submits;
        they do not let us restate the problem. So ours goes first and is
        clearly framed as the harness, and the official Starter prompt follows
        unmodified — it is what actually defines the task, and it carries the
        Report instruction that submissions are graded on.
        """
        harness = direct_prompt(
            lane=lane, slug=self.slug, workdir=workdir,
            deadline_minutes=deadline_minutes, submission_mode=submission_mode,
        )
        if not self.starter_prompt:
            return harness
        return (
            f"{harness}\n\n---\n\n"
            "# 官方题面（主办方原文，逐字转发，未做任何改写）\n\n"
            "以下是本题的权威描述。它与上面的系统约定冲突时，**题目要求以下面为准**，\n"
            "只有候选交付协议和提交额度纪律仍由上面的系统层规定。\n\n"
            f"{self.starter_prompt}\n"
        )

    def _lane_continuation(self, lane: str) -> str:
        return self.continuation_prompt_text or continuation_prompt(lane)

    def _prepare(self) -> None:
        if self.session.exists():
            raise FileExistsError(f"session already exists: {self.session}")
        for path in (self.search_root, self.shared, self.control, self.lanes):
            path.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.status_path,
            {
                "schema_version": 1, "status": "running", "slug": self.slug,
                "session": str(self.session), "started_at": time.time(),
                "duration_minutes": self.duration_minutes,
                "competition_mode": self.competition_mode,
                "live_submission": self.live,
            },
        )

    async def _run_search(self) -> Path:
        c = self.config.search
        sandbox = self._sandbox_for(
            self.search_root, read_only_roots=(self.assets_dir,)
        )
        # Keyed by role, not only by backend: the analyst frames the task and
        # the researchers dig, and a runner instance carries one model, so the
        # two jobs would otherwise be forced onto the same one. Backend keys
        # stay as the fallback for roles with no per-role override.
        def claude_for(model: str, effort: str) -> Any:
            return self._claude_runner(
                self.config.claude.integrated_profile_dir,
                model=model or None, effort=effort or None, sandbox=sandbox,
            )

        runners: dict[str, Any] = {
            "codex": self._codex_runner(sandbox=sandbox),
        }
        if "claude" in {c.analyst_backend, *c.research_backends}:
            runners["claude"] = claude_for("", "")
        if c.analyst_backend == "claude" and (c.analyst_model or c.analyst_effort):
            runners["analyst"] = claude_for(c.analyst_model, c.analyst_effort)
        if c.research_model or c.research_effort:
            for index, backend in enumerate(c.research_backends, start=1):
                if backend == "claude":
                    runners[f"R{index}"] = claude_for(
                        c.research_model, c.research_effort
                    )
        # 短排练不能让 Search 吃完整个窗口；正式 6h 仍使用配置的 60min。
        duration = max(
            1,
            int((c.duration_minutes if self.unlimited
                 else min(c.duration_minutes, self.duration_minutes * 0.25)) * 60),
        )
        search_config = SearchRunConfig(
            assets_dir=self.assets_dir,
            output_root=self.search_root,
            prompt_spec=c.prompt_spec,
            duration_s=duration,
            competition_mode=self.competition_mode,
            analyst_backend=c.analyst_backend,
            research_backends=c.research_backends,
        )
        try:
            session = await SearchOrchestrator(search_config, runners=runners).execute()
            self._event("search_finished", session=str(session))
            return session
        finally:
            await asyncio.gather(
                *(runner.terminate_all() for runner in runners.values()),
                return_exceptions=True,
            )

    def _search_asset_snapshot(self) -> Path | None:
        for path in sorted(
            self.search_root.glob("search-*/work/analyst/SEARCH_BUNDLE/ORIGINAL_ASSETS")
        ):
            sums = path.parent / "SHA256SUMS"
            if path.is_dir() and sums.is_file():
                try:
                    _verify_sha256sums(path, sums)
                    return path
                except Exception:  # noqa: BLE001
                    continue
        return None

    async def _wait_for_assets(self, search_task: asyncio.Task | None) -> Path:
        # Assets are infrastructure, not Search output. This used to prefer
        # Search's own snapshot, which lives inside its session dir and dies
        # with it: when Search failed, every lane had been pointed at a path
        # that no longer existed and HearSay crashed at boot on --input-dir
        # (t2run5/t2run6). The controller now always takes its own snapshot;
        # Search's copy stays internal to Search, and its bundle is attached
        # separately only after verification.
        del search_task  # decoupled on purpose; parameter kept for callers
        fallback_root = self.shared / "OFFICIAL_ASSETS"
        if fallback_root.is_dir():
            # Same run-id restart: snapshot_assets refuses to overwrite and
            # re-copying the tree buys nothing — reuse what is already there.
            self._event("official_assets_ready", path=str(fallback_root),
                        reused=True)
            return fallback_root
        snapshot = snapshot_assets(
            self.assets_dir,
            fallback_root,
            sha256sums_path=self.shared / "SHA256SUMS",
            manifest_path=self.shared / "ASSET_MANIFEST.json",
        )
        self._fallback_snapshot = snapshot
        self._event("official_assets_ready", path=str(fallback_root))
        return fallback_root

    def _prepare_direct_lane(self, lane: str, assets: Path) -> Path:
        workdir = self.lanes / lane
        (workdir / "outbox").mkdir(parents=True, exist_ok=True)
        _link(assets, workdir / "OFFICIAL_ASSETS")
        self.eval_root.mkdir(parents=True, exist_ok=True)
        _link(self.eval_root, workdir / "SHARED_EVALUATION")
        feedback = self.control / "feedback" / f"{lane}.jsonl"
        feedback.parent.mkdir(parents=True, exist_ok=True)
        feedback.touch(exist_ok=True)
        _link(feedback, workdir / "FEEDBACK.jsonl")
        return workdir

    def _find_bundle(self, search_session: Path | None) -> Path | None:
        # TEMPORARY: IOAI_REUSE_SEARCH adopts an earlier Search session instead
        # of repeating ~25 minutes and ~$10 while the bundle is constant and
        # only downstream plumbing changes. SHA256SUMS is still verified.
        import os as _os
        if search_session is None:
            _reuse = _os.environ.get("IOAI_REUSE_SEARCH", "").strip()
            if _reuse:
                search_session = Path(_reuse)
                self._event("search_reused", session=str(search_session))
        if search_session:
            status = read_json(search_session / "RUN_STATUS.json", {})
            candidate = Path(status.get("bundle_dir", "")) if status else None
            if candidate and candidate.is_dir() and all(
                (candidate / name).is_file() for name in _REQUIRED_SEARCH
            ):
                _verify_sha256sums(candidate / "ORIGINAL_ASSETS", candidate / "SHA256SUMS")
                return candidate
        return None

    def _candidate_feedback(self, lane: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.control / "feedback" / f"{lane}.jsonl",
            {
                "schema_version": 1,
                "event": "candidate_registration",
                "timestamp": time.time(),
                **payload,
            },
        )

    def _report_candidate_record(self, record: dict) -> None:
        evaluation = record.get("evaluation", {})
        warnings = list(record.get("kernel_format", {}).get("warnings") or [])
        if warnings:
            # Non-blocking, but the lane has to hear it: a kernel with no Report
            # block still scores and still loses the Report grade, and only the
            # lane can fix that in the next candidate.
            self._candidate_feedback(
                record["source_lane"],
                {
                    "candidate_id": record.get("candidate_id"),
                    "status": record.get("status"),
                    "warnings": warnings,
                    "action": "keep this candidate; add the Report block at the "
                              "top of the kernel script in the next version",
                },
            )
        if record.get("status") == "eligible" and evaluation.get("status") in {
            "ok", "pending_contract",
        }:
            return
        self._candidate_feedback(
            record["source_lane"],
            {
                "candidate_id": record.get("candidate_id"),
                "status": record.get("status"),
                "format": record.get("format"),
                "kernel_format": record.get("kernel_format"),
                "evaluation": evaluation,
                "action": "publish a corrected immutable candidate with a new id",
            },
        )

    async def _start_hearsay(
        self, *, bundle: Path | None, assets: Path,
        exploration_deadline: float, submission_mode: str,
    ) -> int:
        remaining_minutes = (
            LANE_ROLLING_MINUTES if getattr(self, "unlimited", False)
            else max(1.0, (exploration_deadline - time.monotonic()) / 60)
        )
        h = self.config.hearsay
        # memory_write targets <repo>/memory/lessons and the repo is bound
        # read-only, so every cross-run lesson an agent tried to keep was lost
        # to "记忆写入因文件系统只读而失败" (dryrun8 evaluator) and each run
        # started from zero. Grant that one subtree; the repo stays immutable.
        lessons = ROOT / "memory" / "lessons"
        lessons.mkdir(parents=True, exist_ok=True)
        # Must exist before the sandbox is built below: the builder binds
        # writable roots by path, and hearsay_ws is created lazily otherwise.
        self.hearsay_ws.mkdir(parents=True, exist_ok=True)
        # _wait_for_assets can return a Search snapshot captured during Search's
        # asset-copy phase; if Search then fails at stage A, that snapshot dir is
        # gone by the time HearSay starts and native.main crashes at boot with
        # "--input-dir is not a directory". This killed the whole HearSay lane —
        # and with it the shared evaluator/ruler — on task 2. Fall back to the
        # raw official assets, which always exist; a missing search bundle
        # likewise falls back to raw assets rather than a dead path.
        assets = Path(assets)
        if not assets.is_dir():
            self._event("hearsay_assets_fallback",
                        missing=str(assets), used=str(self.assets_dir))
            assets = self.assets_dir
        if bundle is not None and not Path(bundle).is_dir():
            self._event("hearsay_bundle_missing", missing=str(bundle))
            bundle = None
        command = [
            sys.executable, "-m", "native.main", "--slug", self.slug,
            "--workspace-dir", str(self.hearsay_ws),
            "--input-dir", str(assets),
            "--external-broker-dir", str(self.control),
            "--submission-mode-hint", submission_mode,
            "--mode", "competition", "--model", h.model,
            "--effort", h.effort, "--solvers", str(h.solvers),
            "--deadline-min", str(remaining_minutes),
            "--rounds", str(h.rounds), "--max-turns", str(h.max_turns),
        ]
        if h.solver_models:
            command += ["--solver-models", h.solver_models]
        if h.solver_efforts:
            command += ["--solver-efforts", h.solver_efforts]
        if h.evaluator_model:
            command += ["--evaluator-model", h.evaluator_model]
        if h.evaluator_effort:
            command += ["--evaluator-effort", h.evaluator_effort]
        if bundle:
            command.extend(["--search-bundle", str(bundle)])
        sandbox = self._sandbox_for(
            self.hearsay_ws,
            extra_writable=(lessons,),
            read_only_roots=tuple(
                path for path in (
                    assets,
                    bundle,
                    self.control / "BROKER_STATUS.json",
                    self.control / "feedback" / "hearsay.jsonl",
                ) if path is not None
            ),
        )
        if sandbox is not None:
            command = sandbox.wrap(command)
        env = {
            key: value for key, value in os.environ.items()
            if key in {
                "PATH", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM",
                "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY",
                "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "CUDA_VISIBLE_DEVICES",
            }
        }
        home = self.hearsay_ws / ".agent_home"; home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        # One auth source, never both — the same three-way choice runners.py
        # makes for the direct lanes. Two stacked blocks used to live here: the
        # first paired the gateway key with the shim URL, the second then
        # overwrote the key from ANTHROPIC_API_KEY. With both keys exported that
        # shipped an sk-ant-... key to Azure, which answers 401 "invalid
        # subscription key" — 719 of them in t2run7, and HearSay's three solvers
        # never completed a single call.
        api_key = os.environ.get(self.config.claude.api_key_env, "").strip()
        if api_key and self._anthropic_base_url:
            env["ANTHROPIC_API_KEY"] = api_key
            env["ANTHROPIC_BASE_URL"] = self._anthropic_base_url
        elif api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        else:
            env["CLAUDE_CONFIG_DIR"] = str(
                self.config.claude.integrated_profile_dir
            )
        if self._tool_bin_dir is not None:
            env["PATH"] = f"{self._tool_bin_dir}:{env.get('PATH', '')}".rstrip(":")
        env["PYTHONUNBUFFERED"] = "1"
        # runners.py pins these for the direct lanes; HearSay never got them, so
        # every subagent and background classifier a solver spawned used Claude
        # Code's own default (claude-opus-4-8), which is not deployed on Azure —
        # 10 x 404 DeploymentNotFound in task 4's first eight minutes, and the
        # evaluator reporting "issue with the selected model". Earlier tasks
        # simply never spawned a subagent, so the gap stayed invisible.
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION"] = "0"
        env["CLAUDE_CODE_BG_CLASSIFIER_MODEL"] = h.model
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = h.model
        # Claude Code resolves internal aliases (compaction, page analysis,
        # the "opus"/"sonnet"/"haiku" tiers) through these defaults, not through
        # --model. Left unset it picked claude-opus-4-8, which is not deployed on
        # Azure, and the evaluator died on a 404 right after a WebFetch — twice,
        # in two consecutive task-4 runs. Only opus-5 and fable-5 exist here.
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = "claude-opus-5"
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = "claude-opus-5"
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = "claude-fable-5"
        env["ANTHROPIC_SMALL_FAST_MODEL"] = "claude-fable-5"
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = os.environ.get(
            "IOAI_CLAUDE_MAX_OUTPUT_TOKENS", "32000")
        env["PYTHONUSERBASE"] = os.environ.get(
            "IOAI_USERBASE", "/data/qyan/ws/pyuserbase")
        env["PIP_CACHE_DIR"] = os.environ.get(
            "IOAI_PIP_CACHE", "/data/qyan/ws/pipcache")
        log = self.control / "hearsay.stdout.log"
        err = self.control / "hearsay.stderr.log"
        with log.open("wb") as stdout, err.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr,
                start_new_session=True,
            )
            self._hearsay_process = process
            self._hearsay_started_monotonic = time.monotonic()
            self._event("hearsay_started", pid=process.pid, bundle=str(bundle or ""))
            try:
                return await process.wait()
            finally:
                self._hearsay_process = None

    async def _stop_hearsay(self) -> None:
        process = self._hearsay_process
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)

    async def _run_fallback_evaluator(
        self, *, started: float, exploration_deadline: float, assets: Path
    ) -> None:
        delay_minutes = min(
            self.config.evaluation.fallback_after_minutes,
            max(1.0, self.duration_minutes * 0.25),
        )
        await asyncio.sleep(max(0.0, started + delay_minutes * 60 - time.monotonic()))
        if (self.eval_root / "contract.json").is_file() \
                or time.monotonic() >= exploration_deadline:
            return
        self.fallback_evaluator_ws.mkdir(parents=True, exist_ok=True)
        _link(assets, self.fallback_evaluator_ws / "OFFICIAL_ASSETS")
        prompt = f"""你是独立的 IOAI 公共验证契约 fallback evaluator。主 evaluator
在规定时间内没有产出可冻结标尺。只读取 `{self.fallback_evaluator_ws / 'OFFICIAL_ASSETS'}`
中的原始比赛资产，独立确定官方 metric 和无泄漏验证划分；不要建模或生成 submission。

在当前目录生成且只生成以下四个核心 artifact：
1. metric.py，暴露 score(y_true, y_pred)。
2. folds.json，包含 ids/fold/labels/scheme，并覆盖真实验证单元。
3. metric_tests.json，格式为 {{"cases":[{{"y_true":[...],"y_pred":[...],
   "expected":0.0,"tolerance":1e-8}}]}}，expected 必须手算。
4. contract_evidence.json，六个非空字段：metric_name、direction、metric_source、
   id_source、label_source、split_rationale；direction 约束是
   `{self.config.run.metric_direction}`。若这里是 `auto`，必须根据官方证据填写
   `maximize` 或 `minimize`；source 必须是实际资产路径。

完成前自己运行 metric tests，并用
`{sys.executable} -m native.scripts.checkfolds --file folds.json` 检查划分。不要访问 Kaggle，
不要读取其他路线，也不要修改 OFFICIAL_ASSETS。"""
        runner = self._codex_runner(
            sandbox=self._sandbox_for(
                self.fallback_evaluator_ws, read_only_roots=(assets,)
            ),
        )
        timeout = min(
            self.config.evaluation.fallback_turn_minutes * 60,
            max(1.0, exploration_deadline - time.monotonic()),
        )
        self._event("fallback_evaluator_started", timeout_seconds=timeout)
        try:
            result = await runner.run(
                agent_id="fallback-evaluator",
                role="evaluation_fallback",
                prompt=prompt,
                workdir=self.fallback_evaluator_ws,
                trace_dir=self.control / "trajectories" / "evaluator_fallback",
                timeout_s=timeout,
            )
            self._event(
                "fallback_evaluator_finished",
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                errors=result.errors,
            )
        finally:
            await runner.terminate_all()

    def _register_direct(self, registry: CandidateRegistry, lane: str) -> int:
        count = 0
        outbox = self.lanes / lane / "outbox"
        for candidate in sorted(outbox.iterdir() if outbox.exists() else []):
            if not candidate.is_dir() or not (candidate / "READY").is_file():
                continue
            try:
                before = len(registry.records())
                record = registry.register(candidate, source_lane=lane)
                is_new = len(registry.records()) > before
                count += is_new
                if is_new:
                    self._report_candidate_record(record)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                key = f"{candidate}:{error}"
                if key not in self._candidate_errors_seen:
                    self._candidate_errors_seen.add(key)
                    self._event(
                        "candidate_rejected", lane=lane, path=str(candidate),
                        error=error,
                    )
                    self._candidate_feedback(
                        lane,
                        {
                            "candidate_path": str(candidate),
                            "status": "rejected",
                            "errors": [error],
                            "action": "fix the artifact and publish a new candidate id",
                        },
                    )
        return count

    @staticmethod
    def _hearsay_resources(solver: Path, submission_mode: str) -> dict[str, Any]:
        if submission_mode == "csv":
            return {"accelerator": "cpu"}
        declaration: dict[str, Any] = {}
        declaration_path = solver / "candidate.json"
        if declaration_path.is_file():
            value = json.loads(declaration_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("HearSay candidate.json must be an object")
            for key in ("accelerator", "estimated_kernel_minutes"):
                if key in value:
                    declaration[key] = value[key]

        if "accelerator" not in declaration:
            metadata_path = solver / "out" / "kernel" / "kernel-metadata.json"
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file() else {}
            )
            if not isinstance(metadata, dict):
                raise ValueError("HearSay kernel-metadata.json must be an object")
            enabled = metadata.get("enable_gpu") is True or str(
                metadata.get("enable_gpu", "")
            ).lower() == "true"
            shape = str(metadata.get("machine_shape", ""))
            declaration["accelerator"] = (
                "t4" if enabled and shape == "NvidiaTeslaT4"
                else "p100" if enabled else "cpu"
            )
        return declaration

    async def _await_contract(
        self, *, hard_deadline: float,
        hearsay_task: "asyncio.Task | None" = None,
    ) -> None:
        """Block until the shared contract exists, or give up and let lanes run.

        The 8-minute window is anchored to hearsay_started, not run start: the
        ruler is compiled by HearSay's evaluator, and when Search took 9.4 min
        (dryrun2) a start-anchored window closed before the compiler was even
        born. hard_deadline still caps the total wait so a stuck Search cannot
        starve the direct lanes. Giving up is deliberate: if HearSay is
        disabled or fails, the lanes run and guess, as before, by exception.
        """
        target = self.eval_root / "contract.json"
        while time.monotonic() < hard_deadline:
            if target.is_file():
                self._event("contract_ready_before_lanes",
                            waited_s=round(time.monotonic() - self._t_start, 1))
                return
            # The freeze relay normally lives in the monitor loop — which
            # starts after this wait. During the wait the evaluator's published
            # artifacts therefore had no path into eval_root, and the window
            # could never be satisfied no matter how the deadline was anchored
            # (dryrun3: ruler published at +17.5m, eval_root still empty).
            # Attempt the same freeze here; it fails fast while artifacts are
            # incomplete and the monitor's own attempt later is a no-op.
            for source in (self.hearsay_ws, self.fallback_evaluator_ws):
                if source.exists():
                    try:
                        freeze_contract(
                            source, self.eval_root,
                            direction=self.config.run.metric_direction,
                        )
                    except (FileNotFoundError, ValueError):
                        pass
            started_at = self._hearsay_started_monotonic
            if started_at is not None:
                if time.monotonic() >= started_at + 8 * 60:
                    break
            elif hearsay_task is not None and hearsay_task.done():
                # HearSay will never start (disabled, or bailed out): no ruler
                # is coming, so waiting any longer only burns the window.
                break
            await asyncio.sleep(2)
        self._event("contract_wait_timeout",
                    waited_s=round(time.monotonic() - self._t_start, 1))

    def _register_hearsay(
        self, registry: CandidateRegistry, submission_mode: str
    ) -> int:
        count = 0
        for solver in sorted(self.hearsay_ws.glob("solver_*")):
            if not (solver / "out" / "submission.csv").is_file():
                continue
            # A solver writes submission.csv first and builds out/kernel/ after.
            # Registration used to fire in that window — measured at 107s — and
            # the candidate was eligible on the strength of its CSV alone.
            if submission_mode == "kernel" and not (
                solver / "out" / "kernel" / "kernel-metadata.json"
            ).is_file():
                continue
            try:
                fingerprint = candidate_fingerprint(solver)
            except Exception:
                continue
            # Nine of solver_c's snapshots in one run were the same
            # submission re-registered every time an unrelated file (a log, a
            # cache, a rewritten oof) changed the directory fingerprint. The
            # registry did mark them duplicate — after copying a full snapshot
            # each time. Check the one hash that defines a duplicate first.
            try:
                sub_sha = hashlib.sha256(
                    (solver / "out" / "submission.csv").read_bytes()
                ).hexdigest()
            except OSError:
                continue
            if sub_sha in self._registered_subs:
                continue
            previous, stable_count = self._stable.get(str(solver), ("", 0))
            stable_count = stable_count + 1 if previous == fingerprint else 1
            self._stable[str(solver)] = (fingerprint, stable_count)
            if stable_count < 2:
                continue
            try:
                resources = self._hearsay_resources(solver, submission_mode)
                manifest = {
                    "schema_version": 1,
                    "candidate_id": f"{solver.name}-{fingerprint[:10]}",
                    "source_lane": "hearsay",
                    "submission_mode": submission_mode,
                    **resources,
                    "purpose": f"stable HearSay snapshot from {solver.name}",
                }
                before = len(registry.records())
                self._registered_subs.add(sub_sha)
                record = registry.register(
                    solver, source_lane="hearsay", manifest=manifest
                )
                is_new = len(registry.records()) > before
                count += is_new
                if is_new:
                    self._report_candidate_record(record)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                key = f"{solver}:{error}"
                if key not in self._candidate_errors_seen:
                    self._candidate_errors_seen.add(key)
                    self._event(
                        "candidate_rejected", lane="hearsay", path=str(solver),
                        error=error,
                    )
                    self._candidate_feedback(
                        "hearsay",
                        {
                            "candidate_path": str(solver),
                            "status": "rejected",
                            "errors": [error],
                            "action": "fix the artifact and publish a stable new version",
                        },
                    )
        return count

    async def _monitor(
        self, *, registry: CandidateRegistry, broker: SubmissionBroker,
        calibrator: FeedbackCalibrator,
        selection_manager: SelectionManager | None,
        started: float, deadline: float, exploration_deadline: float,
        submission_deadline: float, submission_mode: str,
        final_gate_wall: float,
    ) -> None:
        last_scores = 0.0
        last_manager_started = 0.0
        last_manager_context = ""
        last_calibration_version = ""
        contract_frozen = False
        active_submissions: set[asyncio.Task] = set()
        active_manager: asyncio.Task | None = None
        while time.monotonic() < deadline:
            if active_manager is not None and active_manager.done():
                try:
                    recommendation = active_manager.result()
                    self._event(
                        "selection_manager_finished",
                        status="complete" if recommendation else "failed",
                        recommendation=recommendation or {},
                    )
                except Exception as exc:  # noqa: BLE001
                    self._event(
                        "selection_manager_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                active_manager = None
                self._selection_task = None
            completed = {task for task in active_submissions if task.done()}
            for task in completed:
                try:
                    submission = task.result()
                    if submission:
                        self._event("broker_submission", submission=submission)
                except Exception as exc:  # noqa: BLE001
                    self._event(
                        "broker_background_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            active_submissions.difference_update(completed)
            if not contract_frozen:
                for source_name, source in (
                    ("hearsay", self.hearsay_ws),
                    ("fallback", self.fallback_evaluator_ws),
                ):
                    if not source.exists():
                        continue
                    try:
                        contract = freeze_contract(
                            source, self.eval_root,
                            direction=self.config.run.metric_direction,
                        )
                        contract_frozen = True
                        broker.set_direction(contract["direction"])
                        self._event(
                            "evaluation_contract_frozen",
                            evaluator_source=source_name,
                            **contract,
                        )
                        break
                    except (FileNotFoundError, ValueError) as exc:
                        detail = str(exc)[:500]
                        key = f"{source_name}:{detail}"
                        if key not in self._contract_pending_seen:
                            self._contract_pending_seen.add(key)
                            self._event(
                                "evaluation_contract_pending",
                                evaluator_source=source_name,
                                detail=detail,
                            )
            added = self._register_direct(registry, "codex")
            added += self._register_direct(registry, "claude")
            if self.hearsay_ws.exists():
                added += self._register_hearsay(registry, submission_mode)
            if contract_frozen:
                registry.refresh_evaluations()
            elapsed = time.monotonic() - started
            fraction = min(1.0, elapsed / max(deadline - started, 1))
            if time.monotonic() - last_scores >= 60:
                updates = await asyncio.to_thread(broker.refresh_scores)
                if updates:
                    self._event("leaderboard_feedback", updates=updates)
                last_scores = time.monotonic()

            calibration = calibrator.update(
                registry.records(), broker.submissions()
            )
            version = str(calibration.get("version", ""))
            if version and version != last_calibration_version:
                summary = {
                    "schema_version": 1,
                    "event": "calibration_update",
                    "timestamp": time.time(),
                    "version": version,
                    "base_contract": calibration.get("base_contract"),
                    "active": calibration.get("active", False),
                    "reason": calibration.get("reason"),
                    "feedback_count": calibration.get("feedback_count", 0),
                    "baseline_spearman": calibration.get("baseline_spearman"),
                    "calibrated_loo_spearman": calibration.get(
                        "calibrated_loo_spearman"
                    ),
                    "residual_rmse": calibration.get("residual_rmse"),
                    "blend_weight": calibration.get("blend_weight", 0.0),
                }
                for lane in ("global", "hearsay", "codex", "claude"):
                    append_jsonl(
                        self.control / "feedback" / f"{lane}.jsonl", summary
                    )
                self._event("calibration_updated", calibration=summary)
                last_calibration_version = version

            if selection_manager is not None:
                context = broker.manager_context(
                    fraction_elapsed=fraction,
                    force_final=time.time() >= final_gate_wall,
                )
                if context is not None and active_manager is None:
                    now = time.monotonic()
                    digest = str(context["context_sha256"])
                    latest = read_json(selection_manager.latest_path, {})
                    current = (
                        latest.get("context_sha256") == digest
                        and float(latest.get("expires_at", 0)) > time.time()
                    )
                    interval_passed = (
                        now - last_manager_started
                        >= self.config.selection.interval_seconds
                    )
                    if not current and (
                        not last_manager_context or interval_passed
                    ):
                        active_manager = asyncio.create_task(
                            selection_manager.recommend(context)
                        )
                        self._selection_task = active_manager
                        last_manager_started = now
                        last_manager_context = digest
                        self._event(
                            "selection_manager_started",
                            context_sha256=digest,
                            candidate_count=len(context.get("candidates", [])),
                            submission_class=context.get("submission_class"),
                        )
            if time.monotonic() < submission_deadline:
                while (
                    len(active_submissions)
                    < self.config.run.max_inflight_submissions
                ):
                    active_submissions.add(asyncio.create_task(asyncio.to_thread(
                        broker.tick, fraction_elapsed=fraction,
                        force_final=time.time() >= final_gate_wall,
                    )))
            else:
                # 最后缓冲只轮询和恢复，不再启动一个可能越过截止线的新 kernel。
                broker.write_status(fraction_elapsed=fraction)
            if added:
                self._event("candidates_registered", count=added)
            await asyncio.sleep(self.config.run.poll_seconds)
        # submission_deadline 已为这些调用预留了完成时间；不得把线程丢在后台。
        for submission in await asyncio.gather(
            *active_submissions, return_exceptions=True
        ):
            if isinstance(submission, dict):
                self._event("broker_submission", submission=submission)
            elif isinstance(submission, BaseException):
                self._event(
                    "broker_background_error",
                    error=f"{type(submission).__name__}: {submission}",
                )
        if active_manager is not None:
            active_manager.cancel()
            await asyncio.gather(active_manager, return_exceptions=True)
            self._selection_task = None

    def _request_stop(self, reason: str) -> None:
        if self._stop_event is not None and not self._stop_event.is_set():
            self._event("stop_requested", reason=reason)
            self._stop_event.set()

    async def _watch_stop_file(self) -> None:
        """`touch control/STOP` is the supported way to end an unlimited run."""
        marker = self.control / "STOP"
        while self._stop_event is not None and not self._stop_event.is_set():
            if marker.exists():
                self._request_stop("STOP file")
                return
            await asyncio.sleep(2)

    async def _sleep_until(self, deadline_monotonic: float) -> str:
        """Wait for a deadline, or for a stop request — whichever lands first."""
        remaining = deadline_monotonic - time.monotonic()
        if self._stop_event is None:
            await asyncio.sleep(max(0.0, remaining))
            return "deadline"
        timeout = None if remaining == float("inf") else max(0.0, remaining)
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return "deadline"
        return "stopped"

    async def run(self) -> Path:
        self._prepare()
        if self.config.claude.anthropic_base_url:
            self._anthropic_shim = AnthropicShim(
                self.config.claude.anthropic_base_url,
                on_event=lambda **payload: self._event(
                    "anthropic_shim_upstream_error", **payload
                ),
            )
            self._anthropic_base_url = self._anthropic_shim.start()
            self._event(
                "anthropic_shim_started",
                upstream=self.config.claude.anthropic_base_url,
                base_url=self._anthropic_base_url,
            )
        # IOAI requires the system to fetch its own data. The credential stays
        # outside the sandbox; agents get a shim that can only read.
        self._gateway = KaggleReadOnlyGateway(
            self.gateway_root, kaggle_bin=_kaggle_bin(), python=sys.executable,
            events_path=self.control / "kaggle_gateway_events.jsonl",
        )
        await self._gateway.start()
        self._tool_bin_dir = self._gateway.bin_dir
        self._event(
            "kaggle_gateway_started",
            socket=str(self._gateway.socket_path),
            allowed=sorted(f"{a} {b}" for a, b in GATEWAY_ALLOWED),
        )
        try:
            probe_secret = self.control / "security" / "sandbox_probe_secret"
            self._agent_sandbox = build_agent_sandbox(
                probe_secret=probe_secret,
                protected_write_roots=(
                    ROOT, self.session, self.config.resources.root, self.assets_dir,
                ),
            )
            verify_agent_sandbox(
                self._agent_sandbox, probe_secret=probe_secret
            )
            self._event(
                "agent_sandbox_verified",
                denied_paths=[str(path) for path in self._agent_sandbox.denied_paths],
                protected_write_roots=[
                    str(path) for path in self._agent_sandbox.protected_write_roots
                ],
            )
        except RuntimeError as exc:
            if self.live:
                raise
            # 非 live 诊断可在不支持 Seatbelt 的系统运行；正式提交始终 fail-closed。
            self._event("agent_sandbox_unavailable", detail=str(exc))
        self._stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(
                    sig, self._request_stop, f"signal {sig.name}"
                )
        stop_watch = asyncio.create_task(self._watch_stop_file())
        started = time.monotonic()
        INF = float("inf")
        if self.unlimited:
            wall_deadline = self.deadline_epoch or INF
            deadline = (
                started + (wall_deadline - time.time())
                if wall_deadline != INF else INF
            )
        else:
            deadline = started + self.duration_minutes * 60
            wall_deadline = time.time() + self.duration_minutes * 60
        # Minutes actually available, for the sizing maths below. In unlimited
        # mode the lanes get a rolling budget instead of a share of a window:
        # they deliver, the loop restarts them with the continuation prompt.
        span_minutes = (
            self.duration_minutes if not self.unlimited
            else (wall_deadline - time.time()) / 60 if wall_deadline != INF
            else LANE_ROLLING_MINUTES
        )
        # Everything that decides "are we out of time" must key off the real
        # competition deadline when we know it. Keying off the run window made
        # the broker spend its final reserve 49 minutes early and then stop.
        hard_deadline = self.deadline_epoch or wall_deadline  # may be inf
        final_gate_wall = INF if hard_deadline == INF else hard_deadline - min(
            self.config.run.submission_only_minutes_before_end,
            max(0.0, (hard_deadline - time.time()) / 60 * 0.20),
        ) * 60
        stop_before = min(
            self.config.run.stop_exploration_minutes_before_end,
            max(0.0, span_minutes * 0.45),
        )
        exploration_deadline = (
            deadline - stop_before * 60 if deadline != INF else INF
        )
        # Monotonic mirror of the real deadline, so the "stop starting new
        # kernels" line is drawn against the competition clock rather than
        # against this run's window.
        if hard_deadline == INF:
            submission_deadline = INF
        else:
            hard_deadline_mono = started + (hard_deadline - time.time())
            submission_deadline = hard_deadline_mono - min(
                self.config.run.submission_only_minutes_before_end,
                max(0.0, (hard_deadline - time.time()) / 60 * 0.20),
            ) * 60
        search_task: asyncio.Task | None = None
        if self.config.search.enabled:
            search_task = asyncio.create_task(self._run_search())
        assets = await self._wait_for_assets(search_task)

        codex_work = self._prepare_direct_lane("codex", assets)
        claude_work = self._prepare_direct_lane("claude", assets)
        resource_gate: DayResourceGate | None = None
        if self.live:
            resources = self.config.resources
            resource_gate = DayResourceGate(
                resources.root / _safe_id(self.resource_pool_id),
                pool_id=self.resource_pool_id,
                floor_group_id=self.floor_group_id,
                expected_slugs=self.day_slugs,
                gpu_limit_hours=resources.gpu_quota_hours,
                gpu_concurrency=resources.gpu_concurrency,
                cpu_concurrency=resources.cpu_concurrency,
                poll_seconds=resources.acquire_poll_seconds,
                acquire_wait_seconds=resources.acquire_wait_seconds,
                floor_grace_seconds=resources.floor_grace_minutes * 60,
            )
            adapter: Any = KaggleAdapter(
                slug=self.slug, root=self.control, kaggle_user=self.kaggle_user,
                submission_mode=self.config.run.submission_mode,
                assets=assets,
                resource_gate=resource_gate,
                competition_deadline_epoch=hard_deadline,
                kernel_version_budget=self.config.run.kernel_version_budget,
                default_gpu_kernel_minutes=resources.default_gpu_kernel_minutes,
                default_cpu_kernel_minutes=resources.default_cpu_kernel_minutes,
                kernel_start_margin_minutes=resources.kernel_start_margin_minutes,
                # 0 means "do not pass --timeout at all": task 3's kernel only
                # writes a base64 CSV in seconds and its rules say the default
                # is plenty, while the 600s budget belongs to the grader, not
                # to Kaggle. Passing a flag the task tells you to omit is a
                # rule violation, not a safety margin.
                kernel_timeout_s=(self.config.run.kernel_timeout_seconds
                                  or None),
            )
            remaining = await asyncio.to_thread(adapter.remaining_today)
            if remaining is None:
                raise RuntimeError(
                    "live mode could not establish Kaggle remaining-today quota; "
                    "refusing to create a second, guessed budget"
                )
            max_submissions = min(self.config.run.max_submissions, remaining)
            submission_mode = await asyncio.to_thread(adapter.detect_mode)
        else:
            adapter = DryRunAdapter(self.slug, self.control)
            max_submissions = self.config.run.max_submissions
            submission_mode = (
                self.config.run.submission_mode
                if self.config.run.submission_mode != "auto" else "kernel"
            )
        if max_submissions <= 0:
            raise RuntimeError("Kaggle reports zero remaining submissions")

        registry = CandidateRegistry(
            self.control / "candidates", assets=assets, evaluation=self.eval_root
        )
        selection = self.config.selection
        calibrator = FeedbackCalibrator(
            self.control / "calibration",
            min_feedback=selection.calibration_min_feedback,
            ridge_strength=selection.calibration_ridge_strength,
            min_rank_gain=selection.calibration_min_rank_gain,
            max_blend_weight=selection.calibration_max_blend_weight,
        )
        selection_runner = self._selection_runner() if selection.enabled else None
        selection_manager = SelectionManager(
            self.control / "selection_manager",
            runner=selection_runner,
            model=selection.model,
            effort=selection.effort,
            timeout_seconds=selection.timeout_seconds,
            recommendation_ttl_seconds=selection.recommendation_ttl_seconds,
        ) if selection_runner is not None else None
        broker = SubmissionBroker(
            self.control, registry=registry, adapter=adapter,
            max_submissions=max_submissions,
            final_reserve=min(self.config.run.final_reserve, max_submissions - 1),
            initial_calibrations=min(
                self.config.run.initial_calibrations, max(0, max_submissions - 1)
            ),
            final_start_fraction=self.config.run.final_start_fraction,
            anti_monopoly_fraction=self.config.run.anti_monopoly_fraction,
            min_local_gain=self.config.run.min_local_gain,
            direction=self.config.run.metric_direction,
            manager_enabled=selection.enabled,
            recommendation_path=(
                selection_manager.latest_path if selection_manager else None
            ),
            calibration_path=calibrator.latest_path,
            manager_fallback_seconds=selection.fallback_seconds,
            max_retryable_attempts=self.config.run.max_retryable_attempts,
            retry_backoff_seconds=self.config.run.retry_backoff_seconds,
            floor_done=(
                (lambda: resource_gate.floor_done(self.slug))
                if self.floor_group_id else None
            ),
            floor_settle_seconds=self.config.run.floor_settle_seconds,
            floor_settle_candidates=self.config.run.floor_settle_candidates,
        )
        # Leaderboard scores land on HearSay's board, not just in its private
        # feedback file — the evaluator calibrates against them and a revived
        # solver can still read them.
        broker.hearsay_board_path = self.hearsay_ws / "facts.jsonl"

        # Late bundle delivery. HearSay used to wait for Search to finish so
        # the run contract could attest the bundle hash at boot — which held
        # the evaluator (and the whole review chain) hostage to Search's 9-12
        # minutes. Now HearSay boots immediately against an empty staging dir
        # (truthfully attested empty); when Search completes, the verified
        # bundle is copied in from outside the sandbox — the read-only bind
        # reflects the files inside — and the delivery is attested here in the
        # controller ledger with its tree hash. Solvers are told at boot that
        # material may arrive mid-run (native prompt).
        bundle_staging = self.shared / "SEARCH_BUNDLE_STAGING"
        bundle_staging.mkdir(parents=True, exist_ok=True)

        async def deliver_bundle_late() -> None:
            if search_task is None:
                return
            try:
                search_session = await search_task
            except Exception as exc:  # noqa: BLE001
                self._event("search_failed", error=f"{type(exc).__name__}: {exc}")
                return
            bundle = self._find_bundle(search_session)
            self._event(
                "search_handoff", status="verified_bundle" if bundle else "raw_assets_fallback",
                bundle=str(bundle or ""), bundle_sha256=tree_hash(bundle) if bundle else "",
            )
            if not bundle:
                return
            try:
                for item in Path(bundle).iterdir():
                    target = bundle_staging / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
                (bundle_staging / "ARRIVAL.md").write_text(
                    "Search bundle delivered mid-run by the controller.\n"
                    "Contents are verified and read-only; see controller_events"
                    " (search_bundle_delivered) for the attested tree hash.\n"
                )
                self._event(
                    "search_bundle_delivered",
                    staging=str(bundle_staging),
                    tree_sha256=tree_hash(bundle_staging),
                )
            except Exception as exc:  # noqa: BLE001
                self._event("search_bundle_delivery_failed",
                            error=f"{type(exc).__name__}: {exc}")

        async def start_hearsay_now() -> int:
            if not self.config.hearsay.enabled or time.monotonic() >= exploration_deadline:
                return 0
            return await self._start_hearsay(
                bundle=bundle_staging, assets=assets,
                exploration_deadline=exploration_deadline,
                submission_mode=submission_mode,
            )

        hearsay_task = asyncio.create_task(start_hearsay_now())
        self._bundle_delivery_task = asyncio.create_task(deliver_bundle_late())

        # The shared ruler makes the three lanes comparable and is compiled by
        # HearSay's evaluator. It used to be awaited BEFORE HearSay was even
        # created, so the 8-minute wait could never be satisfied and timed out
        # on 6 of 7 runs — pure loss. HearSay now starts first (above); this
        # wait gives its evaluator a real window to freeze the contract before
        # the direct lanes boot. The wait itself stays: direct lanes starting
        # against an empty SHARED_EVALUATION/ once burned a 45-minute first
        # round on a guessed unit layout (codex 624 rows vs expected 920).
        await self._await_contract(
            hard_deadline=self._t_start + min(
                20 * 60, self.duration_minutes * 60 * 0.35),
            hearsay_task=hearsay_task,
        )
        codex_runner = self._codex_runner(
            sandbox=self._sandbox_for(
                codex_work,
                read_only_roots=(
                    assets, self.eval_root,
                    self.control / "feedback" / "codex.jsonl",
                ),
            )
        )
        claude_runner = self._claude_runner(
            self.config.claude.direct_profile_dir,
            sandbox=self._sandbox_for(
                claude_work,
                read_only_roots=(
                    assets, self.eval_root,
                    self.control / "feedback" / "claude.jsonl",
                ),
            ),
        )
        direct_tasks = [
            asyncio.create_task(run_lane_loop(
                lane="codex", runner=codex_runner,
                prompt=self._lane_prompt(
                    lane="codex", workdir=codex_work,
                    deadline_minutes=(LANE_ROLLING_MINUTES if self.unlimited
                                      else max(1, (exploration_deadline-started)/60)),
                    submission_mode=submission_mode,
                ),
                continuation=self._lane_continuation("codex"), workdir=codex_work,
                trace_dir=self.control / "trajectories" / "codex",
                deadline_monotonic=exploration_deadline,
                turn_seconds=self.config.codex.turn_minutes * 60,
                max_rounds=(10 ** 9 if self.unlimited
                            else self.config.codex.max_rounds),
            )),
            asyncio.create_task(run_lane_loop(
                lane="claude", runner=claude_runner,
                prompt=self._lane_prompt(
                    lane="claude", workdir=claude_work,
                    deadline_minutes=(LANE_ROLLING_MINUTES if self.unlimited
                                      else max(1, (exploration_deadline-started)/60)),
                    submission_mode=submission_mode,
                ),
                continuation=self._lane_continuation("claude"), workdir=claude_work,
                trace_dir=self.control / "trajectories" / "claude",
                deadline_monotonic=exploration_deadline,
                turn_seconds=self.config.claude.turn_minutes * 60,
                max_rounds=(10 ** 9 if self.unlimited
                            else self.config.claude.max_rounds),
            )),
        ]

        self._fallback_evaluator_task = asyncio.create_task(
            self._run_fallback_evaluator(
                started=started,
                exploration_deadline=exploration_deadline,
                assets=assets,
            )
        )
        monitor_task = asyncio.create_task(self._monitor(
            registry=registry, broker=broker, calibrator=calibrator,
            selection_manager=selection_manager,
            started=started, deadline=deadline,
            exploration_deadline=exploration_deadline,
            submission_deadline=submission_deadline,
            submission_mode=submission_mode,
            final_gate_wall=final_gate_wall,
        ))
        try:
            why = await self._sleep_until(exploration_deadline)
            stop_watch.cancel()
            self._event("exploration_stopped", reason=why)
            await asyncio.gather(
                codex_runner.terminate_all(),
                claude_runner.terminate_all(),
                self._stop_hearsay(), return_exceptions=True,
            )
            extra_tasks = (
                [self._fallback_evaluator_task]
                if self._fallback_evaluator_task is not None else []
            )
            for task in direct_tasks + [hearsay_task, *extra_tasks]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *direct_tasks, hearsay_task, *extra_tasks, return_exceptions=True
            )
            await monitor_task
        finally:
            if not monitor_task.done():
                monitor_task.cancel()
                await asyncio.gather(monitor_task, return_exceptions=True)
            if self._selection_task is not None:
                self._selection_task.cancel()
                await asyncio.gather(
                    self._selection_task, return_exceptions=True
                )
                self._selection_task = None
            if self._fallback_evaluator_task is not None \
                    and not self._fallback_evaluator_task.done():
                self._fallback_evaluator_task.cancel()
                await asyncio.gather(
                    self._fallback_evaluator_task, return_exceptions=True
                )
            if self._gateway is not None:
                await self._gateway.stop()
                self._gateway = None
            if self._anthropic_shim is not None:
                self._anthropic_shim.stop()
                self._anthropic_shim = None
            await asyncio.gather(
                codex_runner.terminate_all(),
                claude_runner.terminate_all(),
                *(
                    [selection_runner.terminate_all()]
                    if selection_runner is not None else []
                ),
                self._stop_hearsay(), return_exceptions=True,
            )

        if self._fallback_snapshot:
            verify_snapshot(self._fallback_snapshot, verify_source=True)
        else:
            sums = assets.parent / "SHA256SUMS"
            if sums.is_file():
                _verify_sha256sums(assets, sums)
        final_status = read_json(self.status_path, {})
        final_status.update({
            "status": "complete", "finished_at": time.time(),
            "evaluation_contract": read_json(self.eval_root / "contract.json"),
            "broker": broker.write_status(fraction_elapsed=1.0),
            "candidate_count": len(registry.records()),
            "calibration": read_json(calibrator.latest_path, {}),
            "selection_manager": read_json(
                selection_manager.status_path, {}
            ) if selection_manager else {"status": "disabled"},
            "shared_resources": (
                resource_gate.status() if resource_gate is not None else None
            ),
        })
        atomic_json(self.status_path, final_status)
        self._event("run_complete", status=str(self.status_path))
        return self.session
