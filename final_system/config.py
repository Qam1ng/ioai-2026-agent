from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _section(value: dict[str, Any], name: str) -> dict[str, Any]:
    section = value.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return section


@dataclass(frozen=True)
class RunConfig:
    workspace_root: Path
    poll_seconds: float
    max_submissions: int
    final_reserve: int
    initial_calibrations: int
    final_start_fraction: float
    anti_monopoly_fraction: float
    min_local_gain: float
    metric_direction: str
    submission_mode: str
    max_inflight_submissions: int
    stop_exploration_minutes_before_end: float
    submission_only_minutes_before_end: float
    max_retryable_attempts: int
    retry_backoff_seconds: float
    floor_settle_seconds: float
    floor_settle_candidates: int
    kernel_timeout_seconds: float


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool
    duration_minutes: float
    analyst_backend: str
    research_backends: tuple[str, str, str, str]
    prompt_spec: Path
    # Empty means "inherit the backend's own model/effort". The analyst frames
    # the task and the researchers dig; those are different jobs and can want
    # different models on the same backend.
    analyst_model: str = ""
    analyst_effort: str = ""
    research_model: str = ""
    research_effort: str = ""


@dataclass(frozen=True)
class ClaudeConfig:
    binary: Path
    model: str
    effort: str
    integrated_profile_dir: Path
    direct_profile_dir: Path
    turn_minutes: float
    max_rounds: int


@dataclass(frozen=True)
class CodexConfig:
    binary: Path
    model: str
    effort: str
    provider_id: str
    base_url: str
    api_key_env: str
    supports_web_search: bool
    turn_minutes: float
    max_rounds: int


@dataclass(frozen=True)
class HearSayConfig:
    enabled: bool
    model: str
    effort: str
    solvers: int
    rounds: int
    max_turns: int
    solver_models: str = ""
    solver_efforts: str = ""
    evaluator_model: str = ""
    evaluator_effort: str = ""


@dataclass(frozen=True)
class SelectionConfig:
    enabled: bool
    model: str
    effort: str
    interval_seconds: float
    timeout_seconds: float
    recommendation_ttl_seconds: float
    fallback_seconds: float
    calibration_min_feedback: int
    calibration_ridge_strength: float
    calibration_min_rank_gain: float
    calibration_max_blend_weight: float


@dataclass(frozen=True)
class ResourceConfig:
    root: Path
    gpu_quota_hours: float
    gpu_concurrency: int
    cpu_concurrency: int
    default_gpu_kernel_minutes: float
    default_cpu_kernel_minutes: float
    kernel_start_margin_minutes: float
    acquire_poll_seconds: float
    acquire_wait_seconds: float
    floor_grace_minutes: float


@dataclass(frozen=True)
class EvaluationConfig:
    fallback_after_minutes: float
    fallback_turn_minutes: float


@dataclass(frozen=True)
class SystemConfig:
    run: RunConfig
    search: SearchConfig
    claude: ClaudeConfig
    codex: CodexConfig
    hearsay: HearSayConfig
    selection: SelectionConfig
    resources: ResourceConfig
    evaluation: EvaluationConfig

    @classmethod
    def load(cls, path: Path, *, repo_root: Path) -> "SystemConfig":
        path = Path(path).expanduser().resolve()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        run = _section(raw, "run")
        search = _section(raw, "search")
        claude = _section(raw, "claude")
        codex = _section(raw, "codex")
        hearsay = _section(raw, "hearsay")
        selection = _section(raw, "selection")
        resources = _section(raw, "resources")
        evaluation = _section(raw, "evaluation")

        workspace = Path(run.get("workspace_root", "workspace/final_system"))
        if not workspace.is_absolute():
            workspace = repo_root / workspace
        resource_root = Path(
            resources.get("root", "workspace/final_system/_resource_pools")
        )
        if not resource_root.is_absolute():
            resource_root = repo_root / resource_root
        prompt = Path(search.get("prompt_spec", "search_system/SEARCH_PROMPTS_ZH.md"))
        if not prompt.is_absolute():
            prompt = repo_root / prompt
        default_profile = claude.get("profile_dir", "~/.claude")
        integrated_profile = Path(
            claude.get("integrated_profile_dir", default_profile)
        ).expanduser()
        direct_profile = Path(
            claude.get("direct_profile_dir", default_profile)
        ).expanduser()

        value = cls(
            run=RunConfig(
                workspace_root=workspace.resolve(),
                poll_seconds=float(run.get("poll_seconds", 10)),
                max_submissions=int(run.get("max_submissions", 50)),
                final_reserve=int(run.get("final_reserve", 8)),
                initial_calibrations=int(run.get("initial_calibrations", 3)),
                final_start_fraction=float(run.get("final_start_fraction", 0.80)),
                anti_monopoly_fraction=float(run.get("anti_monopoly_fraction", 0.50)),
                min_local_gain=float(run.get("min_local_gain", 0.0)),
                metric_direction=str(run.get("metric_direction", "auto")),
                submission_mode=str(run.get("submission_mode", "auto")),
                max_inflight_submissions=int(
                    run.get("max_inflight_submissions", 2)
                ),
                stop_exploration_minutes_before_end=float(
                    run.get("stop_exploration_minutes_before_end", 120)
                ),
                submission_only_minutes_before_end=float(
                    run.get("submission_only_minutes_before_end", 20)
                ),
                max_retryable_attempts=int(run.get("max_retryable_attempts", 3)),
                retry_backoff_seconds=float(run.get("retry_backoff_seconds", 60)),
                floor_settle_seconds=float(run.get("floor_settle_seconds", 300)),
                floor_settle_candidates=int(
                    run.get("floor_settle_candidates", 3)
                ),
                kernel_timeout_seconds=float(
                    run.get("kernel_timeout_seconds", 1800)
                ),
            ),
            search=SearchConfig(
                enabled=bool(search.get("enabled", True)),
                duration_minutes=float(search.get("duration_minutes", 60)),
                analyst_backend=str(search.get("analyst_backend", "claude")),
                research_backends=tuple(
                    search.get("research_backends", ["claude"] * 4)
                ),  # type: ignore[arg-type]
                prompt_spec=prompt.resolve(),
                analyst_model=str(search.get("analyst_model", "")),
                analyst_effort=str(search.get("analyst_effort", "")),
                research_model=str(search.get("research_model", "")),
                research_effort=str(search.get("research_effort", "")),
            ),
            claude=ClaudeConfig(
                binary=Path(claude.get("binary", "/opt/homebrew/bin/claude")),
                model=str(claude.get("model", "claude-fable-5")),
                effort=str(claude.get("effort", "high")),
                integrated_profile_dir=integrated_profile.resolve(),
                direct_profile_dir=direct_profile.resolve(),
                turn_minutes=float(claude.get("turn_minutes", 45)),
                max_rounds=int(claude.get("max_rounds", 12)),
            ),
            codex=CodexConfig(
                binary=Path(codex.get("binary", "/opt/homebrew/bin/codex")),
                model=str(codex.get("model", "openai/gpt-5.6-sol")),
                effort=str(codex.get("effort", "high")),
                provider_id=str(codex.get("provider_id", "openrouter")),
                base_url=str(codex.get("base_url", "https://openrouter.ai/api/v1")),
                api_key_env=str(codex.get("api_key_env", "OPENROUTER_API_KEY")),
                supports_web_search=bool(codex.get("supports_web_search", True)),
                turn_minutes=float(codex.get("turn_minutes", 45)),
                max_rounds=int(codex.get("max_rounds", 12)),
            ),
            hearsay=HearSayConfig(
                enabled=bool(hearsay.get("enabled", True)),
                model=str(hearsay.get("model", "claude-fable-5")),
                effort=str(hearsay.get("effort", "high")),
                solvers=int(hearsay.get("solvers", 3)),
                rounds=int(hearsay.get("rounds", 40)),
                max_turns=int(hearsay.get("max_turns", 250)),
                solver_models=str(hearsay.get("solver_models", "")),
                solver_efforts=str(hearsay.get("solver_efforts", "")),
                evaluator_model=str(hearsay.get("evaluator_model", "")),
                evaluator_effort=str(hearsay.get("evaluator_effort", "")),
            ),
            selection=SelectionConfig(
                enabled=bool(selection.get("enabled", True)),
                model=str(selection.get("model", "claude-fable-5")),
                effort=str(selection.get("effort", "high")),
                interval_seconds=float(selection.get("interval_seconds", 90)),
                timeout_seconds=float(selection.get("timeout_seconds", 180)),
                recommendation_ttl_seconds=float(
                    selection.get("recommendation_ttl_seconds", 300)
                ),
                fallback_seconds=float(selection.get("fallback_seconds", 240)),
                calibration_min_feedback=int(
                    selection.get("calibration_min_feedback", 4)
                ),
                calibration_ridge_strength=float(
                    selection.get("calibration_ridge_strength", 10.0)
                ),
                calibration_min_rank_gain=float(
                    selection.get("calibration_min_rank_gain", 0.05)
                ),
                calibration_max_blend_weight=float(
                    selection.get("calibration_max_blend_weight", 0.50)
                ),
            ),
            resources=ResourceConfig(
                root=resource_root.resolve(),
                gpu_quota_hours=float(resources.get("gpu_quota_hours", 30)),
                gpu_concurrency=int(resources.get("gpu_concurrency", 2)),
                cpu_concurrency=int(resources.get("cpu_concurrency", 5)),
                default_gpu_kernel_minutes=float(
                    resources.get("default_gpu_kernel_minutes", 45)
                ),
                default_cpu_kernel_minutes=float(
                    resources.get("default_cpu_kernel_minutes", 10)
                ),
                kernel_start_margin_minutes=float(
                    resources.get("kernel_start_margin_minutes", 5)
                ),
                acquire_poll_seconds=float(resources.get("acquire_poll_seconds", 2)),
                acquire_wait_seconds=float(resources.get("acquire_wait_seconds", 5)),
                floor_grace_minutes=float(resources.get("floor_grace_minutes", 15)),
            ),
            evaluation=EvaluationConfig(
                fallback_after_minutes=float(
                    evaluation.get("fallback_after_minutes", 90)
                ),
                fallback_turn_minutes=float(
                    evaluation.get("fallback_turn_minutes", 25)
                ),
            ),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if not 1 <= self.run.max_submissions <= 50:
            raise ValueError("max_submissions must be in 1..50")
        if not 0 <= self.run.final_reserve < self.run.max_submissions:
            raise ValueError("final_reserve must be smaller than max_submissions")
        if self.run.initial_calibrations < 0:
            raise ValueError("initial_calibrations must be non-negative")
        if self.run.metric_direction not in {"auto", "maximize", "minimize"}:
            raise ValueError("metric_direction must be auto, maximize or minimize")
        if self.run.submission_mode not in {"auto", "csv", "kernel"}:
            raise ValueError("submission_mode must be auto, csv, or kernel")
        if self.run.max_inflight_submissions not in range(1, 3):
            raise ValueError("max_inflight_submissions must be 1 or 2")
        if self.run.max_retryable_attempts < 1:
            raise ValueError("max_retryable_attempts must be positive")
        if self.run.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if not 0 < self.run.final_start_fraction < 1:
            raise ValueError("final_start_fraction must be between 0 and 1")
        if self.run.stop_exploration_minutes_before_end < 0:
            raise ValueError("stop_exploration_minutes_before_end must be non-negative")
        if len(self.search.research_backends) != 4:
            raise ValueError("search.research_backends must contain R1..R4")
        if self.hearsay.solvers not in range(1, 9):
            raise ValueError("hearsay.solvers must be in 1..8")
        if self.selection.interval_seconds <= 0 or self.selection.timeout_seconds <= 0:
            raise ValueError("selection interval/timeout must be positive")
        if self.selection.recommendation_ttl_seconds <= 0:
            raise ValueError("selection recommendation TTL must be positive")
        if self.selection.fallback_seconds < self.selection.timeout_seconds:
            raise ValueError("selection fallback_seconds must cover timeout_seconds")
        if self.selection.calibration_min_feedback < 4:
            raise ValueError("calibration_min_feedback must be at least 4")
        if self.selection.calibration_ridge_strength < 0:
            raise ValueError("calibration ridge strength must be non-negative")
        if not 0 <= self.selection.calibration_min_rank_gain <= 2:
            raise ValueError("calibration rank gain must be in 0..2")
        if not 0 <= self.selection.calibration_max_blend_weight <= 0.5:
            raise ValueError("calibration blend weight must be in 0..0.5")
        if self.resources.gpu_quota_hours <= 0:
            raise ValueError("resources.gpu_quota_hours must be positive")
        if self.resources.gpu_concurrency < 1 or self.resources.cpu_concurrency < 1:
            raise ValueError("resource concurrency must be positive")
        if (
            self.resources.default_gpu_kernel_minutes <= 0
            or self.resources.default_cpu_kernel_minutes <= 0
            or self.resources.kernel_start_margin_minutes < 0
        ):
            raise ValueError("resource runtime estimates must be positive")
        if self.resources.acquire_poll_seconds <= 0:
            raise ValueError("resources.acquire_poll_seconds must be positive")
        if self.resources.acquire_wait_seconds <= 0:
            raise ValueError("resources.acquire_wait_seconds must be positive")
        if self.resources.floor_grace_minutes < 0:
            raise ValueError("resources.floor_grace_minutes must be non-negative")
        if (
            self.evaluation.fallback_after_minutes <= 0
            or self.evaluation.fallback_turn_minutes <= 0
        ):
            raise ValueError("evaluation fallback timing must be positive")
