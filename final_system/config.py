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


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool
    duration_minutes: float
    analyst_backend: str
    research_backends: tuple[str, str, str, str]
    prompt_spec: Path


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


@dataclass(frozen=True)
class SystemConfig:
    run: RunConfig
    search: SearchConfig
    claude: ClaudeConfig
    codex: CodexConfig
    hearsay: HearSayConfig

    @classmethod
    def load(cls, path: Path, *, repo_root: Path) -> "SystemConfig":
        path = Path(path).expanduser().resolve()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        run = _section(raw, "run")
        search = _section(raw, "search")
        claude = _section(raw, "claude")
        codex = _section(raw, "codex")
        hearsay = _section(raw, "hearsay")

        workspace = Path(run.get("workspace_root", "workspace/final_system"))
        if not workspace.is_absolute():
            workspace = repo_root / workspace
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
                metric_direction=str(run.get("metric_direction", "maximize")),
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
            ),
            search=SearchConfig(
                enabled=bool(search.get("enabled", True)),
                duration_minutes=float(search.get("duration_minutes", 60)),
                analyst_backend=str(search.get("analyst_backend", "claude")),
                research_backends=tuple(
                    search.get("research_backends", ["claude"] * 4)
                ),  # type: ignore[arg-type]
                prompt_spec=prompt.resolve(),
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
        if self.run.metric_direction not in {"maximize", "minimize"}:
            raise ValueError("metric_direction must be maximize or minimize")
        if self.run.submission_mode not in {"auto", "csv", "kernel"}:
            raise ValueError("submission_mode must be auto, csv, or kernel")
        if self.run.max_inflight_submissions not in range(1, 3):
            raise ValueError("max_inflight_submissions must be 1 or 2")
        if not 0 < self.run.final_start_fraction < 1:
            raise ValueError("final_start_fraction must be between 0 and 1")
        if self.run.stop_exploration_minutes_before_end < 0:
            raise ValueError("stop_exploration_minutes_before_end must be non-negative")
        if len(self.search.research_backends) != 4:
            raise ValueError("search.research_backends must contain R1..R4")
        if self.hearsay.solvers not in range(1, 9):
            raise ValueError("hearsay.solvers must be in 1..8")
