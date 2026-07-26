"""Swarm configuration: model routing, parallelism, gates, budgets.

Loaded from YAML with environment-variable overrides so a competition-day run
can be re-pointed at a different model without editing code (the runtime model
is expected to change from Gemini 3.1 Pro during development to Fable 5 /
Opus 5 on the day).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # PyYAML is in requirements but keep the import soft for a bare checkout
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

ROLES = (
    "profiler",
    "manager",
    "designer",
    "coder",
    "tuner",
    "verifier",
    "prober",
    "compliance",
    "aggregator",
    "reporter",
)


@dataclass
class ModelSpec:
    """Which brain a role uses.

    ``backend`` selects the provider adapter (``claude`` | ``openai``).
    Gemini and other OpenAI-compatible internal endpoints use ``openai`` with
    ``base_url`` pointed at the gateway.
    """

    backend: str = "claude"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    max_tokens: int = 8000

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateConfig:
    """Hard deadlines the orchestrator enforces regardless of what the Manager wants."""

    floor_submit_min: float = 45.0  # a valid submission must exist by T+45min
    freeze_min: float = 315.0  # T+5h15m: no new directions, consolidate only
    final_submit_buffer_min: float = 20.0  # last submission no later than T-20min
    report_deadline_min: float = 30.0  # report due within 30min of window close


@dataclass
class ParallelConfig:
    max_candidates: int = 5  # concurrent solution candidates per problem
    max_candidates_hard: int = 8  # ceiling when GPU is plentiful
    max_concurrent_kernels: int = 3  # concurrent Kaggle kernel runs
    coder_timeout_s: int = 1800
    tuner_timeout_s: int = 2400


@dataclass
class SubmitConfig:
    max_submissions: int = 50
    reserve_for_final: int = 4  # never spend these on exploration
    floor_lane: int = 1
    probe_lane: int = 12  # cheap CPU probes; free against GPU quota
    milestone_min_gain_sigma: float = 1.0  # gain must exceed this x noise band
    default_gpu_seconds: float = 1800.0  # reservation per GPU kernel
    poll_interval_s: int = 30
    poll_max_min: float = 45.0


@dataclass
class SwarmConfig:
    slug: str = ""
    workspace: str = ""
    deadline_min: float = 360.0
    max_cost_usd: float = 1000.0
    gpu_quota_hours: float = 30.0
    kaggle_user: str = "qam1ng"
    default_model: ModelSpec = field(default_factory=ModelSpec)
    models: dict[str, ModelSpec] = field(default_factory=dict)
    gates: GateConfig = field(default_factory=GateConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    submit: SubmitConfig = field(default_factory=SubmitConfig)
    dry_run: bool = False  # never touch Kaggle; used for local rehearsal

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "SwarmConfig":
        data: dict = {}
        if path:
            p = Path(path)
            if p.exists():
                if yaml is None:
                    raise RuntimeError("PyYAML required to load a config file")
                data = yaml.safe_load(p.read_text()) or {}

        cfg = cls()
        for key in ("slug", "workspace", "kaggle_user"):
            if key in data:
                setattr(cfg, key, data[key])
        for key in ("deadline_min", "max_cost_usd", "gpu_quota_hours"):
            if key in data:
                setattr(cfg, key, float(data[key]))
        if "dry_run" in data:
            cfg.dry_run = bool(data["dry_run"])

        if "default_model" in data:
            cfg.default_model = ModelSpec(**data["default_model"])
        for role, spec in (data.get("models") or {}).items():
            cfg.models[role] = ModelSpec(**spec)
        for section, klass in (
            ("gates", GateConfig),
            ("parallel", ParallelConfig),
            ("submit", SubmitConfig),
        ):
            if section in data:
                setattr(cfg, section, klass(**data[section]))

        cfg._apply_env()
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def _apply_env(self) -> None:
        """Environment wins over the file — this is the competition-day switch."""
        backend = os.environ.get("SWARM_BACKEND")
        model = os.environ.get("SWARM_MODEL")
        base_url = os.environ.get("SWARM_BASE_URL")
        key_env = os.environ.get("SWARM_API_KEY_ENV")
        if backend:
            self.default_model.backend = backend
        if model:
            self.default_model.model = model
        if base_url:
            self.default_model.base_url = base_url
        if key_env:
            self.default_model.api_key_env = key_env
        if os.environ.get("SWARM_DRY_RUN"):
            self.dry_run = os.environ["SWARM_DRY_RUN"].lower() not in ("0", "false", "")
        if os.environ.get("SWARM_MAX_CANDIDATES"):
            self.parallel.max_candidates = int(os.environ["SWARM_MAX_CANDIDATES"])
        if os.environ.get("SWARM_GPU_QUOTA_HOURS"):
            self.gpu_quota_hours = float(os.environ["SWARM_GPU_QUOTA_HOURS"])

    # ------------------------------------------------------------- lookups
    def model_for(self, role: str) -> ModelSpec:
        return self.models.get(role, self.default_model)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["default_model"] = self.default_model.to_dict()
        d["models"] = {k: v.to_dict() for k, v in self.models.items()}
        return d
