"""Shared data contracts for the multi-agent layer.

Every artifact that crosses a role boundary is one of these. They are plain
dataclasses with explicit ``to_dict``/``from_dict`` so the blackboard stays
human-readable JSON that the Jury can audit after the competition.

Label conventions used across the swarm:

* ``task_type`` — routes to a playbook: ``supervised`` | ``imitation`` |
  ``interactive`` | ``unknown``.
* ``lane`` — a submission lane: ``floor`` (insurance), ``probe`` (cheap,
  information-seeking, CPU kernel), ``milestone`` (a genuine candidate) or
  ``final`` (end-of-window consolidation).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

TaskType = str  # supervised | imitation | interactive | unknown
Lane = str  # floor | probe | milestone | final
Accel = str  # cpu | p100 | t4


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _now() -> float:
    return time.time()


@dataclass
class Constraint:
    """One rule lifted verbatim from the task statement.

    ``quote`` must be the literal source text. Compliance decisions are made
    against the quote, never against a paraphrase, because a paraphrase is
    where a violation quietly enters.
    """

    kind: str  # must | must_not
    quote: str
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Constraint":
        return cls(**d)


@dataclass
class TaskCard:
    """Profiler output: the machine-readable reading of the task."""

    slug: str
    title: str = ""
    task_type: TaskType = "unknown"
    metric_name: str = ""
    metric_description: str = ""
    metric_direction: str = "maximize"  # maximize | minimize
    submission_format: str = ""
    data_summary: str = ""
    constraints: list[Constraint] = field(default_factory=list)
    grouping_variable: str = ""  # what CV folds must be grouped by, "" if none
    risks: list[str] = field(default_factory=list)
    routing_confidence: float = 0.0  # 0..1; low -> pod runs two playbooks
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["constraints"] = [c.to_dict() for c in self.constraints]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskCard":
        d = dict(d)
        d["constraints"] = [Constraint.from_dict(c) for c in d.get("constraints", [])]
        return cls(**d)


@dataclass
class PlanCard:
    """Designer output: one candidate approach, before any code exists."""

    plan_id: str = field(default_factory=lambda: new_id("plan"))
    title: str = ""
    family: str = ""  # method family; used to enforce diversity across candidates
    architecture: str = ""
    data_strategy: str = ""
    loss: str = ""
    training_config: str = ""
    validation_strategy: str = ""
    expected_runtime_min: float = 0.0
    accelerator: Accel = "cpu"
    rationale: str = ""
    revises: str = ""  # plan_id this revises, "" if fresh
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanCard":
        return cls(**d)


@dataclass
class CandidateState:
    """Live state of one parallel solution candidate (an AIBuildAI 'repo')."""

    candidate_id: str = field(default_factory=lambda: new_id("cand"))
    plan_id: str = ""
    family: str = ""
    status: str = "created"  # created|coding|tuning|ready|failed|retired
    local_score: float | None = None
    local_std: float | None = None
    #: Who produced local_score. Only "harness" numbers count for selection.
    #: Run 3 measured why: a coder's self-reported 0.1215 was quoted by the
    #: Manager as "verified"; the harness later said 0.1669 (+0.045).
    score_source: str = ""  # "" | harness | coder
    #: A coder's own unverified measurement, quarantined from selection.
    claimed_score: float | None = None
    folds: list[float] = field(default_factory=list)
    kernel_dir: str = ""  # relative to workspace
    last_error: str = ""
    retire_reason: str = ""
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateState":
        return cls(**d)


@dataclass
class Experiment:
    """One ledger row. Appended, never mutated."""

    exp_id: str = field(default_factory=lambda: new_id("exp"))
    candidate_id: str = ""
    role: str = ""
    description: str = ""
    local_score: float | None = None
    local_std: float | None = None
    folds: list[float] = field(default_factory=list)
    runtime_s: float = 0.0
    accepted: bool = False
    reject_reason: str = ""
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        return cls(**d)


@dataclass
class SubmissionRecord:
    """One real Kaggle submission. The audit trail the Jury will read."""

    sub_id: str = field(default_factory=lambda: new_id("sub"))
    candidate_id: str = ""
    lane: Lane = "milestone"
    purpose: str = ""
    kernel_ref: str = ""
    kernel_version: int | None = None
    accelerator: Accel = "cpu"
    local_score: float | None = None
    lb_score: float | None = None
    status: str = "queued"  # queued|running|scored|error
    gpu_seconds: float = 0.0
    error: str = ""
    submitted_at: float = field(default_factory=_now)
    scored_at: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SubmissionRecord":
        return cls(**d)


@dataclass
class ProbeResult:
    """What a probe submission taught us about the hidden test set."""

    probe_id: str = field(default_factory=lambda: new_id("probe"))
    probe_kind: str = ""  # constant | granularity | split_shift | noise_floor | calibration
    hypothesis: str = ""
    lb_score: float | None = None
    inference: str = ""  # plain-language conclusion drawn from the score
    numeric: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeResult":
        return cls(**d)


@dataclass
class Calibration:
    """Regression of leaderboard score on local CV score.

    ``noise_band`` is the practical significance threshold: a local gain
    smaller than this is indistinguishable from noise and must not trigger a
    milestone submission.
    """

    n_points: int = 0
    slope: float | None = None
    intercept: float | None = None
    residual_std: float | None = None
    noise_band: float = 0.0
    gap: float | None = None  # mean(local) - mean(lb); positive == optimistic
    warning: str = ""
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(**d)


@dataclass
class ManagerAction:
    """The single action a Manager step emits (AIBuildAI's one-action loop)."""

    action: str  # profile|design|code|tune|verify|probe|aggregate|report|retire|wait|stop
    target: str = ""  # candidate_id / plan_id, when applicable
    reason: str = ""
    instructions: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ManagerAction":
        return cls(
            action=d.get("action", "wait"),
            target=d.get("target", "") or "",
            reason=d.get("reason", "") or "",
            instructions=d.get("instructions", "") or "",
        )


VALID_ACTIONS = {
    "profile",
    "design",
    "code",
    "tune",
    "verify",
    "probe",
    "aggregate",
    "report",
    "retire",
    "wait",
    "stop",
}
