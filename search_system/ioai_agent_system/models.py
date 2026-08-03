from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HypothesisCard:
    id: str
    owner: str
    claim: str
    data_view: str
    representation: str
    model_family: str
    structural_bias: str
    objective: str
    augmentation: str
    inference: str
    expected_gain: str = "medium"
    uncertainty: str = "high"
    forbidden_overlap: tuple[str, ...] = ()

    @property
    def signature(self) -> tuple[str, ...]:
        return (
            self.data_view,
            self.representation,
            self.model_family,
            self.structural_bias,
            self.objective,
            self.augmentation,
            self.inference,
        )

    def distance(self, other: "HypothesisCard") -> int:
        return sum(a != b for a, b in zip(self.signature, other.signature, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HypothesisCard":
        allowed = cls.__dataclass_fields__.keys()
        cleaned = {key: value[key] for key in allowed if key in value}
        cleaned["forbidden_overlap"] = tuple(cleaned.get("forbidden_overlap", ()))
        return cls(**cleaned)


@dataclass
class AgentResult:
    agent_id: str
    role: str
    workdir: Path
    trace_path: Path
    exit_code: int | None
    timed_out: bool
    duration_s: float
    resolved_model: str | None = None
    session_id: str | None = None
    cost_usd: float = 0.0
    result_text: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class CandidateEvaluation:
    candidate_id: str
    workdir: Path
    submission_path: Path | None
    valid: bool
    offline_score: float | None
    count_mae: float | None
    count_mape: float | None
    density_mse: float | None
    prediction_hash: str | None
    runtime_s: float | None
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def rank_key(self) -> tuple[int, float, float]:
        score = self.offline_score if self.offline_score is not None else float("-inf")
        runtime = self.runtime_s if self.runtime_s is not None else float("inf")
        return (int(self.valid), score, -runtime)

