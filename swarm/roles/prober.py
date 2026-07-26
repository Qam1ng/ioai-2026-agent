"""Prober: designs cheap submissions that buy information about the hidden test set.

Submissions are the pod's cheapest measuring instrument — 50 per problem against
a 6-hour clock and a shared 30h GPU quota — but only while the probe kernel is
CPU-only. The accelerator is therefore forced to ``cpu`` here rather than left to
the model: a GPU probe spends the resource the real candidates are competing for.
"""

from __future__ import annotations

from typing import Any

from ..schemas import ProbeResult
from .base import Role, RoleResult

PROBE_KINDS = ("constant", "granularity", "split_shift", "noise_floor", "calibration")


class ProberRole(Role):
    """Turns "what don't we know about the test set?" into a ranked probe list."""

    role_name = "prober"
    prompt_name = "prober"
    # It reads the submission history to avoid re-buying known information, but
    # cannot submit: the broker owns the lane budget.
    tools = ("run_python", "read_file", "list_dir", "kaggle_submissions", "memory_recall")
    max_steps = 14

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_probes(report: dict) -> list[dict]:
        """Normalise the probe list: known kinds, CPU only, sorted by priority.

        A probe without a ``decision_rule`` is dropped. A probe whose outcome
        changes nothing is a submission spent on curiosity.
        """
        if not isinstance(report, dict):
            return []
        raw = report.get("probes")
        if not isinstance(raw, list):
            raw = [report] if report.get("probe_kind") else []

        out: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("probe_kind", "") or "").strip().lower()
            hypothesis = str(item.get("hypothesis", "") or "").strip()
            decision = str(item.get("decision_rule", "") or "").strip()
            if not hypothesis or not decision:
                continue
            try:
                priority = int(item.get("priority", 99))
            except (TypeError, ValueError):
                priority = 99
            try:
                runtime = float(item.get("expected_runtime_min", 0.0) or 0.0)
            except (TypeError, ValueError):
                runtime = 0.0
            out.append(
                {
                    "probe_kind": kind if kind in PROBE_KINDS else "constant",
                    "hypothesis": hypothesis,
                    "payload": str(item.get("payload", "") or ""),
                    "accelerator": "cpu",  # never negotiable: GPU quota is for candidates
                    "expected_runtime_min": runtime,
                    "decision_rule": decision,
                    "priority": priority,
                    "lane": "probe",
                }
            )
        out.sort(key=lambda p: p["priority"])
        return out

    # ---------------------------------------------------------------- run
    def plan_probes(
        self,
        instructions: str = "",
        context: dict | None = None,
        n: int = 3,
    ) -> list[dict]:
        """Return up to ``n`` probes, ranked by information per submission."""
        n = max(1, int(n))
        instructions = instructions or (
            f"Design up to {n} CPU probes that buy information about the hidden test set "
            "(base rate, test size from score granularity, distribution shift, LB noise "
            "floor, local-to-LB calibration). Each needs a decision rule. Skip anything "
            "the blackboard already answers."
        )
        ctx = dict(context or {})
        ctx.setdefault("existing_probes", [p.to_dict() for p in self.bb.get_probes()])
        ctx.setdefault("calibration", self.bb.get_calibration().to_dict())
        ctx.setdefault(
            "scored_submissions",
            [
                {"lane": s.lane, "local_score": s.local_score, "lb_score": s.lb_score}
                for s in self.bb.scored_submissions()
            ],
        )

        res: RoleResult = self.run(instructions, ctx)
        probes = self.parse_probes(res.report)[:n]
        self._emit("probes_planned", n=len(probes),
                   kinds=[p["probe_kind"] for p in probes])
        return probes

    # -------------------------------------------------------------- record
    @staticmethod
    def to_probe_result(probe: dict, lb_score: float | None = None, inference: str = "") -> ProbeResult:
        """Turn a planned probe plus its returned score into a ledger row."""
        return ProbeResult(
            probe_kind=probe.get("probe_kind", "constant"),
            hypothesis=probe.get("hypothesis", ""),
            lb_score=lb_score,
            inference=inference,
            numeric={"expected_runtime_min": probe.get("expected_runtime_min", 0.0)},
        )
