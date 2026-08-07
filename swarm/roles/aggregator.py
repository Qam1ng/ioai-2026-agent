"""Aggregator: combine verified candidates without retraining from scratch.

It runs at consolidation time, after the freeze gate, so there is neither clock
nor GPU quota for a fresh training run. The only fitting allowed is combination
weights on out-of-fold predictions.

The strategy must match the metric, not habit: rank-blend for ranking metrics,
probability averaging for threshold metrics, probability-map averaging for dense
prediction, weighted averaging for regression. And the blend must fit one kernel
runtime — five members at 20 minutes do not fit a 30-minute budget, so members
get dropped here rather than discovered too late on Kaggle.
"""

from __future__ import annotations

from typing import Any

from ..schemas import CandidateState
from .base import Role, RoleResult

STRATEGIES = (
    "rank_blend",
    "prob_average",
    "prob_map_average",
    "weighted_average",
    "vote",
    "single_best",
)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class AggregatorRole(Role):
    """Builds the consolidation kernel from already-verified candidates."""

    role_name = "aggregator"
    prompt_name = "aggregator"
    tools = ("run_python", "write_file", "read_file", "list_dir", "memory_recall")
    max_steps = 26

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_blend(report: dict, runtime_budget_min: float | None = None) -> dict:
        """Normalise the blend spec and flag anything that would not survive Kaggle."""
        rep = report if isinstance(report, dict) else {}
        problems: list[str] = []
        if not rep:
            problems.append("aggregator produced no JSON report")

        strategy = str(rep.get("strategy", "") or "").strip().lower()
        if strategy not in STRATEGIES:
            problems.append(f"unrecognised strategy {rep.get('strategy')!r}")
            strategy = strategy or "single_best"

        members: list[dict] = []
        for m in rep.get("members") or []:
            if isinstance(m, dict) and m.get("candidate_id"):
                members.append(
                    {
                        "candidate_id": str(m["candidate_id"]),
                        "weight": _as_float(m.get("weight")) or 0.0,
                        "local_score": _as_float(m.get("local_score")),
                        "family": str(m.get("family", "") or ""),
                    }
                )
            elif isinstance(m, str) and m.strip():
                members.append({"candidate_id": m.strip(), "weight": 0.0,
                                "local_score": None, "family": ""})
        if not members:
            problems.append("no blend members named")

        code = rep.get("code")
        code = code if isinstance(code, str) else ""
        if not code.strip():
            problems.append("no kernel code produced")

        runtime = _as_float(rep.get("expected_runtime_min")) or 0.0
        if runtime_budget_min and runtime > runtime_budget_min:
            problems.append(
                f"blend runtime {runtime:g}min exceeds the kernel budget "
                f"{runtime_budget_min:g}min; drop members"
            )

        # Weights are advisory; normalise so a downstream kernel can use them
        # directly, and fall back to equal weights when the model left them at 0.
        total = sum(m["weight"] for m in members)
        if members and total <= 0:
            for m in members:
                m["weight"] = 1.0 / len(members)
        elif total > 0:
            for m in members:
                m["weight"] = m["weight"] / total

        return {
            "strategy": strategy,
            "members": members,
            "dropped": [d for d in (rep.get("dropped") or []) if isinstance(d, dict)],
            "oof_score": _as_float(rep.get("oof_score")),
            "oof_std": _as_float(rep.get("oof_std")),
            "expected_runtime_min": runtime,
            "code": code,
            "notes": str(rep.get("notes", "") or ""),
            "problems": problems,
            "ok": not problems,
        }

    # ---------------------------------------------------------------- run
    def aggregate(
        self,
        candidates: list[CandidateState] | None = None,
        instructions: str = "",
        context: dict | None = None,
        *,
        runtime_budget_min: float = 30.0,
    ) -> dict:
        """Build a blend over ``candidates`` (default: live, scored candidates)."""
        pool = candidates if candidates is not None else [
            c for c in self.bb.live_candidates() if c.local_score is not None
        ]
        instructions = instructions or (
            "Combine these verified candidates into one kernel that fits "
            f"{runtime_budget_min:g} minutes. Do not retrain from scratch and do not add a "
            "new family. Pick the combination rule from the metric, choose members for "
            "decorrelation, and drop members until the runtime fits."
        )
        ctx = dict(context or {})
        ctx["members"] = [c.to_dict() for c in pool]
        ctx["runtime_budget_min"] = runtime_budget_min
        card = self.bb.get_task_card()
        ctx["metric"] = {
            "name": card.metric_name if card else "",
            "description": card.metric_description if card else "",
            "direction": card.metric_direction if card else "maximize",
        }

        res: RoleResult = self.run(instructions, ctx)
        out = self.parse_blend(res.report, runtime_budget_min=runtime_budget_min)
        if not res.ok and not res.report:
            out["ok"] = False
            out["problems"].append(f"aggregator step failed: {res.error or 'no report'}")
        self._emit("aggregate", strategy=out["strategy"], n_members=len(out["members"]),
                   ok=out["ok"])
        return out
