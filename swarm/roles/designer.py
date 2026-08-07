"""Designer: K decorrelated approaches, as PlanCards, before any code exists.

Diversity is enforced in code, not only in the prompt. Models asked for five
approaches reliably return five variants of one approach; five plans that fail
for the same reason buy one plan's worth of information while costing five
plans' worth of GPU quota. The ``family`` label is the axis of decorrelation and
duplicates are dropped here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields as dc_fields
from typing import Any

from ..schemas import PlanCard
from .base import Role, RoleResult

_ACCELERATORS = ("cpu", "p100", "t4")


def _kwargs_for(cls: type, d: dict) -> dict:
    """Keep only keys that are real dataclass fields (models invent extras)."""
    allowed = {f.name for f in dc_fields(cls)}
    return {k: v for k, v in d.items() if k in allowed}


def _norm_family(value: Any) -> str:
    """Families are compared case/spacing-insensitively so 'GBDT' == 'gbdt'."""
    return "_".join(str(value or "").strip().lower().replace("-", "_").split())


class DesignerRole(Role):
    """Produces the plan set the pod's candidates are built from."""

    role_name = "designer"
    prompt_name = "designer"
    # Read-only: the Designer decides, the Coder executes. Giving it execution
    # tools is how a "plan" quietly becomes a half-written implementation that
    # nobody verified.
    tools = ("read_file", "list_dir", "memory_recall", "skill_list", "skill_load")
    max_steps = 14

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_plans(report: dict) -> list[PlanCard]:
        """Coerce a raw report into PlanCards. Unusable entries are skipped."""
        if not isinstance(report, dict):
            return []
        raw_plans = report.get("plans")
        if not isinstance(raw_plans, list):
            # Tolerate a single-plan report shaped like one plan.
            raw_plans = [report] if report.get("family") or report.get("title") else []

        out: list[PlanCard] = []
        for item in raw_plans:
            if not isinstance(item, dict):
                continue
            kwargs = _kwargs_for(PlanCard, item)
            kwargs.pop("plan_id", None)  # ids are minted here, never by the model
            kwargs.pop("created_at", None)
            try:
                kwargs["expected_runtime_min"] = float(kwargs.get("expected_runtime_min", 0.0) or 0.0)
            except (TypeError, ValueError):
                kwargs["expected_runtime_min"] = 0.0
            accel = str(kwargs.get("accelerator", "cpu") or "cpu").strip().lower()
            kwargs["accelerator"] = accel if accel in _ACCELERATORS else "cpu"
            kwargs["family"] = _norm_family(kwargs.get("family"))
            plan = PlanCard(**kwargs)
            if not plan.family and not plan.title:
                continue  # an unlabelled plan cannot be diversified or tracked
            if not plan.family:
                plan.family = _norm_family(plan.title)
            out.append(plan)
        return out

    @staticmethod
    def duplicate_families(plans: list[PlanCard]) -> list[str]:
        """Families that appear more than once — the diversity violation."""
        counts = Counter(_norm_family(p.family) for p in plans)
        return sorted(f for f, n in counts.items() if f and n > 1)

    @staticmethod
    def enforce_diversity(plans: list[PlanCard]) -> list[PlanCard]:
        """Keep the first plan of each family; drop later duplicates.

        First wins because the Designer is told to put the floor plan first: the
        cheap, submittable one must survive de-duplication.
        """
        seen: set[str] = set()
        kept: list[PlanCard] = []
        for p in plans:
            fam = _norm_family(p.family)
            if fam in seen:
                continue
            seen.add(fam)
            kept.append(p)
        return kept

    # ---------------------------------------------------------------- run
    def design(self, instructions: str = "", context: dict | None = None, k: int = 3) -> list[PlanCard]:
        """Run the Designer and return up to ``k`` family-distinct plans.

        Returns fewer than ``k`` plans rather than padding with duplicates: the
        Manager can ask again, but a duplicate family spends a candidate slot on
        information the pod already has.
        """
        k = max(1, int(k))
        instructions = instructions or (
            f"Produce {k} candidate approaches with distinct method families. Plan 1 is the "
            "floor: the simplest thing that yields a valid submission fast, CPU if possible."
        )
        res: RoleResult = self.run(instructions, context)
        plans = self.parse_plans(res.report)
        dups = self.duplicate_families(plans)
        if dups:
            self._emit("plan_diversity_violation", families=dups, n_plans=len(plans))
        plans = self.enforce_diversity(plans)[:k]
        if len(plans) < k:
            self._emit("plan_shortfall", asked=k, got=len(plans), duplicate_families=dups)
        for p in plans:
            self.bb.add_plan(p)
        return plans
