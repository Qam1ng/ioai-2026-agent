"""Profiler: the competition, read once, into one machine-readable TaskCard.

Only this role reads the raw task. Everything downstream trusts its card, which
is why the parse below is strict about the two fields the phase gate depends on
(metric and submission format) and lenient about everything else: a card missing
the metric cannot be corrected later by a role that never looks at the task,
whereas a missing risk list costs nothing.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Any

from ..schemas import Constraint, TaskCard
from .base import Role, RoleResult

_TASK_TYPES = ("supervised", "imitation", "interactive", "unknown")
_DIRECTIONS = ("maximize", "minimize")


def _kwargs_for(cls: type, d: dict) -> dict:
    """Keep only keys that are real dataclass fields (models invent extras)."""
    allowed = {f.name for f in dc_fields(cls)}
    return {k: v for k, v in d.items() if k in allowed}


class ProfilerRole(Role):
    """Reads the competition and emits the TaskCard."""

    role_name = "profiler"
    prompt_name = "profiler"
    tools = (
        "kaggle_download",
        "list_dir",
        "read_file",
        "run_bash",
        "run_python",
        "write_file",
        "memory_recall",
        "skill_list",
        "skill_load",
    )
    max_steps = 22

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @classmethod
    def parse_card(cls, report: dict, slug: str = "") -> TaskCard:
        """Coerce a raw role report into a validated :class:`TaskCard`.

        Raises ``ValueError`` when the card is unusable, because a pod that
        proceeds on a card without a metric will optimise the wrong quantity for
        six hours and never find out.
        """
        if not isinstance(report, dict) or not report:
            raise ValueError("profiler produced no JSON report")

        raw = dict(report)
        raw.setdefault("slug", slug)
        if not raw.get("slug"):
            raw["slug"] = slug

        constraints: list[Constraint] = []
        for c in raw.get("constraints") or []:
            if not isinstance(c, dict):
                continue
            quote = str(c.get("quote", "")).strip()
            if not quote:
                continue  # a constraint without its literal quote is unenforceable
            kind = str(c.get("kind", "must_not")).strip().lower()
            constraints.append(
                Constraint(
                    kind=kind if kind in ("must", "must_not") else "must_not",
                    quote=quote,
                    rationale=str(c.get("rationale", "")),
                )
            )

        risks = [str(r) for r in (raw.get("risks") or []) if str(r).strip()]

        try:
            conf = float(raw.get("routing_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0

        card_kwargs = _kwargs_for(TaskCard, raw)
        card_kwargs.update(
            constraints=constraints,
            risks=risks,
            routing_confidence=min(1.0, max(0.0, conf)),
        )
        card = TaskCard(**card_kwargs)

        # Normalise the two enum-ish fields rather than rejecting the card: an
        # unrecognised label routes to the conservative branch, it does not
        # invalidate a metric that was read correctly.
        if card.task_type not in _TASK_TYPES:
            card.task_type = "unknown"
            card.routing_confidence = min(card.routing_confidence, 0.4)
        if card.metric_direction not in _DIRECTIONS:
            card.risks.append(
                f"metric_direction was reported as {card.metric_direction!r}; "
                "defaulted to maximize and needs confirmation"
            )
            card.metric_direction = "maximize"

        missing = [
            k
            for k in ("slug", "metric_name", "submission_format")
            if not str(getattr(card, k, "") or "").strip()
        ]
        if missing:
            raise ValueError(f"task card missing required field(s): {', '.join(missing)}")
        return card

    # ---------------------------------------------------------------- run
    def profile(self, instructions: str = "", context: dict | None = None) -> TaskCard:
        """Run the profiler and return the validated card.

        The card is written to the blackboard here so that a crash after the
        LLM spend but before the pod stores it does not throw the work away.
        """
        instructions = instructions or (
            "Download and read the competition. Produce the TaskCard: metric and its "
            "direction, exact submission format, data summary with the runtime path "
            "discovery rule, the grouping variable for CV, and every constraint as a "
            "literal quote."
        )
        res: RoleResult = self.run(instructions, context)
        if not res.ok and not res.report:
            raise ValueError(f"profiler failed: {res.error or 'no report'}")
        card = self.parse_card(res.report, slug=self.cfg.slug)
        self.bb.put_task_card(card)
        return card
