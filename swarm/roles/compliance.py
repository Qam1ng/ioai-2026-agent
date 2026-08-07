"""Compliance: a read-only gate over plans, diffs and kernel scripts.

A violation is a hard stop for two reasons, and the second is the expensive one:
it wastes a submission, and it leaves durable evidence of cheating in the
trajectory the Jury audits after the competition. So the parse is fail-closed —
no report, or an unparseable one, is a fail, never a pass — and any listed
violation forces ``pass`` to False regardless of what the model concluded.

Read-only tools by design: a checker that can execute is a checker that can be
blamed for the state it changed.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Constraint, TaskCard
from .base import Role, RoleResult


class ComplianceRole(Role):
    """Checks an artifact against the TaskCard constraints and the standing rules."""

    role_name = "compliance"
    prompt_name = "compliance"
    tools = ("read_file", "list_dir", "memory_recall")
    max_steps = 12

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_check(report: dict) -> dict:
        """Normalise the check. Violations without a literal quote are dropped.

        A violation stated as a paraphrase cannot be adjudicated — the paraphrase
        is exactly where an imagined rule enters — so it is downgraded to a note
        rather than blocking a candidate on a rule nobody wrote.
        """
        rep = report if isinstance(report, dict) else {}
        if not rep:
            return {
                "pass": False,
                "violations": [
                    {
                        "quote": "(compliance produced no JSON report)",
                        "why": "the gate could not be evaluated; failing closed",
                    }
                ],
                "checked": [],
                "notes": "",
            }

        violations: list[dict] = []
        unquoted: list[str] = []
        for v in rep.get("violations") or []:
            if not isinstance(v, dict):
                continue
            quote = str(v.get("quote", "") or "").strip()
            why = str(v.get("why", "") or "").strip()
            if not quote or not why:
                if why or quote:
                    unquoted.append(why or quote)
                continue
            violations.append({"quote": quote, "why": why})

        notes = str(rep.get("notes", "") or "")
        if unquoted:
            notes = (notes + " | unquoted findings (not treated as violations): "
                     + "; ".join(unquoted)).strip(" |")

        return {
            "pass": bool(rep.get("pass")) and not violations,
            "violations": violations,
            "checked": [str(c) for c in (rep.get("checked") or [])],
            "notes": notes,
        }

    # ---------------------------------------------------------------- run
    def check(
        self,
        plan_or_diff: Any,
        instructions: str = "",
        context: dict | None = None,
    ) -> dict:
        """Check one artifact against the constraints and return the verdict.

        ``plan_or_diff`` may be a PlanCard, a dict, or raw text (a diff or a
        kernel script) — the gate is the same in all three cases.
        """
        card: TaskCard | None = self.bb.get_task_card()
        constraints: list[Constraint] = card.constraints if card else []

        if hasattr(plan_or_diff, "to_dict"):
            artifact: Any = plan_or_diff.to_dict()
        elif isinstance(plan_or_diff, (dict, list)):
            artifact = plan_or_diff
        else:
            artifact = str(plan_or_diff or "")

        instructions = instructions or (
            "Check this artifact against the task card constraints and the standing "
            "competition rules. Quote the literal rule text for every violation."
        )
        ctx = dict(context or {})
        ctx["artifact"] = artifact
        ctx["constraints"] = [c.to_dict() for c in constraints]
        ctx["submission_format"] = card.submission_format if card else ""

        res: RoleResult = self.run(instructions, ctx)
        out = self.parse_check(res.report)
        if not res.ok and not res.report:
            out["pass"] = False
            out["violations"].append(
                {
                    "quote": "(compliance step failed)",
                    "why": res.error or "no report; failing closed",
                }
            )
        self._emit("compliance", passed=out["pass"], n_violations=len(out["violations"]))
        return out
