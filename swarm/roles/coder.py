"""Coder: one PlanCard becomes one self-contained Kaggle kernel script.

The submitted artifact is a script that trains inside the Kaggle kernel, not a
model we trained. That single fact drives the whole tool allowlist: the Coder
needs execution and file writes to build and smoke-test the kernel, and needs
nothing at all that touches Kaggle — pushing and submitting belong to the pod's
runner and broker, which own the quota and the lane budget.
"""

from __future__ import annotations

from typing import Any

from ..schemas import CandidateState, PlanCard
from .base import Role, RoleResult

_STATUSES = ("ready", "blocked", "failed")


class CoderRole(Role):
    """Implements a plan as a kernel directory and smoke-tests it locally."""

    role_name = "coder"
    prompt_name = "coder"
    tools = (
        "run_python",
        "run_bash",
        "write_file",
        "read_file",
        "list_dir",
        "memory_recall",
        "skill_load",
    )
    max_steps = 40

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_build(report: dict, candidate_id: str = "") -> dict:
        """Normalise the Coder report into the build record the pod stores.

        A build that claims ``ready`` without a kernel directory is downgraded
        to ``failed``: the pod would otherwise queue a push against a path that
        does not exist and burn a kernel slot discovering it.
        """
        rep = report if isinstance(report, dict) else {}
        status = str(rep.get("status", "") or "").strip().lower()
        if status not in _STATUSES:
            status = "ready" if rep.get("kernel_dir") else "failed"

        sanity_raw = rep.get("sanity")
        sanity = sanity_raw if isinstance(sanity_raw, dict) else {}
        kernel_dir = str(rep.get("kernel_dir", "") or "")

        problems: list[str] = []
        if not rep:
            problems.append("coder produced no JSON report")
        if not kernel_dir:
            problems.append("no kernel_dir reported")
        if not sanity.get("ran_end_to_end"):
            problems.append("no end-to-end subsample run was reported")
        if not sanity.get("submission_written"):
            problems.append("subsample run did not report writing a submission")
        if not sanity.get("format_matches_sample"):
            problems.append("submission format was not confirmed against the sample")

        if status == "ready" and not kernel_dir:
            status = "failed"

        try:
            runtime = float(rep.get("expected_runtime_min", 0.0) or 0.0)
        except (TypeError, ValueError):
            runtime = 0.0
        accel = str(rep.get("accelerator", "cpu") or "cpu").strip().lower()

        return {
            "candidate_id": str(rep.get("candidate_id", "") or candidate_id),
            "kernel_dir": kernel_dir,
            "entry_file": str(rep.get("entry_file", "") or "main.py"),
            "accelerator": accel if accel in ("cpu", "p100", "t4") else "cpu",
            "expected_runtime_min": runtime,
            "sanity": sanity,
            "status": status,
            "notes": str(rep.get("notes", "") or ""),
            "problems": problems,
        }

    # ---------------------------------------------------------------- run
    def implement(
        self,
        plan: PlanCard | dict,
        candidate: CandidateState | str = "",
        instructions: str = "",
        context: dict | None = None,
    ) -> dict:
        """Build the kernel for ``plan`` and return the normalised build record.

        The candidate's blackboard state is updated here so a crash between the
        build and the Manager's next step does not lose an implemented kernel.
        """
        plan_d = plan.to_dict() if isinstance(plan, PlanCard) else dict(plan or {})
        cand_id = candidate.candidate_id if isinstance(candidate, CandidateState) else str(candidate or "")
        kernel_rel = f"candidates/{cand_id}/kernel" if cand_id else "candidates/kernel"

        instructions = instructions or (
            f"Implement this plan as a self-contained Kaggle kernel under {kernel_rel}/. "
            "It must train inside the kernel, discover /kaggle/input paths at runtime, "
            "always write /kaggle/working/submission.csv, and be smoke-tested on a "
            "subsample before any full run."
        )
        ctx = dict(context or {})
        ctx["plan"] = plan_d
        ctx["candidate_id"] = cand_id
        ctx["kernel_dir"] = kernel_rel

        res: RoleResult = self.run(instructions, ctx)
        build = self.parse_build(res.report, candidate_id=cand_id)
        if not res.ok and not res.report:
            build["status"] = "failed"
            build["problems"].append(f"coder step failed: {res.error or 'no report'}")

        if isinstance(candidate, CandidateState):
            candidate.kernel_dir = build["kernel_dir"] or candidate.kernel_dir
            candidate.status = "ready" if build["status"] == "ready" else "failed"
            candidate.last_error = "; ".join(build["problems"])[:500]
            self.bb.put_candidate(candidate)
        return build
