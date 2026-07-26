"""The task pod: everything that happens for one problem, for six hours.

Division of authority, which is the whole design in one sentence: **the pod
enforces what must never go wrong, the Manager decides everything that needs
intelligence.** Deadlines, the floor submission, the freeze window and the
significance gate are deterministic code the Manager cannot argue with;
which approach to try, what to fix, what to abandon is entirely the Manager's.

That split exists because the failure modes are asymmetric. A Manager that
picks a mediocre model loses a few points; a Manager that talks itself out of
submitting anything scores zero.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import PodBudget, QuotaPool
from .bus import Blackboard
from .config import SwarmConfig
from .ledger import fit_calibration
from .schemas import (
    CandidateState,
    Experiment,
    ManagerAction,
    PlanCard,
    TaskCard,
)


@dataclass
class PodResult:
    slug: str
    ok: bool
    best_local: float | None = None
    best_lb: float | None = None
    n_submissions: int = 0
    n_candidates: int = 0
    stop_reason: str = ""
    report_path: str = ""
    errors: list[str] = field(default_factory=list)


class TaskPod:
    """Drives one problem end to end."""

    def __init__(
        self,
        cfg: SwarmConfig,
        bb: Blackboard,
        budget: PodBudget,
        quota: QuotaPool,
        broker: Any,
        roles: dict[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.bb = bb
        self.budget = budget
        self.quota = quota
        self.broker = broker
        self._roles_override = roles or {}
        self._pool = ThreadPoolExecutor(
            max_workers=max(2, cfg.parallel.max_candidates), thread_name_prefix="cand"
        )
        self._futures: dict[str, Future] = {}
        self._errors: list[str] = []
        self._floor_done = False
        self._stop_reason = ""

    # ------------------------------------------------------------- roles
    def _role(self, name: str, **kwargs: Any):
        """Instantiate a specialist. Roles are short-lived by design."""
        if name in self._roles_override:
            return self._roles_override[name]
        from .roles import ROLE_CLASSES

        klass = ROLE_CLASSES[name]
        return klass(name, self.cfg, self.bb, self.budget, **kwargs)

    # ------------------------------------------------------------- gates
    def _minutes(self) -> float:
        return self.budget.elapsed() / 60.0

    def in_freeze(self) -> bool:
        return self._minutes() >= self.cfg.gates.freeze_min

    def floor_overdue(self) -> bool:
        return (not self._floor_done) and self._minutes() >= self.cfg.gates.floor_submit_min

    def _has_floor(self) -> bool:
        return any(s.lane == "floor" for s in self.bb.get_submissions())

    def enforce_gates(self) -> str | None:
        """Deterministic obligations. Returns a stop reason when the pod is done."""
        halt = self.budget.exhausted()
        if halt:
            return halt

        if self._has_floor():
            self._floor_done = True

        # Obligation 1: a valid submission must exist early. Everything else is
        # optional; this is not. A pod with no submission scores zero, and the
        # score is normalised within the AI track, so zero is maximally costly.
        if self.floor_overdue():
            self.bb.event("gate", gate="floor_overdue", minutes=round(self._minutes(), 1))
            self._force_floor_submission()

        # Obligation 2: after the freeze, stop opening new directions.
        if self.in_freeze():
            self.bb.event("gate", gate="freeze", minutes=round(self._minutes(), 1))

        last_call = self.cfg.deadline_min - self.cfg.gates.final_submit_buffer_min
        if self._minutes() >= last_call:
            return "final submission buffer reached"
        return None

    def _force_floor_submission(self) -> None:
        """Submit the simplest thing that exists, right now.

        Called when the Manager has not produced a floor submission in time.
        Prefers a real candidate; falls back to a trivial constant-prediction
        kernel, because a scored trivial submission beats an unscored good one.
        """
        try:
            best = self.bb.best_candidate()
            if best is not None and best.kernel_dir:
                code = (Path(self.bb.ws) / best.kernel_dir / "kernel.py")
                if code.exists():
                    self.broker.submit(
                        code=code.read_text(),
                        candidate_id=best.candidate_id,
                        lane="floor",
                        purpose="forced floor submission (gate)",
                        accelerator="cpu",
                        local_score=best.local_score,
                    )
                    self._floor_done = True
                    return
            fallback = self._fallback_kernel()
            if fallback:
                self.broker.submit(
                    code=fallback,
                    candidate_id="fallback",
                    lane="floor",
                    purpose="forced floor submission (trivial fallback)",
                    accelerator="cpu",
                    local_score=None,
                )
                self._floor_done = True
        except Exception as exc:
            self._errors.append(f"floor submission failed: {exc}")
            self.bb.event("gate_error", gate="floor", error=str(exc))

    def _fallback_kernel(self) -> str:
        """The dumbest entry that can still score: echo whatever sample exists.

        Not every task ships a CSV sample -- practice task 2 expects a
        ``predictions.zip`` -- so this searches for any sample-shaped artifact
        and reproduces it under its own name. When a task has no sample at all
        there is nothing generic left to guess, and the kernel says so loudly
        rather than writing a file the grader will silently reject.

        This is a last resort. The real floor comes from the first plan, which
        the Designer is instructed to make a minimal valid submission.
        """
        return (
            "import os, shutil\n"
            "\n"
            "CANDIDATES = ('sample_submission', 'submission', 'sample_predictions', 'predictions')\n"
            "found = []\n"
            "for root, _dirs, files in os.walk('/kaggle/input'):\n"
            "    for f in files:\n"
            "        stem, ext = os.path.splitext(f)\n"
            "        if stem.lower() in CANDIDATES and ext.lower() in ('.csv', '.zip', '.json'):\n"
            "            found.append(os.path.join(root, f))\n"
            "os.makedirs('/kaggle/working', exist_ok=True)\n"
            "if not found:\n"
            "    raise SystemExit(\n"
            "        'FLOOR KERNEL: no sample submission under /kaggle/input; this task needs a\\n'\n"
            "        'task-specific floor. Listing the input tree so the next attempt can see it:\\n'\n"
            "        + '\\n'.join(\n"
            "            os.path.join(r, f)\n"
            "            for r, _d, fs in os.walk('/kaggle/input')\n"
            "            for f in fs\n"
            "        )[:4000]\n"
            "    )\n"
            "# Prefer a CSV when several samples exist: it is the commonest grader input.\n"
            "found.sort(key=lambda p: (not p.lower().endswith('.csv'), len(p)))\n"
            "src = found[0]\n"
            "dst = os.path.join('/kaggle/working', os.path.basename(src))\n"
            "shutil.copyfile(src, dst)\n"
            "print('floor kernel copied', src, '->', dst)\n"
        )

    # -------------------------------------------------------- bootstrap
    def ensure_task_card(self) -> TaskCard:
        card = self.bb.get_task_card()
        if card is not None:
            return card
        role = self._role("profiler")
        res = role.run(
            "Read the competition data and statement. Produce the task card: task type, "
            "metric (and its exact definition), submission format, data summary, every "
            "MUST/MUST-NOT constraint quoted verbatim from the statement, the variable "
            "that cross-validation folds must be grouped by, and the main risks."
        )
        card = role.parse(res) if hasattr(role, "parse") else None
        if card is None:
            card = TaskCard(slug=self.cfg.slug, routing_confidence=0.0)
            self._errors.append("profiler produced no task card; running with an empty one")
        self.bb.put_task_card(card)
        return card

    def ensure_plans(self, k: int) -> list[PlanCard]:
        plans = self.bb.get_plans()
        if len(plans) >= k:
            return plans
        role = self._role("designer")
        res = role.run(
            f"Propose {k} candidate approaches.\n\n"
            "PLAN 1 IS MANDATORY AND FIXED: the fastest possible *valid* submission for this "
            "task -- a constant or trivial prediction in the exact required output format, "
            "CPU-only, no training. Two of the three practice competitions ship no sample "
            "submission file, so a generic 'copy the sample' fallback cannot save us; the "
            "floor has to be built from the task's own output spec. Give it family 'floor'.\n\n"
            f"Plans 2..{k} must span genuinely different method families -- not variants of "
            "one idea. For each give architecture, data strategy, loss, training config, "
            "validation strategy, expected runtime and the accelerator it needs. Write no code.",
            context={"task_card": (self.bb.get_task_card() or TaskCard(self.cfg.slug)).to_dict()},
        )
        new = role.parse(res) if hasattr(role, "parse") else []
        for p in new:
            self.bb.add_plan(p)
        return self.bb.get_plans()

    # -------------------------------------------------------- candidates
    def launch_candidate(self, plan: PlanCard, instructions: str = "") -> str:
        """Start a candidate in the background. Returns its id immediately."""
        cand = CandidateState(plan_id=plan.plan_id, family=plan.family, status="coding")
        cand.kernel_dir = f"candidates/{cand.candidate_id}"
        self.bb.put_candidate(cand)
        sandbox = self.bb.candidate_dir(cand.candidate_id)

        def _work() -> None:
            try:
                coder = self._role("coder", sandbox=sandbox)
                res = coder.run(
                    instructions
                    or (
                        "Implement this plan as a self-contained Kaggle kernel `kernel.py` that "
                        "TRAINS ON THE KAGGLE KERNEL ITSELF (no uploaded checkpoints, no internet, "
                        "only pre-installed packages). Discover /kaggle/input paths at runtime and "
                        "write /kaggle/working/submission.csv. Sanity-check on a subsample first, "
                        "then report."
                    ),
                    context={"plan": plan.to_dict()},
                )
                c = self.bb.get_candidate(cand.candidate_id) or cand
                if not res.ok:
                    c.status = "failed"
                    c.last_error = res.error or "coder produced no report"
                else:
                    c.status = "ready"
                    c.last_error = ""
                self.bb.put_candidate(c)
                self.bb.add_experiment(
                    Experiment(
                        candidate_id=cand.candidate_id,
                        role="coder",
                        description=plan.title,
                        runtime_s=res.elapsed_s,
                        accepted=res.ok,
                        reject_reason=res.error,
                    )
                )
            except Exception:
                c = self.bb.get_candidate(cand.candidate_id) or cand
                c.status = "failed"
                c.last_error = traceback.format_exc()[-1500:]
                self.bb.put_candidate(c)
                self._errors.append(f"candidate {cand.candidate_id} crashed")

        self._futures[cand.candidate_id] = self._pool.submit(_work)
        self.bb.event("candidate_launched", candidate_id=cand.candidate_id, family=plan.family)
        return cand.candidate_id

    def _running(self) -> int:
        return sum(1 for f in self._futures.values() if not f.done())

    def _capacity(self) -> int:
        return max(0, self.cfg.parallel.max_candidates - self._running())

    # ----------------------------------------------------------- actions
    def dispatch(self, action: ManagerAction) -> str:
        """Execute one Manager action. Returns a short observation string."""
        a = action.action
        try:
            if a == "design":
                plans = self.ensure_plans(len(self.bb.get_plans()) + 2)
                return f"designer produced {len(plans)} plans total"

            if a == "code":
                if self._capacity() <= 0:
                    return "no free candidate slot; wait for a running candidate"
                plans = self.bb.get_plans()
                used = {c.plan_id for c in self.bb.get_candidates()}
                fresh = [p for p in plans if p.plan_id not in used]
                if action.target:
                    fresh = [p for p in plans if p.plan_id == action.target] or fresh
                if not fresh:
                    return "no unused plan available; design more first"
                cid = self.launch_candidate(fresh[0], action.instructions)
                return f"launched candidate {cid} on plan {fresh[0].plan_id}"

            if a in ("tune", "verify"):
                cid = action.target
                cand = self.bb.get_candidate(cid) if cid else None
                if cand is None:
                    return f"unknown candidate: {cid!r}"
                role = self._role(
                    "tuner" if a == "tune" else "verifier",
                    **({"sandbox": self.bb.candidate_dir(cid)} if a == "tune" else {}),
                )
                res = role.run(
                    action.instructions or f"{a} candidate {cid}",
                    context={"candidate": cand.to_dict()},
                )
                if a == "verify" and res.report:
                    cand.local_score = res.report.get("local_score", cand.local_score)
                    cand.local_std = res.report.get("local_std", cand.local_std)
                    cand.folds = res.report.get("folds", cand.folds) or cand.folds
                    cand.status = "ready"
                    self.bb.put_candidate(cand)
                    self.bb.put_calibration(fit_calibration(self.bb))
                self.bb.add_experiment(
                    Experiment(
                        candidate_id=cid,
                        role=a,
                        description=action.instructions[:200],
                        local_score=cand.local_score,
                        local_std=cand.local_std,
                        folds=cand.folds,
                        runtime_s=res.elapsed_s,
                        accepted=res.ok,
                        reject_reason=res.error,
                    )
                )
                return f"{a} on {cid}: ok={res.ok} score={cand.local_score}"

            if a == "probe":
                return self._do_probe(action)

            if a == "retire":
                cand = self.bb.get_candidate(action.target)
                if cand is None:
                    return f"unknown candidate: {action.target!r}"
                cand.status = "retired"
                cand.retire_reason = action.reason
                self.bb.put_candidate(cand)
                return f"retired {action.target}"

            if a == "aggregate":
                return self._do_aggregate(action)

            if a == "wait":
                time.sleep(5)
                return f"waited; {self._running()} candidates running"

            if a == "stop":
                self._stop_reason = action.reason or "manager stopped"
                return "stopping"

            if a in ("profile", "report"):
                return f"{a} handled by the pod lifecycle, not the loop"

            return f"unknown action {a!r}"
        except Exception as exc:
            self._errors.append(f"dispatch {a} failed: {exc}")
            return f"[error] {type(exc).__name__}: {exc}"

    def _do_probe(self, action: ManagerAction) -> str:
        role = self._role("prober")
        res = role.run(
            action.instructions
            or "Choose the single most informative probe given what we already know, and "
            "give the CPU kernel code for it.",
            context={"snapshot": self.bb.snapshot()},
        )
        code = (res.report or {}).get("code", "")
        if not code:
            return "prober produced no probe code"
        ok, why = self.broker.may_submit("probe")
        if not ok:
            return f"probe blocked: {why}"
        rec = self.broker.submit(
            code=code,
            candidate_id="probe",
            lane="probe",
            purpose=(res.report or {}).get("hypothesis", "probe"),
            accelerator="cpu",
            local_score=None,
        )
        return f"probe submitted: {rec.sub_id}"

    def _do_aggregate(self, action: ManagerAction) -> str:
        role = self._role("aggregator", sandbox=self.bb.ws)
        ready = [c.to_dict() for c in self.bb.get_candidates() if c.local_score is not None]
        if not ready:
            return "nothing to aggregate: no scored candidate"
        res = role.run(
            action.instructions
            or "Combine the scored candidates into the strongest single kernel that fits the "
            "runtime budget. Do not retrain from scratch.",
            context={"candidates": ready, "calibration": self.bb.get_calibration().to_dict()},
        )
        code = (res.report or {}).get("code", "")
        if not code:
            return "aggregator produced no code"
        ok, why = self.broker.may_submit("final")
        if not ok:
            return f"final submission blocked: {why}"
        rec = self.broker.submit(
            code=code,
            candidate_id="ensemble",
            lane="final",
            purpose=(res.report or {}).get("strategy", "ensemble"),
            accelerator=(res.report or {}).get("accelerator", "cpu"),
            local_score=(res.report or {}).get("local_score"),
        )
        return f"final submission: {rec.sub_id}"

    # -------------------------------------------------------------- run
    def preflight(self) -> str | None:
        """Fail fast on a dead brain. Returns a reason, or None when healthy.

        Without this the pod burns its whole window: every role returns empty,
        the floor gate fires, and the run *looks* successful from the outside
        while having produced nothing but the trivial fallback kernel.
        """
        backends = {self.cfg.model_for(r).backend for r in ("profiler", "manager", "coder")}
        if "claude_code" in backends:
            from .runners import check_auth

            ok, detail = check_auth()
            if not ok:
                return f"claude_code backend unusable: {detail}"
        return None

    def run(self) -> PodResult:
        self.bb.event("pod_start", slug=self.cfg.slug, config=self.cfg.to_dict())
        problem = self.preflight()
        if problem:
            self.bb.event("preflight_failed", reason=problem)
            self._errors.append(problem)
            return PodResult(
                slug=self.cfg.slug,
                ok=False,
                stop_reason=f"preflight failed: {problem}",
                errors=self._errors,
            )
        try:
            self.ensure_task_card()
            self.ensure_plans(min(self.cfg.parallel.max_candidates + 1, 6))

            # Seed the parallel candidates up front: exploration breadth is the
            # one thing we can buy with the unrestricted agent-side compute.
            for plan in self.bb.get_plans()[: self.cfg.parallel.max_candidates]:
                if self._capacity() > 0:
                    self.launch_candidate(plan)

            manager = self._role("manager")
            observation = "pod started; candidates launching"

            while True:
                stop = self.enforce_gates()
                if stop:
                    self._stop_reason = stop
                    break
                if self._stop_reason:
                    break

                self.broker.poll_scores()
                snapshot = self.bb.snapshot()
                snapshot["budget"] = self.budget.status()
                snapshot["minutes_elapsed"] = round(self._minutes(), 1)
                snapshot["in_freeze"] = self.in_freeze()
                snapshot["has_floor_submission"] = self._has_floor()
                snapshot["running_candidates"] = self._running()
                snapshot["free_slots"] = self._capacity()
                snapshot["last_observation"] = observation

                res = manager.run(
                    "Choose the single next action. Emit exactly one action object.",
                    context=snapshot,
                )
                action = (
                    manager.parse(res)
                    if hasattr(manager, "parse")
                    else ManagerAction.from_dict(res.report or {})
                )
                self.bb.event(
                    "manager_action",
                    action=action.action,
                    target=action.target,
                    reason=action.reason[:300],
                )
                observation = self.dispatch(action)

            # Consolidation happens whether or not the Manager asked for it.
            if not self._has_floor():
                self._force_floor_submission()
            self._finalize()
        except Exception:
            self._errors.append(traceback.format_exc()[-2000:])
            self._stop_reason = self._stop_reason or "pod crashed"
        finally:
            self._pool.shutdown(wait=False, cancel_futures=True)

        best = self.bb.best_candidate()
        subs = self.bb.get_submissions()
        return PodResult(
            slug=self.cfg.slug,
            ok=not self._errors,
            best_local=best.local_score if best else None,
            best_lb=max((s.lb_score for s in subs if s.lb_score is not None), default=None),
            n_submissions=len(subs),
            n_candidates=len(self.bb.get_candidates()),
            stop_reason=self._stop_reason,
            report_path=str(self.bb.ws / "REPORT.md"),
            errors=self._errors,
        )

    def _finalize(self) -> None:
        """Pick the final entry and write the required technical report."""
        try:
            self.bb.put_calibration(fit_calibration(self.bb))
            if hasattr(self.broker, "final_selection"):
                chosen = self.broker.final_selection()
                if chosen:
                    self.bb.event("final_selection", sub_id=chosen.sub_id, lb=chosen.lb_score)
        except Exception as exc:
            self._errors.append(f"final selection failed: {exc}")

        try:
            from .report import write_report

            card = self.bb.get_task_card() or TaskCard(self.cfg.slug)
            write_report(self.bb, self.budget, self.quota, self.cfg, card)
        except Exception as exc:
            self._errors.append(f"report generation failed: {exc}")
