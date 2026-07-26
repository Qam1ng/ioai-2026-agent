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

import threading
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
        self._shutdown = threading.Event()
        self._watchdog: threading.Thread | None = None
        # Verification is a pipeline, not a Manager decision: candidates flow
        # ready -> verified -> (milestone) on their own. One worker, because a
        # verification runs fold training and two at once would thrash the CPU.
        self._verify_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verify")
        self._verify_inflight: set[str] = set()
        self._verify_lock = threading.Lock()
        self._harness_lock = threading.Lock()
        self._frozen = False
        self._aggregate_started = False
        self._aggregate_thread: threading.Thread | None = None
        self._cand_started: dict[str, float] = {}
        self._action_log: list[dict] = []

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

        # Floor, freeze and freeze-time aggregation are owned by the gates
        # watchdog thread; the loop only honours the flags.

        last_call = self.cfg.deadline_min - self.cfg.gates.final_submit_buffer_min
        if self._minutes() >= last_call:
            return "final submission buffer reached"
        return None

    def _floor_healthy(self) -> bool:
        """A floor exists and has not errored. `scored`/`queued`/`running` count;
        an errored floor is no floor at all — that distinction is the fix for
        the rehearsal's silent failure mode, where a fallback kernel that
        cannot work on this competition sat in the log marked 'scored'."""
        return any(
            s.lane == "floor" and s.status != "error" for s in self.bb.get_submissions()
        )

    def _start_gates_watchdog(self) -> None:
        """Every deterministic obligation, on a clock the main flow cannot stall.

        The rehearsal proved gates must not depend on the Manager loop being
        alive: a synchronous dispatch froze the loop for 15 minutes and the
        freeze gate was only stamped when the loop happened to return. This
        thread owns: the floor deadline, floor retry after an errored
        submission, the freeze flag, and freeze-time aggregation.
        """

        def _watch() -> None:
            while not self._shutdown.is_set():
                m = self._minutes()
                try:
                    # -- floor: insurance by the deadline, and re-insurance on error
                    if not self._floor_healthy():
                        if self._floor_done or m >= self.cfg.gates.floor_submit_min:
                            self.bb.event(
                                "gate", gate="floor_watchdog", minutes=round(m, 1)
                            )
                            self._force_floor_submission()
                    else:
                        self._floor_done = True

                    # -- freeze: flip once, then make consolidation actually happen
                    if m >= self.cfg.gates.freeze_min and not self._frozen:
                        self._frozen = True
                        self.bb.event("gate", gate="freeze", minutes=round(m, 1))
                    if self._frozen and not self._aggregate_started:
                        scored = [
                            c
                            for c in self.bb.get_candidates()
                            if c.local_score is not None
                        ]
                        has_final = any(
                            s.lane == "final" for s in self.bb.get_submissions()
                        )
                        if scored and not has_final:
                            self._start_aggregate("freeze watchdog")
                except Exception as exc:  # the watchdog itself must never die
                    self._errors.append(f"gates watchdog: {exc}")
                self._shutdown.wait(10)

        self._watchdog = threading.Thread(target=_watch, name="gates-watchdog", daemon=True)
        self._watchdog.start()

    def _force_floor_submission(self) -> None:
        """Submit the best thing that exists, right now.

        Preference order matters and was measured to matter: any READY
        candidate with a kernel on disk (floor family first) beats the generic
        fallback — the old code asked ``best_candidate()``, which only returns
        *scored* candidates, so a ready-but-unscored floor kernel sat unused
        while a sample-echo fallback (which cannot work on a competition that
        ships no sample file) was submitted instead.
        """
        try:
            ready = [
                c
                for c in self.bb.get_candidates()
                if c.status == "ready"
                and c.kernel_dir
                and (self.bb.ws / c.kernel_dir / "kernel.py").exists()
            ]
            # floor family first, then scored ones, then whatever is ready
            ready.sort(key=lambda c: (c.family != "floor", c.local_score is None))
            for cand in ready:
                code = self.bb.ws / cand.kernel_dir / "kernel.py"
                ok, why = self.broker.may_submit("floor")
                if not ok:
                    self.bb.event("gate_error", gate="floor", error=why)
                    return
                self.broker.submit(
                    code=code.read_text(),
                    candidate_id=cand.candidate_id,
                    lane="floor",
                    purpose=f"floor submission ({cand.family} candidate)",
                    accelerator="cpu",
                    local_score=cand.local_score,
                )
                self._floor_done = True
                return
            ok, why = self.broker.may_submit("floor")
            if not ok:
                self.bb.event("gate_error", gate="floor", error=why)
                return
            self.broker.submit(
                code=self._fallback_kernel(),
                candidate_id="fallback",
                lane="floor",
                purpose="forced floor submission (trivial fallback; no candidate ready)",
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
        # NB the exact method name matters: an earlier version probed for a
        # generic `parse` that no role implements, so every profiler report was
        # silently discarded and two rehearsals ran on an empty task card.
        try:
            card = role.parse_card(res.report, slug=self.cfg.slug)
        except (ValueError, AttributeError) as exc:
            card = TaskCard(slug=self.cfg.slug, routing_confidence=0.0)
            self._errors.append(f"profiler card unusable ({exc}); running with an empty one")
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
        new = role.parse_plans(res.report) if res.report else []
        if not new and res.report:
            self._errors.append("designer report parsed to zero plans")
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
                # Ready flows straight into verification. In the rehearsal the
                # floor candidate sat ready and unmeasured for 24 minutes
                # because verification waited on a Manager decision.
                if c.status == "ready":
                    self._enqueue_verify(c.candidate_id)
            except Exception:
                c = self.bb.get_candidate(cand.candidate_id) or cand
                c.status = "failed"
                c.last_error = traceback.format_exc()[-1500:]
                self.bb.put_candidate(c)
                self._errors.append(f"candidate {cand.candidate_id} crashed")

        self._futures[cand.candidate_id] = self._pool.submit(_work)
        self._cand_started[cand.candidate_id] = time.time()
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

            if a == "tune":
                cid = action.target
                cand = self.bb.get_candidate(cid) if cid else None
                if cand is None:
                    return f"unknown candidate: {cid!r}"
                role = self._role("tuner", sandbox=self.bb.candidate_dir(cid))
                res = role.run(
                    action.instructions or f"tune candidate {cid}",
                    context={"candidate": cand.to_dict()},
                )
                self.bb.add_experiment(
                    Experiment(
                        candidate_id=cid,
                        role="tune",
                        description=action.instructions[:200],
                        runtime_s=res.elapsed_s,
                        accepted=res.ok,
                        reject_reason=res.error,
                    )
                )
                return f"tune on {cid}: ok={res.ok}"

            if a == "verify":
                cid = action.target
                if not cid or self.bb.get_candidate(cid) is None:
                    return f"unknown candidate: {cid!r}"
                queued = self._enqueue_verify(cid, requeue=True)
                return (
                    f"re-verification of {cid} queued (runs on the verify worker; "
                    f"the loop continues)"
                    if queued
                    else f"{cid} is already being verified"
                )

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
                started = self._start_aggregate("manager")
                return (
                    "aggregation started in the background"
                    if started
                    else "aggregation already ran or is running"
                )

            if a == "wait":
                # Block until the world changes or ~60s passes. The rehearsal
                # Manager burned eight Opus calls polling a snapshot that could
                # not have changed; a wait that actually waits makes each
                # Manager step worth reading.
                before = self._progress_key(self.bb.snapshot())
                deadline = time.time() + 60
                while time.time() < deadline and not self._shutdown.is_set():
                    time.sleep(5)
                    if self._progress_key(self.bb.snapshot()) != before:
                        return f"state changed while waiting; {self._running()} running"
                    if self._stop_reason:
                        break
                return f"waited 60s; {self._running()} candidates running, no state change"

            if a == "stop":
                self._stop_reason = action.reason or "manager stopped"
                return "stopping"

            if a == "profile":
                # Must actually re-profile. Returning a polite no-op here made
                # the Manager's most-chosen action change nothing, so it asked
                # again, forever: 11 profile requests, 0 profiler runs.
                self.bb.task_card_path.unlink(missing_ok=True)
                card = self.ensure_task_card()
                return (
                    f"re-profiled: task_type={card.task_type} metric={card.metric_name!r} "
                    f"constraints={len(card.constraints)}"
                )

            if a == "report":
                self._finalize()
                return "report written"

            return f"unknown action {a!r}"
        except Exception as exc:
            self._errors.append(f"dispatch {a} failed: {exc}")
            return f"[error] {type(exc).__name__}: {exc}"

    #: How many times an action may leave the world unchanged before the pod
    #: stops trusting the Manager's routing and forces the next step itself.
    STALL_LIMIT = 3

    def _progress_key(self, snap: dict) -> tuple:
        """A coarse fingerprint of 'did anything actually happen'."""
        return (
            snap.get("task_type"),
            snap.get("n_plans"),
            len(snap.get("candidates") or []),
            snap.get("n_submissions"),
            snap.get("best_local"),
        )

    def _check_progress(self, action: ManagerAction, snap: dict, observation: str) -> str:
        """Break livelocks: a Manager whose actions change nothing must be overruled.

        Observed on the radar rehearsal: 23 Manager steps alternating profile
        and design, 10 Designer runs that each returned zero plans, 927k input
        tokens spent, and no candidate ever created. The Manager was reasoning
        correctly about a world that never moved; only the pod can see that.
        """
        after = self._progress_key(self.bb.snapshot())
        before = self._progress_key(snap)
        if after != before:
            self._stall_count = 0
            return observation

        # Waiting while candidates are genuinely running is patience, not a
        # stall: the world *is* moving, just not in the snapshot yet. Observed:
        # three legitimate waits during parallel coding tripped the breaker and
        # spent the floor submission four minutes early.
        if action.action == "wait" and self._running() > 0:
            self._stall_count = 0
            return observation

        self._stall_count = getattr(self, "_stall_count", 0) + 1
        if self._stall_count < self.STALL_LIMIT:
            return (
                f"{observation}\nWARNING: that action changed nothing "
                f"({self._stall_count}/{self.STALL_LIMIT} consecutive). Pick a different action."
            )

        self.bb.event("stall_break", action=action.action, count=self._stall_count)
        self._stall_count = 0
        forced = self._force_progress()
        return f"{observation}\nSTALL BROKEN BY POD: {forced}"

    def _force_progress(self) -> str:
        """Do the most useful thing that is guaranteed to move state."""
        if not self._has_floor():
            self._force_floor_submission()
            return "forced the floor submission"
        if not self.bb.get_plans():
            # The Designer keeps failing; seed a floor plan so coding can start
            # at all. A weak plan that produces a candidate beats a perfect plan
            # that never gets written.
            plan = PlanCard(
                title="minimal valid submission",
                family="floor",
                architecture="constant or trivial prediction in the required output format",
                data_strategy="none; read only what the output format needs",
                validation_strategy="format validation only",
                expected_runtime_min=2.0,
                accelerator="cpu",
                rationale="pod-seeded after the designer failed to produce any plan",
            )
            self.bb.add_plan(plan)
            return f"seeded fallback plan {plan.plan_id}"
        used = {c.plan_id for c in self.bb.get_candidates()}
        fresh = [p for p in self.bb.get_plans() if p.plan_id not in used]
        if fresh and self._capacity() > 0:
            cid = self.launch_candidate(fresh[0])
            return f"launched candidate {cid} without waiting for the manager"
        return "no forced action available; waiting on running candidates"

    # -------------------------------------------------------- verification
    def _run_eval_harness(self, candidate_id: str, timeout_s: float | None = None) -> dict | None:
        """Score a candidate with the deterministic harness, if one exists.

        An LLM estimates; a harness measures. Once the Verifier has built
        ``eval/run_eval.py``, every re-verification is a subprocess — free,
        reproducible, same folds every time — which is what makes the
        significance gate's comparisons between candidates meaningful at all.
        Returns the parsed result dict, or None when no harness exists yet.
        """
        import subprocess
        import sys as _sys

        script = self.bb.ws / "eval" / "run_eval.py"
        if not script.exists():
            return None
        try:
            proc = subprocess.run(
                [_sys.executable, str(script), "--candidate", f"candidates/{candidate_id}"],
                cwd=str(self.bb.ws),
                capture_output=True,
                text=True,
                timeout=timeout_s or self.cfg.parallel.tuner_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"problems": [f"eval harness timed out"], "error": "timeout"}
        import json as _json

        for line in reversed((proc.stdout or "").strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    out = _json.loads(line)
                    if proc.returncode != 0:
                        out.setdefault("problems", []).append(
                            f"harness exit={proc.returncode}: {(proc.stderr or '')[-300:]}"
                        )
                    return out
                except _json.JSONDecodeError:
                    break
        return {
            "problems": [f"harness produced no JSON (exit={proc.returncode})"],
            "error": (proc.stderr or proc.stdout or "")[-400:],
        }

    def _enqueue_verify(self, candidate_id: str, requeue: bool = False) -> bool:
        """Queue a candidate for verification. Returns False if already queued."""
        with self._verify_lock:
            if candidate_id in self._verify_inflight and not requeue:
                return False
            self._verify_inflight.add(candidate_id)
        self.bb.event("verify_queued", candidate_id=candidate_id)
        self._verify_pool.submit(self._verify_worker, candidate_id)
        return True

    def _verify_worker(self, cid: str) -> None:
        """Measure one candidate; runs on the verify thread, never the Manager's.

        In the rehearsal a synchronous verify froze the Manager loop for 15.2
        minutes, during which two candidates finished unnoticed and the freeze
        gate passed with nobody watching. Verification is mechanical once the
        harness exists, so it lives in a pipeline the Manager merely observes.
        """
        try:
            cand = self.bb.get_candidate(cid)
            if cand is None:
                return
            result = self._run_eval_harness(cid)
            built = False
            elapsed = 0.0
            if result is None:
                with self._harness_lock:  # the harness is built exactly once
                    result = self._run_eval_harness(cid)
                    if result is None:
                        role = self._role("verifier")
                        res = role.run(
                            "No eval/ harness exists yet. Build it (metric.py with a passing "
                            "self-test, run_eval.py honouring the black-box contract), then "
                            f"verify candidate {cid} by running it. Report red-team findings.",
                            context={
                                "candidate": cand.to_dict(),
                                "task_card": (
                                    self.bb.get_task_card() or TaskCard(self.cfg.slug)
                                ).to_dict(),
                            },
                        )
                        built, elapsed = res.ok, res.elapsed_s
                        # The subprocess outranks the transcript.
                        result = self._run_eval_harness(cid)
                        if result is None and res.report:
                            result = res.report
                            result.setdefault("problems", []).append(
                                "verifier emitted numbers but no eval/run_eval.py harness"
                            )
            if not result:
                self.bb.event("verify_failed", candidate_id=cid, reason="no harness, no report")
                return

            score = result.get("local_score")
            if score is not None:
                cand.local_score = float(score)
                cand.local_std = float(result.get("local_std") or 0.0)
                cand.folds = list(result.get("folds") or [])
                cand.status = "ready"
                self.bb.put_candidate(cand)
                self.bb.put_calibration(fit_calibration(self.bb))
            self.bb.add_experiment(
                Experiment(
                    candidate_id=cid,
                    role="verify",
                    description=("harness" + ("+build" if built else ""))
                    if score is not None
                    else "verify failed",
                    local_score=cand.local_score,
                    local_std=cand.local_std,
                    folds=cand.folds,
                    runtime_s=elapsed,
                    accepted=score is not None,
                    reject_reason="; ".join(map(str, result.get("problems") or []))[:400],
                )
            )
            self.bb.event(
                "verified",
                candidate_id=cid,
                local_score=cand.local_score,
                problems=(result.get("problems") or [])[:3],
            )
            # A verified score that clears the gate goes to the leaderboard
            # without waiting to be asked: leaderboard points are what buy the
            # calibration every later decision depends on.
            if score is not None:
                self._try_milestone(cand)
        except Exception:
            self._errors.append(f"verify worker crashed for {cid}: {traceback.format_exc()[-600:]}")
            self.bb.event("verify_failed", candidate_id=cid, reason="worker crash")
        finally:
            with self._verify_lock:
                self._verify_inflight.discard(cid)

    def _candidate_accelerator(self, cand: CandidateState) -> str:
        """Read the accelerator the coder declared in its kernel metadata."""
        import json as _json

        meta = self.bb.ws / cand.kernel_dir / "kernel-metadata.json"
        try:
            d = _json.loads(meta.read_text())
        except (OSError, ValueError):
            return "cpu"
        if str(d.get("enable_gpu", "false")).lower() not in ("true", "1"):
            return "cpu"
        shape = str(d.get("machine_shape", "") or "")
        return "p100" if "P100" in shape else "t4"

    def _try_milestone(self, cand: CandidateState) -> None:
        """Submit a verified candidate when the significance gate clears.

        This closes the rehearsal's largest gap: a verified 0.1378 that never
        reached the leaderboard because no code path ever asked the broker for
        the milestone lane.
        """
        kernel = self.bb.ws / cand.kernel_dir / "kernel.py"
        if not kernel.exists():
            return
        accel = self._candidate_accelerator(cand)
        ok, why = self.broker.may_submit(
            "milestone", local_score=cand.local_score, candidate_id=cand.candidate_id,
            accelerator=accel,
        )
        self.bb.event(
            "milestone_gate", candidate_id=cand.candidate_id, allowed=ok, why=why[:200]
        )
        if not ok:
            return
        try:
            rec = self.broker.submit(
                code=kernel.read_text(),
                candidate_id=cand.candidate_id,
                lane="milestone",
                purpose=f"verified {cand.family} local={cand.local_score}",
                accelerator=accel,
                local_score=cand.local_score,
            )
            self.bb.event("milestone_submitted", candidate_id=cand.candidate_id, sub_id=rec.sub_id)
        except Exception as exc:
            self._errors.append(f"milestone submit failed for {cand.candidate_id}: {exc}")

    #: Prompt-schema kinds -> swarm.submit.probes registry names. The Prober
    #: PLANS probes (kind + hypothesis + decision rule); the kernel code always
    #: comes from the audited probe library, never from the model's transcript.
    PROBE_KINDS = {
        "constant": "constant_baseline",
        "constant_baseline": "constant_baseline",
        "granularity": "score_granularity",
        "score_granularity": "score_granularity",
        "split_shift": "split_shift",
        "noise_floor": "noise_floor",
        "calibration": "calibration",
    }

    def _do_probe(self, action: ManagerAction) -> str:
        from .submit.probes import get_probe

        role = self._role("prober")
        res = role.run(
            action.instructions or "",
            context={"snapshot": self.bb.snapshot()},
        )
        probes = role.parse_probes(res.report) if res.report else []
        if not probes:
            return "prober planned no probes (or its report failed to parse)"

        submitted, skipped = [], []
        for plan in probes[:2]:  # at most two per manager step; they queue on Kaggle
            kind = self.PROBE_KINDS.get(str(plan.get("probe_kind", "")).lower())
            if kind is None:
                skipped.append(f"unknown kind {plan.get('probe_kind')!r}")
                continue
            ok, why = self.broker.may_submit("probe", accelerator="cpu")
            if not ok:
                skipped.append(f"blocked: {why}")
                break
            try:
                code = get_probe(kind).build_code(self.bb.get_task_card())
            except Exception as exc:
                skipped.append(f"{kind}: build_code failed: {exc}")
                continue
            rec = self.broker.submit(
                code=code,
                candidate_id=f"probe-{kind}",
                lane="probe",
                purpose=f"{plan.get('hypothesis', kind)} | rule: {plan.get('decision_rule', '')}"[:200],
                accelerator="cpu",
                local_score=None,
            )
            submitted.append(f"{kind}->{rec.sub_id}")
        out = f"probes submitted: {submitted}" if submitted else "no probe submitted"
        if skipped:
            out += f" (skipped: {skipped[:3]})"
        return out

    def _start_aggregate(self, trigger: str) -> bool:
        """Kick off final consolidation exactly once, off the Manager's thread.

        Fired by the freeze watchdog (the normal path — the rehearsal showed
        aggregation never happens if it waits for a Manager that may be busy)
        or by an explicit Manager action.
        """
        if self._aggregate_started:
            return False
        self._aggregate_started = True
        self.bb.event("aggregate_start", trigger=trigger)
        self._aggregate_thread = threading.Thread(
            target=self._run_aggregate, name="aggregate", daemon=True
        )
        self._aggregate_thread.start()
        return True

    def _run_aggregate(self) -> None:
        try:
            scored = [c for c in self.bb.get_candidates() if c.local_score is not None]
            if len(scored) >= 2:
                role = self._role("aggregator", sandbox=self.bb.ws)
                res = role.run(
                    "Combine the scored candidates into the strongest single kernel that "
                    "fits the runtime budget. Do not retrain from scratch.",
                    context={
                        "candidates": [c.to_dict() for c in scored],
                        "calibration": self.bb.get_calibration().to_dict(),
                    },
                )
                report = res.report or {}
                code = report.get("code", "")
                if code:
                    ok, why = self.broker.may_submit("final")
                    if ok:
                        rec = self.broker.submit(
                            code=code,
                            candidate_id="ensemble",
                            lane="final",
                            purpose=report.get("strategy", "ensemble"),
                            accelerator=report.get("accelerator", "cpu"),
                            # The aggregator schema names its OOF score "oof_score".
                            local_score=report.get("oof_score", report.get("local_score")),
                        )
                        self.bb.event("final_submitted", sub_id=rec.sub_id, via="ensemble")
                        return
                    self.bb.event("aggregate_blocked", why=why[:200])
            # One (or zero ensemble-worthy) scored candidates: the best single
            # verified kernel IS the final. Better a proven single model than
            # no final entry at all.
            best = self.bb.best_candidate()
            if best is None:
                self.bb.event("aggregate_skipped", why="no scored candidate")
                return
            kernel = self.bb.ws / best.kernel_dir / "kernel.py"
            if not kernel.exists():
                self.bb.event("aggregate_skipped", why=f"{best.candidate_id} has no kernel")
                return
            ok, why = self.broker.may_submit("final")
            if not ok:
                self.bb.event("aggregate_blocked", why=why[:200])
                return
            rec = self.broker.submit(
                code=kernel.read_text(),
                candidate_id=best.candidate_id,
                lane="final",
                purpose=f"best single verified candidate ({best.family})",
                accelerator=self._candidate_accelerator(best),
                local_score=best.local_score,
            )
            self.bb.event("final_submitted", sub_id=rec.sub_id, via="best_single")
        except Exception:
            self._errors.append(f"aggregate crashed: {traceback.format_exc()[-600:]}")

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
        self._start_gates_watchdog()
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
                snapshot["in_freeze"] = self._frozen
                snapshot["has_floor_submission"] = self._floor_healthy()
                snapshot["running_candidates"] = self._running()
                snapshot["free_slots"] = self._capacity()
                snapshot["last_observation"] = observation
                # Pipeline visibility: what the pod does on its own, so the
                # Manager stops routing what no longer needs routing.
                with self._verify_lock:
                    snapshot["verify_queue"] = sorted(self._verify_inflight)
                snapshot["aggregate_started"] = self._aggregate_started
                now = time.time()
                snapshot["candidate_runtimes_min"] = {
                    cid: round((now - t0) / 60, 1)
                    for cid, t0 in self._cand_started.items()
                    if cid in self._futures and not self._futures[cid].done()
                }
                # Recent action history with outcomes: lets the Manager see a
                # dead path as a *pattern*, not a one-off observation.
                snapshot["recent_actions"] = self._action_log[-8:]

                res = manager.run(
                    "Choose the single next action. Emit exactly one action object.",
                    context=snapshot,
                )
                action = manager.parse_action(res.report or {})
                self.bb.event(
                    "manager_action",
                    action=action.action,
                    target=action.target,
                    reason=action.reason[:300],
                )
                observation = self.dispatch(action)
                observation = self._check_progress(action, snapshot, observation)
                self._action_log.append(
                    {"action": action.action, "target": action.target,
                     "observation": observation[:160]}
                )

            # Consolidation happens whether or not the Manager asked for it.
            if not self._has_floor():
                self._force_floor_submission()
            self._finalize()
        except Exception:
            self._errors.append(traceback.format_exc()[-2000:])
            self._stop_reason = self._stop_reason or "pod crashed"
        finally:
            self._shutdown.set()
            if self._watchdog is not None:
                self._watchdog.join(timeout=5)
            if self._aggregate_thread is not None:
                self._aggregate_thread.join(timeout=120)
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._verify_pool.shutdown(wait=False, cancel_futures=True)

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
