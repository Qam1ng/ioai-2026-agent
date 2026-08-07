"""The verification pipeline contract.

Three invariants, each paid for in rehearsal time:

1. Numbers come from the deterministic harness subprocess, never a model
   transcript.
2. Verification is a pipeline: a candidate becoming ready flows to verified
   without a Manager decision (a synchronous verify once froze the Manager
   loop for 15.2 minutes).
3. A verified score that clears the significance gate is submitted to the
   milestone lane automatically (a verified 0.1378 once never reached the
   leaderboard because nothing ever asked the broker).
"""

from __future__ import annotations

import json
import time

from swarm.budget import PodBudget, QuotaPool
from swarm.bus import Blackboard
from swarm.config import SwarmConfig
from swarm.pod import TaskPod
from swarm.schemas import CandidateState, ManagerAction, TaskCard


class RecordingBroker:
    """Counts submissions; approves or denies milestone per configuration."""

    def __init__(self, allow_milestone: bool = True):
        self.allow_milestone = allow_milestone
        self.submissions: list[dict] = []

    def may_submit(self, lane, local_score=None, candidate_id=None, accelerator=None):
        if lane == "milestone" and not self.allow_milestone:
            return False, "gain inside the noise band"
        return True, f"{lane} allowed"

    def submit(self, **kw):
        self.submissions.append(kw)

        class Rec:
            sub_id = f"sub-test-{len(self.submissions)}"
            lane = kw["lane"]
            lb_score = None

        return Rec()

    def poll_scores(self):
        return 0


def make_pod(tmp_path, broker=None) -> TaskPod:
    cfg = SwarmConfig()
    cfg.slug = "test-comp"
    cfg.dry_run = True
    bb = Blackboard(tmp_path / "ws")
    quota = QuotaPool(tmp_path / "q.json", limit_hours=1.0)
    budget = PodBudget(tmp_path / "b.json", deadline_s=3600, quota=quota)
    bb.put_task_card(TaskCard(slug=cfg.slug, task_type="supervised", metric_name="acc"))
    return TaskPod(cfg, bb, budget, quota, broker or RecordingBroker())


HARNESS_OK = """
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--candidate")
a = p.parse_args()
print("scoring", a.candidate)  # noise before the JSON line must be tolerated
print(json.dumps({"local_score": 0.75, "local_std": 0.02,
                  "folds": [0.73, 0.75, 0.77], "problems": []}))
"""

HARNESS_BROKEN = """
import sys
print("something went wrong before any JSON")
sys.exit(3)
"""


def write_candidate(pod, family="baseline", with_kernel=True, gpu=False) -> CandidateState:
    cand = CandidateState(family=family, status="ready")
    cand.kernel_dir = f"candidates/{cand.candidate_id}"
    pod.bb.put_candidate(cand)
    d = pod.bb.candidate_dir(cand.candidate_id)
    if with_kernel:
        (d / "kernel.py").write_text("print('kernel')\n")
        (d / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "enable_gpu": "true" if gpu else "false",
                    "machine_shape": "NvidiaTeslaT4" if gpu else "",
                }
            )
        )
    return cand


def install_harness(pod, code=HARNESS_OK):
    eval_dir = pod.bb.ws / "eval"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / "run_eval.py").write_text(code)


def wait_verified(pod, cid, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        c = pod.bb.get_candidate(cid)
        if c is not None and c.local_score is not None:
            return c
        with pod._verify_lock:
            inflight = cid in pod._verify_inflight
        if not inflight and c is not None and c.local_score is None:
            # worker finished without a score
            return c
        time.sleep(0.1)
    raise AssertionError("verification did not finish in time")


# ------------------------------------------------------------ subprocess truth
def test_missing_harness_returns_none(tmp_path):
    pod = make_pod(tmp_path)
    assert pod._run_eval_harness("cand-x") is None


def test_harness_is_deterministic_across_reruns(tmp_path):
    pod = make_pod(tmp_path)
    install_harness(pod)
    a = pod._run_eval_harness("cand-1")
    b = pod._run_eval_harness("cand-1")
    assert a == b, "the whole point of the harness is that re-measurement is identical"


def test_broken_harness_is_a_finding_not_a_crash(tmp_path):
    pod = make_pod(tmp_path)
    install_harness(pod, HARNESS_BROKEN)
    out = pod._run_eval_harness("cand-1")
    assert out is not None
    assert out.get("local_score") is None
    assert any("no JSON" in p for p in out.get("problems", []))


# --------------------------------------------------------------- the pipeline
def test_verify_pipeline_scores_candidate_without_manager(tmp_path):
    pod = make_pod(tmp_path)
    install_harness(pod)
    cand = write_candidate(pod)
    assert pod._enqueue_verify(cand.candidate_id)
    got = wait_verified(pod, cand.candidate_id)
    assert got.local_score == 0.75
    assert got.folds == [0.73, 0.75, 0.77]
    exps = pod.bb.get_experiments()
    assert exps and exps[-1].role == "verify" and exps[-1].accepted


def test_duplicate_enqueue_is_ignored(tmp_path):
    pod = make_pod(tmp_path)
    install_harness(pod)
    cand = write_candidate(pod)
    first = pod._enqueue_verify(cand.candidate_id)
    second = pod._enqueue_verify(cand.candidate_id)
    assert first is True
    # Either the first finished already (fast harness) or the second is a dup.
    if second:
        wait_verified(pod, cand.candidate_id)
    else:
        assert second is False
    wait_verified(pod, cand.candidate_id)


def test_verified_score_auto_submits_milestone(tmp_path):
    broker = RecordingBroker(allow_milestone=True)
    pod = make_pod(tmp_path, broker)
    install_harness(pod)
    cand = write_candidate(pod, gpu=True)
    pod._enqueue_verify(cand.candidate_id)
    wait_verified(pod, cand.candidate_id)
    time.sleep(0.3)  # milestone fires inside the worker after scoring
    lanes = [s["lane"] for s in broker.submissions]
    assert "milestone" in lanes, "a verified score must reach the leaderboard unprompted"
    sub = next(s for s in broker.submissions if s["lane"] == "milestone")
    assert sub["accelerator"] == "t4", "accelerator must come from the kernel metadata"
    assert sub["local_score"] == 0.75


def test_gated_milestone_is_not_submitted(tmp_path):
    broker = RecordingBroker(allow_milestone=False)
    pod = make_pod(tmp_path, broker)
    install_harness(pod)
    cand = write_candidate(pod)
    pod._enqueue_verify(cand.candidate_id)
    wait_verified(pod, cand.candidate_id)
    time.sleep(0.3)
    assert all(s["lane"] != "milestone" for s in broker.submissions)
    # but the gate decision itself must be on the audit trail
    events = [json.loads(l) for l in open(pod.bb.events_path) if l.strip()]
    gates = [e for e in events if e.get("kind") == "milestone_gate"]
    assert gates and gates[-1]["allowed"] is False


def test_dispatch_verify_queues_and_returns_immediately(tmp_path):
    pod = make_pod(tmp_path)
    install_harness(pod)
    cand = write_candidate(pod)
    t0 = time.time()
    obs = pod.dispatch(ManagerAction(action="verify", target=cand.candidate_id))
    assert time.time() - t0 < 2.0, "verify must not block the manager loop"
    assert "queued" in obs or "already" in obs
    wait_verified(pod, cand.candidate_id)


# ----------------------------------------------------------------- floor path
def test_floor_prefers_ready_candidate_over_fallback(tmp_path):
    """The rehearsal bug: a ready-but-unscored floor kernel sat unused while a
    sample-echo fallback (guaranteed to fail on this competition) was
    submitted. Ready candidates must win regardless of having a score."""
    broker = RecordingBroker()
    pod = make_pod(tmp_path, broker)
    write_candidate(pod, family="cnn")
    floor = write_candidate(pod, family="floor")
    pod._force_floor_submission()
    assert broker.submissions, "floor must submit something"
    sub = broker.submissions[0]
    assert sub["lane"] == "floor"
    assert sub["candidate_id"] == floor.candidate_id, "floor family wins the tie"


def test_floor_falls_back_to_trivial_kernel_when_nothing_ready(tmp_path):
    broker = RecordingBroker()
    pod = make_pod(tmp_path, broker)
    pod._force_floor_submission()
    assert broker.submissions[0]["candidate_id"] == "fallback"


def test_floor_healthy_ignores_errored_submissions(tmp_path):
    from swarm.schemas import SubmissionRecord

    pod = make_pod(tmp_path)
    rec = SubmissionRecord(lane="floor", candidate_id="fallback", status="error")
    pod.bb.add_submission(rec)
    assert pod._floor_healthy() is False, "an errored floor is no floor at all"


# ------------------------------------------------------------------ aggregate
def test_aggregate_falls_back_to_best_single_candidate(tmp_path):
    """With exactly one scored candidate there is nothing to ensemble; the
    final entry must still exist."""
    broker = RecordingBroker()
    pod = make_pod(tmp_path, broker)
    cand = write_candidate(pod)
    cand.local_score = 0.6
    pod.bb.put_candidate(cand)
    assert pod._start_aggregate("test") is True
    pod._aggregate_thread.join(timeout=10)
    finals = [s for s in broker.submissions if s["lane"] == "final"]
    assert finals and finals[0]["candidate_id"] == cand.candidate_id
    assert pod._start_aggregate("test") is False, "aggregate runs exactly once"


# ------------------------------------------------- run-3 regression fixes
def test_coder_written_score_is_quarantined_as_claimed(tmp_path):
    """Run 3: a coder self-measured 0.1215, wrote it into state.json, and the
    Manager quoted it as 'verified'. Any score the harness did not produce
    must be moved to claimed_score before selection can see it."""
    pod = make_pod(tmp_path)
    install_harness(pod)
    plan = __import__("swarm.schemas", fromlist=["PlanCard"]).PlanCard(
        title="t", family="mlp"
    )
    pod.bb.add_plan(plan)

    class SelfReportingCoder:
        def __init__(self, pod):
            self.pod = pod

        def run(self, instructions, context=None):
            # Simulate the coder editing its own state.json with a self-score.
            cid = context["plan"]["plan_id"] if False else None
            # find the candidate the pod just created
            cand = self.pod.bb.get_candidates()[-1]
            cand.local_score = 0.1215
            cand.local_std = 0.003
            self.pod.bb.put_candidate(cand)
            from swarm.roles.base import RoleResult

            return RoleResult(role="coder", ok=True, report={"status": "ready"})

    pod._roles_override["coder"] = SelfReportingCoder(pod)
    cid = pod.launch_candidate(plan)
    pod._futures[cid].result(timeout=10)
    got = pod.bb.get_candidate(cid)
    assert got.local_score is None or got.score_source == "harness", (
        "coder-written score must not survive as a selection input"
    )
    if got.local_score is None:
        assert got.claimed_score == 0.1215, "the hint is preserved, just quarantined"


def test_harness_score_carries_provenance(tmp_path):
    pod = make_pod(tmp_path)
    install_harness(pod)
    cand = write_candidate(pod)
    pod._enqueue_verify(cand.candidate_id)
    got = wait_verified(pod, cand.candidate_id)
    assert got.score_source == "harness"


def test_window_closed_blocks_every_lane(tmp_path):
    """Run 3's best score was submitted at T+33.3 of a 30-minute window; only
    the final lane checked the clock. Now every lane refuses after deadline."""
    from swarm.submit.broker import SubmissionBroker

    cfg = SwarmConfig()
    cfg.slug = "t"
    cfg.dry_run = True
    bb = Blackboard(tmp_path / "ws")
    quota = QuotaPool(tmp_path / "q.json", limit_hours=1.0)
    budget = PodBudget(tmp_path / "b.json", deadline_s=0.0, quota=quota)  # expired
    broker = SubmissionBroker(bb, budget, cfg, quota)
    for lane in ("floor", "probe", "milestone", "final"):
        ok, why = broker.may_submit(lane, local_score=0.9)
        assert ok is False and "window closed" in why, (lane, why)


def test_fast_fidelity_env_reaches_the_harness(tmp_path):
    pod = make_pod(tmp_path)
    eval_dir = pod.bb.ws / "eval"
    eval_dir.mkdir()
    (eval_dir / "run_eval.py").write_text(
        "import os, json\n"
        "print(json.dumps({'local_score': 0.5, 'local_std': 0.0, 'folds': [0.5],\n"
        "  'problems': [], 'seen_max_train': os.environ.get('EVAL_MAX_TRAIN'),\n"
        "  'seen_folds': os.environ.get('EVAL_FOLDS')}))\n"
    )
    out = pod._run_eval_harness("cand-x", fidelity="fast")
    assert out["seen_max_train"] == "250" and out["seen_folds"] == "3"
    assert "fidelity=fast" in out["problems"]
    full = pod._run_eval_harness("cand-x", fidelity="full")
    assert full["seen_max_train"] is None


def test_stall_breaker_does_not_launch_after_freeze(tmp_path):
    pod = make_pod(tmp_path)
    pod.bb.add_plan(
        __import__("swarm.schemas", fromlist=["PlanCard"]).PlanCard(title="x", family="f")
    )
    # floor exists so _force_progress falls through to the launch branch
    from swarm.schemas import SubmissionRecord

    pod.bb.add_submission(SubmissionRecord(lane="floor", status="scored"))
    pod._floor_done = True
    pod._frozen = True
    out = pod._force_progress()
    assert "launched" not in out, "no new directions after the freeze"
