"""The verification contract: numbers come from a subprocess, never a model.

These tests pin the two halves of the Verifier redesign: the pod can score a
candidate by executing ``eval/run_eval.py`` deterministically, and when the
harness is missing or broken the failure is a *finding* rather than a crash.
"""

from __future__ import annotations

import json

from swarm.budget import PodBudget, QuotaPool
from swarm.bus import Blackboard
from swarm.config import SwarmConfig
from swarm.pod import TaskPod
from swarm.schemas import CandidateState, ManagerAction, TaskCard


def make_pod(tmp_path) -> TaskPod:
    cfg = SwarmConfig()
    cfg.slug = "test-comp"
    cfg.dry_run = True
    bb = Blackboard(tmp_path / "ws")
    quota = QuotaPool(tmp_path / "q.json", limit_hours=1.0)
    budget = PodBudget(tmp_path / "b.json", deadline_s=3600, quota=quota)
    bb.put_task_card(TaskCard(slug=cfg.slug, task_type="supervised", metric_name="acc"))

    class NullBroker:
        def submit(self, **kw):  # pragma: no cover - not reached in these tests
            raise AssertionError("no submissions expected")

    return TaskPod(cfg, bb, budget, quota, NullBroker())


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


def test_missing_harness_returns_none(tmp_path):
    pod = make_pod(tmp_path)
    assert pod._run_eval_harness("cand-x") is None


def test_harness_numbers_flow_into_candidate_state(tmp_path):
    pod = make_pod(tmp_path)
    cand = CandidateState(family="baseline", status="coding")
    pod.bb.put_candidate(cand)

    eval_dir = pod.bb.ws / "eval"
    eval_dir.mkdir()
    (eval_dir / "run_eval.py").write_text(HARNESS_OK)

    obs = pod._do_verify(ManagerAction(action="verify", target=cand.candidate_id))
    assert "0.75" in obs

    back = pod.bb.get_candidate(cand.candidate_id)
    assert back.local_score == 0.75
    assert back.local_std == 0.02
    assert back.folds == [0.73, 0.75, 0.77]
    assert back.status == "ready"

    exps = pod.bb.get_experiments()
    assert exps and exps[-1].role == "verify" and exps[-1].accepted


def test_harness_is_deterministic_across_reruns(tmp_path):
    pod = make_pod(tmp_path)
    eval_dir = pod.bb.ws / "eval"
    eval_dir.mkdir()
    (eval_dir / "run_eval.py").write_text(HARNESS_OK)
    a = pod._run_eval_harness("cand-1")
    b = pod._run_eval_harness("cand-1")
    assert a == b, "the whole point of the harness is that re-measurement is identical"


def test_broken_harness_is_a_finding_not_a_crash(tmp_path):
    pod = make_pod(tmp_path)
    eval_dir = pod.bb.ws / "eval"
    eval_dir.mkdir()
    (eval_dir / "run_eval.py").write_text(HARNESS_BROKEN)
    out = pod._run_eval_harness("cand-1")
    assert out is not None
    assert out.get("local_score") is None
    assert any("no JSON" in p for p in out.get("problems", []))


def test_verify_without_harness_uses_verifier_role_then_reruns(tmp_path):
    """When no harness exists, the Verifier role is invoked to build one, and
    the numbers are then taken from the harness it built — not its transcript."""
    pod = make_pod(tmp_path)
    cand = CandidateState(family="baseline")
    pod.bb.put_candidate(cand)

    class FakeVerifier:
        def run(self, instructions, context=None):
            # Simulate the role building a real harness on disk.
            eval_dir = pod.bb.ws / "eval"
            eval_dir.mkdir(exist_ok=True)
            (eval_dir / "run_eval.py").write_text(HARNESS_OK)

            class R:
                ok = True
                error = ""
                elapsed_s = 1.0
                # Deliberately wrong number in the report: the subprocess must win.
                report = {"local_score": 0.99, "local_std": 0.0, "folds": [0.99]}

            return R()

    pod._roles_override["verifier"] = FakeVerifier()
    pod._do_verify(ManagerAction(action="verify", target=cand.candidate_id))
    back = pod.bb.get_candidate(cand.candidate_id)
    assert back.local_score == 0.75, "harness measurement must override the model's claim"
