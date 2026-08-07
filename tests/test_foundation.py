"""Tests for the shared foundation: schemas, blackboard, budgets, quota, config.

These cover the invariants the competition day depends on. In particular the
budget-persistence test exists because the pre-existing single-agent harness
reset its clock and counters on every process start, which silently granted a
resumed run a second full budget.
"""

from __future__ import annotations

import json
import threading

import pytest

from swarm.budget import PodBudget, QuotaPool
from swarm.bus import Blackboard
from swarm.config import ModelSpec, SwarmConfig
from swarm.schemas import (
    Calibration,
    CandidateState,
    Constraint,
    Experiment,
    ManagerAction,
    PlanCard,
    ProbeResult,
    SubmissionRecord,
    TaskCard,
    VALID_ACTIONS,
)


# --------------------------------------------------------------- schemas
def test_task_card_roundtrip_preserves_constraint_quotes():
    card = TaskCard(
        slug="s",
        task_type="supervised",
        metric_name="0.5*acc_old + 0.5*acc_new",
        constraints=[
            Constraint(kind="must", quote="You must start from the provided checkpoint"),
            Constraint(kind="must_not", quote="No training from scratch"),
        ],
    )
    back = TaskCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert [c.quote for c in back.constraints] == [c.quote for c in card.constraints]
    assert back.constraints[0].kind == "must"


def test_manager_action_degrades_gracefully_on_garbage():
    a = ManagerAction.from_dict({})
    assert a.action == "wait"
    a2 = ManagerAction.from_dict({"action": "code", "target": None, "reason": None})
    assert a2.action == "code" and a2.target == "" and a2.reason == ""


def test_valid_actions_cover_the_documented_set():
    assert {"code", "design", "verify", "probe", "aggregate", "retire", "stop"} <= VALID_ACTIONS


# ------------------------------------------------------------ blackboard
def test_blackboard_persists_every_artifact_type(tmp_path):
    bb = Blackboard(tmp_path)
    bb.put_task_card(TaskCard(slug="s", task_type="imitation", metric_name="success_rate"))
    bb.add_plan(PlanCard(title="bc", family="behaviour_cloning"))
    cand = CandidateState(family="behaviour_cloning", local_score=0.7, folds=[0.6, 0.8])
    bb.put_candidate(cand)
    bb.add_experiment(Experiment(candidate_id=cand.candidate_id, role="coder", accepted=True))
    bb.add_submission(SubmissionRecord(candidate_id=cand.candidate_id, lane="floor"))
    bb.add_probe(ProbeResult(probe_kind="constant", inference="majority class is 0.31"))
    bb.put_calibration(Calibration(n_points=3, noise_band=0.02))

    assert bb.get_task_card().task_type == "imitation"
    assert len(bb.get_plans()) == 1
    assert bb.get_candidate(cand.candidate_id).local_score == 0.7
    assert len(bb.get_experiments()) == 1
    assert len(bb.get_submissions()) == 1
    assert bb.get_probes()[0].inference.startswith("majority")
    assert bb.get_calibration().noise_band == 0.02
    assert bb.best_candidate().candidate_id == cand.candidate_id


def test_blackboard_snapshot_is_json_serialisable(tmp_path):
    bb = Blackboard(tmp_path)
    bb.put_task_card(TaskCard(slug="s"))
    bb.put_candidate(CandidateState(local_score=0.5))
    json.dumps(bb.snapshot(), default=str)  # must not raise


def test_update_submission_only_touches_the_named_record(tmp_path):
    bb = Blackboard(tmp_path)
    a = SubmissionRecord(lane="floor")
    b = SubmissionRecord(lane="probe")
    bb.add_submission(a)
    bb.add_submission(b)
    bb.update_submission(a.sub_id, lb_score=0.42, status="scored")
    got = {s.sub_id: s for s in bb.get_submissions()}
    assert got[a.sub_id].lb_score == 0.42
    assert got[b.sub_id].lb_score is None


def test_concurrent_appends_do_not_lose_rows(tmp_path):
    """Parallel candidates write the ledger simultaneously; none may be lost."""
    bb = Blackboard(tmp_path)

    def worker(n: int) -> None:
        for i in range(20):
            bb.add_experiment(Experiment(candidate_id=f"c{n}", description=f"e{i}"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(bb.get_experiments()) == 120


def test_live_candidates_excludes_retired_and_failed(tmp_path):
    bb = Blackboard(tmp_path)
    for status in ("ready", "coding", "retired", "failed"):
        bb.put_candidate(CandidateState(status=status))
    assert len(bb.get_candidates()) == 4
    assert len(bb.live_candidates()) == 2


# ---------------------------------------------------------------- quota
def test_quota_cannot_be_over_reserved(tmp_path):
    q = QuotaPool(tmp_path / "q.json", limit_hours=1.0)
    assert q.try_reserve("a", 1800) is True
    assert q.try_reserve("b", 2160) is False  # would exceed the hour
    assert q.try_reserve("c", 1700) is True
    assert q.remaining_seconds() == pytest.approx(100, abs=1)


def test_quota_settle_and_release(tmp_path):
    q = QuotaPool(tmp_path / "q.json", limit_hours=1.0)
    q.try_reserve("a", 1800)
    q.settle("a", 1200)  # kernel finished faster than reserved
    assert q.status()["gpu_hours_used"] == pytest.approx(1200 / 3600, abs=0.01)
    assert q.status()["gpu_hours_reserved"] == 0.0

    q.try_reserve("b", 900)
    q.release("b")  # push failed; the reservation must come back
    assert q.status()["gpu_hours_reserved"] == 0.0


def test_quota_is_shared_across_pods(tmp_path):
    """All three problems of a day share one Kaggle account, hence one pool."""
    path = tmp_path / "q.json"
    pod_a = QuotaPool(path, limit_hours=1.0)
    pod_b = QuotaPool(path, limit_hours=1.0)
    assert pod_a.try_reserve("a", 2000) is True
    assert pod_b.try_reserve("b", 2000) is False


# --------------------------------------------------------------- budget
def test_budget_survives_restart_without_granting_a_second_budget(tmp_path):
    p = tmp_path / "budget.json"
    b1 = PodBudget(p, deadline_s=3600, max_submissions=50)
    b1.note_llm({"input_tokens": 1000, "output_tokens": 500}, 1.25)
    b1.note_submit()
    b1.note_submit()
    t0_before = b1.st.t0

    b2 = PodBudget(p, deadline_s=3600, max_submissions=50)
    assert b2.st.t0 == t0_before, "restart must not reset the clock"
    assert b2.st.submissions == 2
    assert b2.st.cost_usd == pytest.approx(1.25)
    assert b2.st.tokens_in == 1000


def test_budget_halts_on_cost_but_not_on_submission_count(tmp_path):
    b = PodBudget(tmp_path / "b.json", deadline_s=3600, max_submissions=1, max_cost_usd=1.0)
    b.note_submit()
    assert b.can_submit() is False
    assert b.exhausted() is None, "a spent submission budget must not stop the run"
    b.note_llm({"input_tokens": 0, "output_tokens": 0}, 1.0)
    assert b.exhausted() == "LLM cost budget reached"


def test_budget_deadline(tmp_path):
    b = PodBudget(tmp_path / "b.json", deadline_s=0.0)
    assert b.exhausted() == "wall-clock deadline reached"
    assert b.remaining() == 0.0


# --------------------------------------------------------------- config
def test_config_env_overrides_file(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "deadline_min: 100\n"
        "default_model:\n  backend: claude\n  model: from-file\n"
        "parallel:\n  max_candidates: 2\n"
    )
    monkeypatch.setenv("SWARM_MODEL", "from-env")
    monkeypatch.setenv("SWARM_BACKEND", "openai")
    monkeypatch.setenv("SWARM_MAX_CANDIDATES", "7")
    cfg = SwarmConfig.load(cfg_file)
    assert cfg.default_model.model == "from-env"
    assert cfg.default_model.backend == "openai"
    assert cfg.parallel.max_candidates == 7
    assert cfg.deadline_min == 100


def test_model_for_falls_back_to_default():
    cfg = SwarmConfig()
    cfg.default_model = ModelSpec(backend="openai", model="d")
    cfg.models["coder"] = ModelSpec(backend="claude_code", model="c")
    assert cfg.model_for("coder").backend == "claude_code"
    assert cfg.model_for("manager").model == "d"


def test_shipped_default_config_loads_and_is_sane():
    cfg = SwarmConfig.load("configs/default.yaml")
    assert cfg.gates.floor_submit_min < cfg.gates.freeze_min < cfg.deadline_min
    assert cfg.submit.max_submissions == 50
    assert cfg.model_for("coder").backend == "claude_code"
