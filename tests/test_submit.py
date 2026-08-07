"""Tests for the submission layer. No network, no Kaggle account required.

These cover the parts where a silent mistake costs a competition day: metadata
that looks right but downgrades the machine, a version number parsed out of a
URL, a quota that can be double-spent, and a milestone gate that lets noise
through.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.budget import PodBudget, QuotaPool  # noqa: E402
from swarm.bus import Blackboard  # noqa: E402
from swarm.config import SwarmConfig  # noqa: E402
from swarm.schemas import Calibration, SubmissionRecord  # noqa: E402
from swarm.submit.broker import SubmissionBroker  # noqa: E402
from swarm.submit.kernel import (  # noqa: E402
    ACCEL_MACHINE_SHAPE,
    estimate_gpu_seconds,
    make_metadata,
    parse_push_version,
    parse_status,
    unique_kernel_id,
    write_kernel,
)
from swarm.submit.probes import (  # noqa: E402
    PROBES,
    get_probe,
    infer_n_from_scores,
    linear_calibration,
    probe_plan,
)

SLUG = "ioai-2026-ai-models-track-practice-task-1"


# ---------------------------------------------------------------- metadata
@pytest.mark.parametrize(
    "accel,expect_gpu,expect_shape",
    [
        ("cpu", "false", ""),
        ("p100", "true", "NvidiaTeslaP100"),
        ("t4", "true", "NvidiaTeslaT4"),
    ],
)
def test_metadata_accelerators(accel, expect_gpu, expect_shape):
    meta = make_metadata("me/my-kernel", "my_kernel.py", SLUG, accel)
    # enable_gpu must be a STRING; a JSON bool is the documented wrong form.
    assert isinstance(meta["enable_gpu"], str)
    assert meta["enable_gpu"] == expect_gpu
    assert isinstance(meta["machine_shape"], str)
    assert meta["machine_shape"] == expect_shape
    assert meta["kernel_type"] == "script"
    assert meta["is_private"] is True
    assert meta["competition_sources"] == [SLUG]
    assert meta["enable_internet"] == "false"  # scoring kernels have no internet
    assert meta["dataset_sources"] == []


def test_metadata_accelerator_table_is_complete():
    assert set(ACCEL_MACHINE_SHAPE) == {"cpu", "p100", "t4"}


def test_metadata_dataset_sources_workaround():
    """Task-2 workaround: data re-uploaded as a private dataset, competition
    source kept so the kernel still counts as a competition kernel."""
    meta = make_metadata(
        "me/my-kernel", "my_kernel.py", SLUG, "cpu", dataset_sources=["me/task2-data"]
    )
    assert meta["dataset_sources"] == ["me/task2-data"]
    assert meta["competition_sources"] == [SLUG]


def test_metadata_rejects_unknown_accelerator():
    with pytest.raises(ValueError):
        make_metadata("me/k", "k.py", SLUG, "a100")


def test_metadata_title_resolves_to_slug():
    meta = make_metadata("me/my-kernel-abcd", "x.py", SLUG, "cpu")
    assert meta["title"] == "my-kernel-abcd"  # else Kaggle warns about the slug
    assert len(meta["title"]) >= 5


def test_metadata_is_json_serialisable():
    meta = make_metadata("me/my-kernel", "my_kernel.py", SLUG, "t4")
    assert json.loads(json.dumps(meta))["enable_gpu"] == "true"


# --------------------------------------------------------------- kernel ids
def test_unique_kernel_id_differs_per_candidate():
    a = unique_kernel_id("tester", SLUG, "cand-11111111")
    b = unique_kernel_id("tester", SLUG, "cand-22222222")
    assert a != b
    assert a.startswith("tester/")


def test_unique_kernel_id_is_stable_and_short_and_slug_safe():
    a = unique_kernel_id("tester", SLUG, "cand-11111111")
    assert a == unique_kernel_id("tester", SLUG, "cand-11111111")
    slug = a.split("/", 1)[1]
    assert len(slug) <= 50
    assert all(c.islower() or c.isdigit() or c == "-" for c in slug)


def test_unique_kernel_id_distinguishes_competitions_with_shared_prefix():
    long_a = "ioai-2026-ai-models-track-practice-task-1"
    long_b = "ioai-2026-ai-models-track-practice-task-2"
    assert unique_kernel_id("t", long_a, "cand-1") != unique_kernel_id("t", long_b, "cand-1")


def test_write_kernel_materialises_dir(tmp_path):
    meta = make_metadata("me/probe-kernel", "probe_kernel.py", SLUG, "cpu")
    d = write_kernel(tmp_path / "k", "print('hi')", meta)
    assert (d / "probe_kernel.py").read_text() == "print('hi')"
    assert json.loads((d / "kernel-metadata.json").read_text())["id"] == "me/probe-kernel"


# ------------------------------------------------------------- CLI parsing
def test_parse_push_version_documented_shape():
    raw = (
        "Kernel version 7 successfully pushed.  Please check progress at "
        "https://www.kaggle.com/code/tester/my-kernel-1234"
    )
    assert parse_push_version(raw) == 7


def test_parse_push_version_ignores_digits_in_url():
    """run.py's 'last digit token wins' scan returns 1234 here. We must not."""
    raw = (
        "Kernel version 12 successfully pushed.  Please check progress at "
        "https://www.kaggle.com/code/tester/my-kernel-1234"
    )
    assert parse_push_version(raw) == 12


def test_parse_push_version_unknown():
    assert parse_push_version("Kernel push error: something went wrong") is None
    assert parse_push_version("") is None


def test_parse_status():
    assert parse_status('tester/k has status "COMPLETE"') == "COMPLETE"
    assert parse_status('tester/k has status "ERROR"') == "ERROR"
    assert parse_status("no status here") is None


def test_estimate_gpu_seconds():
    assert estimate_gpu_seconds("ran for 1234 seconds") == pytest.approx(1234)
    assert estimate_gpu_seconds("total 1h 05m 30s elapsed") == pytest.approx(3930)
    # Kaggle status carries no duration today -> the reservation estimate stands.
    assert estimate_gpu_seconds('k has status "COMPLETE"', fallback=1800.0) == 1800.0


# ------------------------------------------------------- probe inference
def test_infer_n_from_exact_scores():
    n = 250
    scores = [k / n for k in (183, 186, 190)]
    assert infer_n_from_scores(scores) == n


def test_infer_n_from_rounded_scores():
    """Kaggle reports five decimals, which is what the tolerance is sized for."""
    n = 7
    scores = [round(k / n, 5) for k in (3, 5)]
    assert infer_n_from_scores(scores, max_n=1000) == n


def test_infer_n_single_score_gives_smallest_denominator():
    assert infer_n_from_scores([0.5]) == 2
    assert infer_n_from_scores([0.25, 0.75]) == 4


def test_infer_n_returns_none_when_uninformative():
    assert infer_n_from_scores([]) is None
    assert infer_n_from_scores([0.0, 1.0]) is None  # every N fits
    assert infer_n_from_scores([None]) is None


def test_infer_n_returns_none_for_non_rate_metrics():
    # RMSE-like score outside [0, 1]: the k/N argument does not apply.
    assert infer_n_from_scores([12.34]) is None
    # Irrational-looking score no small denominator explains.
    assert infer_n_from_scores([0.3183098861], max_n=200) is None


def test_linear_calibration_recovers_a_line():
    fit = linear_calibration([(0.5, 0.4), (0.6, 0.5), (0.7, 0.6)])
    assert fit["slope"] == pytest.approx(1.0)
    assert fit["intercept"] == pytest.approx(-0.1)
    assert fit["gap"] == pytest.approx(0.1)  # local is optimistic by 0.1
    assert fit["residual_std"] == pytest.approx(0.0, abs=1e-9)


def test_linear_calibration_needs_two_points():
    assert linear_calibration([(0.5, 0.4)]) == {"n_points": 1}


# --------------------------------------------------------------- probe lib
def test_every_probe_builds_runnable_python():
    for kind, probe in PROBES.items():
        code = probe.build_code(None)
        assert "/kaggle/working/submission.csv" in code
        assert "os.walk" in code, f"{kind} must discover paths, never hardcode them"
        compile(code, f"<{kind}>", "exec")  # syntax check on generated code
        assert probe.accelerator == "cpu", "probes must cost zero GPU quota"


def test_probe_interpret_returns_probe_result():
    res = get_probe("constant_baseline").interpret(0.62, {"local_base_rate": 0.61})
    assert res.probe_kind == "constant"
    assert res.lb_score == 0.62
    assert "0.62" in res.inference or "0.620" in res.inference
    assert res.numeric["base_rate_drift"] == pytest.approx(0.01)


def test_granularity_probe_reports_n_and_quantum():
    res = get_probe("score_granularity").interpret(190 / 250, {"scores": [183 / 250, 186 / 250]})
    assert res.numeric["n_test"] == 250
    assert res.numeric["score_quantum"] == pytest.approx(1 / 250)


def test_noise_floor_probe_sets_the_band():
    res = get_probe("noise_floor").interpret(0.71, {"scores": [0.70, 0.72]})
    assert res.numeric["n_seeds"] == 3
    assert res.numeric["noise_band"] > 0


def test_probe_plan_is_all_known_probes():
    for kind, _opts in probe_plan(None):
        assert kind in PROBES


def test_split_shift_rejects_bad_half():
    with pytest.raises(ValueError):
        get_probe("split_shift").build_code(None, half="middle")


# ------------------------------------------------------------------ quota
def test_quota_reserve_settle_release_arithmetic(tmp_path):
    q = QuotaPool(tmp_path / "quota.json", limit_hours=1.0)
    assert q.remaining_seconds() == pytest.approx(3600.0)

    assert q.try_reserve("s1", 1800.0) is True
    assert q.remaining_seconds() == pytest.approx(1800.0)

    # Settling with the ACTUAL runtime returns the unused part of the estimate.
    q.settle("s1", 600.0)
    assert q.remaining_seconds() == pytest.approx(3000.0)

    assert q.try_reserve("s2", 1000.0) is True
    q.release("s2")  # kernel never ran
    assert q.remaining_seconds() == pytest.approx(3000.0)


def test_quota_cannot_over_reserve(tmp_path):
    q = QuotaPool(tmp_path / "quota.json", limit_hours=1.0)
    assert q.try_reserve("a", 2000.0) is True
    assert q.try_reserve("b", 2000.0) is False  # would exceed 3600s
    assert q.remaining_seconds() == pytest.approx(1600.0)
    # The refused reservation left no trace.
    assert q.try_reserve("b", 1600.0) is True
    assert q.remaining_seconds() == pytest.approx(0.0)


def test_quota_reservations_are_visible_to_a_second_pool_object(tmp_path):
    """Two pods share one account: the file is the shared truth, not memory."""
    path = tmp_path / "quota.json"
    a = QuotaPool(path, limit_hours=1.0)
    a.try_reserve("s1", 3000.0)
    b = QuotaPool(path, limit_hours=1.0)
    assert b.try_reserve("s2", 1000.0) is False


# ----------------------------------------------------------------- fixtures
def _make_broker(tmp_path, *, dry_run=True, elapsed_min=0.0, quota_hours=30.0):
    ws = tmp_path / "ws"
    bb = Blackboard(ws)
    quota = QuotaPool(tmp_path / "quota.json", limit_hours=quota_hours)
    budget = PodBudget(
        tmp_path / "budget.json",
        deadline_s=360 * 60,
        max_submissions=50,
        quota=quota,
    )
    if elapsed_min:
        budget.st.t0 = time.time() - elapsed_min * 60.0
        budget._save()
    cfg = SwarmConfig(
        slug=SLUG, workspace=str(ws), kaggle_user="tester", dry_run=dry_run
    )
    return SubmissionBroker(bb, budget, cfg, quota), bb, budget, quota


def _record(bb, lane, local_score=None, status="scored", lb_score=None):
    rec = SubmissionRecord(
        candidate_id="cand-old",
        lane=lane,
        local_score=local_score,
        lb_score=lb_score,
        status=status,
    )
    bb.add_submission(rec)
    return rec


# ------------------------------------------------------------- lane budget
def test_lane_allowance_reserves_for_final(tmp_path):
    broker, _bb, _budget, _q = _make_broker(tmp_path)
    a = broker.lane_allowance()
    assert a["floor"] == 1
    assert a["probe"] == 12
    # 50 total - 4 final reserve - 1 floor - 12 probe = 33 for milestones
    assert a["milestone"] == 33
    assert a["final"] == 50


def test_lane_allowance_shrinks_as_lanes_are_spent(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    _record(bb, "floor", local_score=0.5)
    a = broker.lane_allowance()
    assert a["floor"] == 0
    assert a["milestone"] == 33
    assert a["final"] == 49


def test_failed_submissions_do_not_consume_the_budget(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    _record(bb, "floor", status="error")
    # A record that never reached Kaggle must not burn the one floor slot.
    assert broker.lane_allowance()["floor"] == 1


# ---------------------------------------------------------------- the gate
def test_floor_lane_is_allowed_first_then_closed(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    ok, why = broker.may_submit("floor")
    assert ok, why
    _record(bb, "floor", local_score=0.5)
    ok, why = broker.may_submit("floor")
    assert not ok and "already exists" in why


def test_probe_lane_must_be_cpu(tmp_path):
    broker, _bb, _budget, _q = _make_broker(tmp_path)
    assert broker.may_submit("probe", accelerator="cpu")[0] is True
    ok, why = broker.may_submit("probe", accelerator="p100")
    assert not ok and "CPU-only" in why


def test_milestone_blocked_inside_the_noise_band(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    bb.put_calibration(Calibration(noise_band=0.01, n_points=3))
    _record(bb, "floor", local_score=0.800)

    ok, why = broker.may_submit("milestone", local_score=0.805)
    assert not ok
    assert "noise" in why.lower()

    ok, why = broker.may_submit("milestone", local_score=0.830)
    assert ok, why


def test_milestone_gate_scales_with_sigma(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    broker.cfg.submit.milestone_min_gain_sigma = 3.0
    bb.put_calibration(Calibration(noise_band=0.01))
    _record(bb, "floor", local_score=0.800)
    # 0.02 clears 1 sigma but not 3.
    assert broker.may_submit("milestone", local_score=0.820)[0] is False
    assert broker.may_submit("milestone", local_score=0.840)[0] is True


def test_milestone_requires_a_local_score(tmp_path):
    broker, _bb, _budget, _q = _make_broker(tmp_path)
    ok, why = broker.may_submit("milestone", local_score=None)
    assert not ok and "local CV score" in why


def test_first_milestone_has_nothing_to_beat(tmp_path):
    broker, _bb, _budget, _q = _make_broker(tmp_path)
    assert broker.may_submit("milestone", local_score=0.5)[0] is True


def test_probe_scores_do_not_raise_the_milestone_bar(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    _record(bb, "probe", local_score=0.99)  # a probe is built to be weird
    assert broker.best_submitted_local() is None
    assert broker.may_submit("milestone", local_score=0.4)[0] is True


def test_final_lane_only_inside_the_freeze_window(tmp_path):
    early, _bb, _budget, _q = _make_broker(tmp_path, elapsed_min=100.0)
    ok, why = early.may_submit("final", local_score=0.9)
    assert not ok and "opens at" in why

    inside, _bb2, _b2, _q2 = _make_broker(tmp_path / "b", elapsed_min=320.0)
    assert inside.may_submit("final", local_score=0.9)[0] is True

    late, _bb3, _b3, _q3 = _make_broker(tmp_path / "c", elapsed_min=350.0)
    ok, why = late.may_submit("final", local_score=0.9)
    assert not ok and "closed at" in why


def test_gpu_lane_blocked_when_quota_is_gone(tmp_path):
    broker, _bb, _budget, quota = _make_broker(tmp_path, quota_hours=0.1)
    ok, why = broker.may_submit("milestone", local_score=0.9, accelerator="p100")
    assert not ok and "GPU quota" in why
    # The same candidate on CPU is fine — CPU costs no quota.
    assert broker.may_submit("milestone", local_score=0.9, accelerator="cpu")[0] is True
    assert quota.remaining_seconds() == pytest.approx(360.0)


def test_unknown_lane_is_refused(tmp_path):
    broker, _bb, _budget, _q = _make_broker(tmp_path)
    ok, why = broker.may_submit("hail-mary")
    assert not ok and "unknown lane" in why


def test_gate_closes_when_the_submission_budget_runs_out(tmp_path):
    broker, _bb, budget, _q = _make_broker(tmp_path)
    budget.st.submissions = budget.st.max_submissions
    budget._save()
    assert broker.may_submit("floor")[0] is False
    assert broker.may_submit("probe", accelerator="cpu")[0] is False


# --------------------------------------------------------------- dry runs
def test_broker_dry_run_produces_a_well_formed_record(tmp_path):
    broker, bb, budget, quota = _make_broker(tmp_path, dry_run=True)
    code = "print('hello kaggle')\n"
    rec = broker.submit(
        code=code,
        candidate_id="cand-abc123",
        lane="floor",
        purpose="insurance baseline",
        accelerator="cpu",
        local_score=0.61,
    )

    assert isinstance(rec, SubmissionRecord)
    assert rec.status == "scored"
    assert rec.lb_score is None  # nothing was really scored
    assert rec.lane == "floor"
    assert rec.accelerator == "cpu"
    assert rec.local_score == 0.61
    assert rec.error == ""
    assert rec.kernel_ref.startswith("tester/")
    assert rec.gpu_seconds == 0.0

    # The kernel really was written: dry-run is a rehearsal, not a mock.
    kdir = Path(bb.candidate_dir("cand-abc123")) / f"kernel_{rec.sub_id}"
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] == "false"
    assert (kdir / meta["code_file"]).read_text() == code

    # And it is on the blackboard and counted.
    assert [r.sub_id for r in bb.get_submissions()] == [rec.sub_id]
    assert budget.st.submissions == 1
    assert quota.remaining_seconds() == pytest.approx(30 * 3600.0)


def test_broker_dry_run_gpu_lane_releases_its_reservation(tmp_path):
    broker, _bb, _budget, quota = _make_broker(tmp_path, dry_run=True)
    rec = broker.submit(
        code="print(1)",
        candidate_id="cand-gpu",
        lane="milestone",
        purpose="gpu candidate",
        accelerator="p100",
        local_score=0.7,
    )
    assert rec.status == "scored"
    # A rehearsal must not eat the real weekly quota.
    assert quota.remaining_seconds() == pytest.approx(30 * 3600.0)


def test_broker_blocked_submission_is_not_recorded(tmp_path):
    broker, bb, budget, _q = _make_broker(tmp_path, dry_run=True)
    bb.put_calibration(Calibration(noise_band=0.05))
    _record(bb, "floor", local_score=0.80)

    rec = broker.submit(
        code="print(1)",
        candidate_id="cand-noise",
        lane="milestone",
        purpose="noise chasing",
        accelerator="cpu",
        local_score=0.801,
    )
    assert rec.status == "error"
    assert rec.error.startswith("blocked:")
    assert len(bb.get_submissions()) == 1  # only the pre-existing floor record
    assert budget.st.submissions == 0


def test_probe_lane_gpu_request_is_refused_at_submit(tmp_path):
    broker, _bb, budget, _q = _make_broker(tmp_path, dry_run=True)
    rec = broker.submit(
        code="print(1)",
        candidate_id="cand-p",
        lane="probe",
        purpose="probe on gpu",
        accelerator="t4",
        local_score=None,
    )
    assert rec.status == "error" and "CPU" in rec.error
    assert budget.st.submissions == 0


# ------------------------------------------------------ winner's curse rule
def test_final_selection_shrinks_toward_the_local_prediction(tmp_path):
    """A lucky spike on a weak candidate must not beat a consistent one."""
    broker, bb, _budget, _q = _make_broker(tmp_path)
    bb.put_calibration(
        Calibration(n_points=4, slope=1.0, intercept=0.0, noise_band=0.05)
    )
    strong = SubmissionRecord(
        candidate_id="strong", lane="milestone", local_score=0.90,
        lb_score=0.90, status="scored",
    )
    lucky = SubmissionRecord(
        candidate_id="lucky", lane="milestone", local_score=0.70,
        lb_score=0.92, status="scored",
    )
    bb.add_submission(strong)
    bb.add_submission(lucky)

    assert max(broker._records(), key=lambda r: r.lb_score).candidate_id == "lucky"
    chosen = broker.final_selection()
    assert chosen is not None
    assert chosen.candidate_id == "strong", "raw argmax LB is the winner's curse"


def test_final_selection_is_raw_argmax_without_a_noise_estimate(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    bb.add_submission(
        SubmissionRecord(candidate_id="a", lane="milestone", local_score=0.9,
                         lb_score=0.80, status="scored")
    )
    bb.add_submission(
        SubmissionRecord(candidate_id="b", lane="milestone", local_score=0.5,
                         lb_score=0.81, status="scored")
    )
    # noise_band == 0 -> no evidence the LB is noisy -> trust it.
    assert broker.final_selection().candidate_id == "b"


def test_final_selection_none_when_nothing_scored(tmp_path):
    broker, _bb, _budget, _q = _make_broker(tmp_path)
    assert broker.final_selection() is None


def test_final_selection_falls_back_to_a_probe_if_that_is_all_we_have(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    bb.add_submission(
        SubmissionRecord(candidate_id="p", lane="probe", lb_score=0.4, status="scored")
    )
    chosen = broker.final_selection()
    assert chosen is not None and chosen.candidate_id == "p"


def test_final_ranking_is_ordered_and_complete(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path)
    bb.put_calibration(Calibration(noise_band=0.02, slope=1.0, intercept=0.0, n_points=3))
    for cid, local, lb in (("a", 0.5, 0.51), ("b", 0.6, 0.62), ("c", 0.55, 0.54)):
        bb.add_submission(
            SubmissionRecord(candidate_id=cid, lane="milestone", local_score=local,
                             lb_score=lb, status="scored")
        )
    ranked = broker.final_ranking()
    assert len(ranked) == 3
    assert [s for _r, s in ranked] == sorted([s for _r, s in ranked], reverse=True)


def test_poll_scores_is_a_noop_in_dry_run(tmp_path):
    broker, bb, _budget, _q = _make_broker(tmp_path, dry_run=True)
    _record(bb, "milestone", status="queued")
    assert broker.poll_scores() == 0


def test_broker_status_summarises_everything_a_manager_needs(tmp_path):
    broker, _bb, _budget, _q = _make_broker(tmp_path)
    st = broker.status()
    for key in ("used_by_lane", "allowance", "gpu_quota", "submissions_left", "noise_band"):
        assert key in st


def test_kernel_id_is_unique_per_submission_not_just_per_candidate():
    """First live run: the floor lane and the milestone lane pushed the SAME
    candidate to the SAME Kaggle kernel slug concurrently; both threads then
    polled one kernel and neither submission completed."""
    from swarm.submit.kernel import unique_kernel_id

    user, slug, cand = "someuser", "ioai-2026-ai-models-track-practice-task-1", "cand-eca53733"
    floor = unique_kernel_id(user, slug, cand, "sub-202705bd")
    milestone = unique_kernel_id(user, slug, cand, "sub-eda5672b")
    assert floor != milestone, "concurrent pushes must not share a kernel slug"
    # still deterministic, still scoped to the competition and the candidate
    assert floor == unique_kernel_id(user, slug, cand, "sub-202705bd")
    assert unique_kernel_id(user, slug, "cand-a") != unique_kernel_id(user, slug, "cand-b")
    for ref in (floor, milestone):
        owner, _, kslug = ref.partition("/")
        assert owner == user and 0 < len(kslug) <= 50


def test_parse_status_accepts_the_real_cli_enum_format():
    """The live CLI prints `has status "KernelWorkerStatus.COMPLETE"`. The old
    pattern matched only the bare word, so a finished kernel polled until the
    timeout and its submission was never made."""
    from swarm.submit.kernel import parse_status

    assert parse_status('ref has status "KernelWorkerStatus.COMPLETE"') == "COMPLETE"
    assert parse_status('ref has status "KernelWorkerStatus.RUNNING"') == "RUNNING"
    assert parse_status('ref has status "KernelWorkerStatus.QUEUED"') == "QUEUED"
    assert (
        parse_status('ref has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"')
        == "CANCEL_ACKNOWLEDGED"
    )
    # the bare form must keep working
    assert parse_status('ref has status "complete"') == "COMPLETE"
    assert parse_status("no status here") is None
