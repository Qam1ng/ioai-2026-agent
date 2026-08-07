"""Tests for the statistical conscience: calibration, significance, shrinkage.

All offline: synthetic blackboards are built on ``tmp_path`` through the real
``Blackboard`` API, so the tests exercise the same JSONL round-trip the
competition run uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.bus import Blackboard  # noqa: E402
from swarm.ledger import (  # noqa: E402
    GAP_WARN,
    ExperimentLedger,
    fit_calibration,
    fold_se_floor,
    improves_most_folds,
    probe_noise_floor,
    predicted_lb,
    shrunk_estimate,
    significant_gain,
)
from swarm.schemas import Calibration, Experiment, ProbeResult, SubmissionRecord  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_bb(tmp_path: Path, name: str = "ws") -> Blackboard:
    return Blackboard(tmp_path / name)


def add_pair(bb: Blackboard, local: float | None, lb: float | None, **kw) -> SubmissionRecord:
    rec = SubmissionRecord(
        candidate_id=kw.pop("candidate_id", "cand-1"),
        lane=kw.pop("lane", "milestone"),
        local_score=local,
        lb_score=lb,
        status="scored" if lb is not None else "queued",
        **kw,
    )
    bb.add_submission(rec)
    return rec


def add_exp(bb: Blackboard, score: float, std: float | None = None, folds=None, **kw) -> Experiment:
    exp = Experiment(
        candidate_id=kw.pop("candidate_id", "cand-1"),
        role=kw.pop("role", "tuner"),
        description=kw.pop("description", "synthetic experiment"),
        local_score=score,
        local_std=std,
        folds=list(folds or []),
        **kw,
    )
    bb.add_experiment(exp)
    return exp


# --------------------------------------------------------------------------- #
# fit_calibration
# --------------------------------------------------------------------------- #
def test_fit_calibration_with_zero_points_refuses_to_invent_a_line(tmp_path: Path):
    bb = make_bb(tmp_path)
    cal = fit_calibration(bb)

    assert cal.n_points == 0
    assert cal.slope is None
    assert cal.intercept is None
    assert cal.gap is None
    assert cal.warning, "a missing calibration must be stated loudly, not silently"
    lowered = cal.warning.lower()
    assert "no local->leaderboard calibration" in lowered
    assert "relative" in lowered


def test_fit_calibration_with_one_point_still_has_no_line(tmp_path: Path):
    bb = make_bb(tmp_path)
    add_pair(bb, 0.90, 0.78)
    cal = fit_calibration(bb)

    assert cal.n_points == 1
    assert cal.slope is None
    assert cal.residual_std is None
    # one point does give an offset, and it must be reported
    assert cal.gap == pytest.approx(0.12)
    assert "calibration" in cal.warning.lower()


def test_fit_calibration_with_unscored_submissions_ignores_them(tmp_path: Path):
    bb = make_bb(tmp_path)
    add_pair(bb, 0.80, None)  # queued, no leaderboard score yet
    add_pair(bb, 0.82, None)
    cal = fit_calibration(bb)
    assert cal.n_points == 0


def test_fit_calibration_recovers_a_known_linear_relation(tmp_path: Path):
    bb = make_bb(tmp_path)
    # lb = 0.9 * local - 0.05, exactly
    locals_ = [0.70, 0.75, 0.80, 0.85, 0.90]
    for x in locals_:
        add_pair(bb, x, 0.9 * x - 0.05)
    cal = fit_calibration(bb)

    assert cal.n_points == 5
    assert cal.slope == pytest.approx(0.9, abs=1e-9)
    assert cal.intercept == pytest.approx(-0.05, abs=1e-9)
    assert cal.residual_std == pytest.approx(0.0, abs=1e-9)
    expected_gap = sum(locals_) / 5 - sum(0.9 * x - 0.05 for x in locals_) / 5
    assert cal.gap == pytest.approx(expected_gap)
    assert predicted_lb(0.80, cal) == pytest.approx(0.9 * 0.80 - 0.05)


def test_fit_calibration_residual_std_reflects_scatter(tmp_path: Path):
    bb = make_bb(tmp_path)
    for x, noise in zip([0.60, 0.70, 0.80, 0.90], [0.01, -0.01, 0.01, -0.01]):
        add_pair(bb, x, x + noise)
    cal = fit_calibration(bb)

    assert cal.n_points == 4
    assert cal.residual_std is not None and cal.residual_std > 0
    # the band can never be smaller than the measured scatter around the line
    assert cal.noise_band >= cal.residual_std * 0.99


def test_fit_calibration_large_gap_fires_the_validation_design_warning(tmp_path: Path):
    """The 0.9156 / 0.78095 situation from memory/lessons/local-cv-optimism.md."""
    bb = make_bb(tmp_path)
    add_pair(bb, 0.9156, 0.78095)
    add_pair(bb, 0.9210, 0.78400)
    add_pair(bb, 0.9080, 0.77900)
    cal = fit_calibration(bb)

    assert cal.n_points == 3
    assert cal.gap is not None and cal.gap > GAP_WARN
    assert cal.gap == pytest.approx(0.13355, abs=1e-4)
    lowered = cal.warning.lower()
    assert "optimistic" in lowered
    assert "validation design fault" in lowered
    # and it must not read as a compliment about the model
    assert "leakage" in lowered or "grouping" in lowered


def test_fit_calibration_small_gap_does_not_warn(tmp_path: Path):
    bb = make_bb(tmp_path)
    for x in (0.70, 0.75, 0.80, 0.85):
        add_pair(bb, x, x - 0.005)
    cal = fit_calibration(bb)
    assert cal.gap == pytest.approx(0.005)
    assert "optimistic" not in cal.warning.lower()


def test_fit_calibration_two_points_flags_the_exact_fit(tmp_path: Path):
    bb = make_bb(tmp_path)
    add_pair(bb, 0.70, 0.69)
    add_pair(bb, 0.80, 0.79)
    cal = fit_calibration(bb)

    assert cal.n_points == 2
    assert cal.residual_std == pytest.approx(0.0)
    assert "two calibration points" in cal.warning.lower()


def test_fit_calibration_flags_anticorrelated_local_cv(tmp_path: Path):
    bb = make_bb(tmp_path)
    for x, y in [(0.70, 0.80), (0.80, 0.75), (0.90, 0.70)]:
        add_pair(bb, x, y)
    cal = fit_calibration(bb)

    assert cal.slope is not None and cal.slope < 0
    assert "anti-correlated" in cal.warning.lower()


def test_noise_band_uses_probe_noise_floor_when_it_is_the_largest_source(tmp_path: Path):
    bb = make_bb(tmp_path)
    # near-perfect line -> tiny residual std
    for x in (0.70, 0.75, 0.80, 0.85):
        add_pair(bb, x, x - 0.01)
    # but two supposedly equivalent submissions scored 0.02 apart
    bb.add_probe(ProbeResult(probe_kind="noise_floor", lb_score=0.700, hypothesis="repeat"))
    bb.add_probe(ProbeResult(probe_kind="noise_floor", lb_score=0.720, hypothesis="repeat"))

    floor = probe_noise_floor(bb)
    assert floor is not None and floor > 0.01

    cal = fit_calibration(bb)
    assert cal.noise_band >= floor * 0.99
    assert cal.noise_band > (cal.residual_std or 0.0)


def test_probe_noise_floor_prefers_an_explicit_number(tmp_path: Path):
    bb = make_bb(tmp_path)
    bb.add_probe(
        ProbeResult(probe_kind="noise_floor", lb_score=0.70, numeric={"noise_floor": 0.031})
    )
    assert probe_noise_floor(bb) == pytest.approx(0.031)


def test_probe_noise_floor_ignores_other_probe_kinds(tmp_path: Path):
    bb = make_bb(tmp_path)
    bb.add_probe(ProbeResult(probe_kind="constant", lb_score=0.10))
    bb.add_probe(ProbeResult(probe_kind="granularity", lb_score=0.60))
    assert probe_noise_floor(bb) is None


def test_fold_se_floor_is_the_standard_error_of_the_cv_mean(tmp_path: Path):
    bb = make_bb(tmp_path)
    add_exp(bb, 0.80, std=0.05, folds=[0.78, 0.82, 0.79, 0.81, 0.80])
    se = fold_se_floor(bb.get_experiments())
    assert se == pytest.approx(0.05 / (5 ** 0.5))


def test_noise_band_exists_before_any_leaderboard_point(tmp_path: Path):
    """Uncalibrated runs still need a band, or the first gate is a coin flip."""
    bb = make_bb(tmp_path)
    add_exp(bb, 0.80, std=0.04, folds=[0.76, 0.82, 0.79, 0.83, 0.80])
    cal = fit_calibration(bb)
    assert cal.n_points == 0
    assert cal.noise_band == pytest.approx(0.04 / (5 ** 0.5), abs=1e-6)


# --------------------------------------------------------------------------- #
# significant_gain / improves_most_folds
# --------------------------------------------------------------------------- #
def _cal(noise_band: float, n_points: int = 4) -> Calibration:
    return Calibration(
        n_points=n_points, slope=1.0, intercept=0.0, residual_std=noise_band,
        noise_band=noise_band, gap=0.0,
    )


def test_significant_gain_accepts_a_real_improvement():
    ok, reason = significant_gain(0.860, 0.800, 0.010, _cal(0.010))
    assert ok is True
    assert "accepted" in reason.lower()
    assert "0.0600" in reason  # the gain is quoted, not just asserted


def test_significant_gain_rejects_a_gain_inside_the_noise_band():
    ok, reason = significant_gain(0.8020, 0.8000, 0.030, _cal(0.010))
    assert ok is False
    assert "noise" in reason.lower()
    assert "rejected" in reason.lower()


def test_significant_gain_rejects_a_regression():
    ok, reason = significant_gain(0.79, 0.80, 0.01, _cal(0.005))
    assert ok is False
    assert "does not beat" in reason.lower()


def test_significant_gain_accepts_the_first_candidate():
    ok, reason = significant_gain(0.55, None, 0.02, _cal(0.01))
    assert ok is True
    assert "first scored candidate" in reason.lower()


def test_significant_gain_rejects_an_unscored_candidate():
    ok, reason = significant_gain(None, 0.80, 0.01, _cal(0.01))
    assert ok is False
    assert "no local score" in reason.lower()


def test_significant_gain_uses_the_candidates_own_noisy_cv_as_the_bar():
    """A candidate whose own CV is noisy must clear a higher bar."""
    cal = _cal(0.001)  # global band is tiny
    ok_quiet, _ = significant_gain(0.82, 0.80, 0.002, cal, new_folds=[0.82] * 5)
    ok_noisy, reason = significant_gain(0.82, 0.80, 0.200, cal, new_folds=[0.82] * 5)
    assert ok_quiet is True
    assert ok_noisy is False
    assert "noise band" in reason.lower()


def test_significant_gain_rejects_a_mean_gain_carried_by_one_lucky_fold():
    cal = _cal(0.005)
    best_folds = [0.80, 0.80, 0.80, 0.80, 0.80]
    new_folds = [0.79, 0.79, 0.79, 0.79, 0.99]  # mean up, but 1/5 folds improve
    new_mean = sum(new_folds) / 5
    ok, reason = significant_gain(new_mean, 0.80, 0.005, cal, new_folds, best_folds)
    assert ok is False
    assert "1/5 folds" in reason
    assert "lucky" in reason.lower()


def test_significant_gain_accepts_a_broad_gain_across_folds():
    cal = _cal(0.005)
    best_folds = [0.80, 0.80, 0.80, 0.80, 0.80]
    new_folds = [0.82, 0.83, 0.81, 0.84, 0.79]
    ok, reason = significant_gain(sum(new_folds) / 5, 0.80, 0.005, cal, new_folds, best_folds)
    assert ok is True
    assert "4/5 folds" in reason


def test_significant_gain_flags_a_run_with_no_noise_evidence_at_all():
    cal = Calibration(n_points=0, noise_band=0.0)
    ok, reason = significant_gain(0.81, 0.80, None, cal)
    assert ok is True
    assert "provisionally" in reason.lower()
    assert "not been tested" in reason.lower()


def test_significant_gain_k_multiplier_tightens_the_gate():
    cal = _cal(0.010)
    ok_1, _ = significant_gain(0.815, 0.800, 0.005, cal, k=1.0)
    ok_2, _ = significant_gain(0.815, 0.800, 0.005, cal, k=2.0)
    assert ok_1 is True
    assert ok_2 is False


def test_improves_most_folds_requires_a_strict_majority():
    assert improves_most_folds([0.9, 0.9, 0.9], [0.8, 0.8, 0.8]) is True
    assert improves_most_folds([0.9, 0.9, 0.7], [0.8, 0.8, 0.8]) is True  # 2/3
    assert improves_most_folds([0.9, 0.7, 0.7], [0.8, 0.8, 0.8]) is False  # 1/3
    assert improves_most_folds([0.9, 0.9, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]) is False  # 2/4 tie


def test_improves_most_folds_counts_ties_against_the_challenger():
    assert improves_most_folds([0.8, 0.8, 0.8], [0.8, 0.8, 0.8]) is False


def test_improves_most_folds_abstains_when_folds_are_not_comparable():
    # No evidence must not become a veto; the mean test then decides alone.
    assert improves_most_folds([], [0.8, 0.8]) is True
    assert improves_most_folds([0.9, 0.9], []) is True
    assert improves_most_folds([0.9, 0.9], [0.8, 0.8, 0.8]) is True


# --------------------------------------------------------------------------- #
# shrinkage / winner's curse
# --------------------------------------------------------------------------- #
def test_shrunk_estimate_pulls_a_noisy_high_lb_score_toward_the_prediction():
    # lb ~= local, with a wide residual spread and a wide noise band
    cal = Calibration(
        n_points=6, slope=1.0, intercept=0.0, residual_std=0.05, noise_band=0.04, gap=0.0
    )
    lb, local = 0.90, 0.80
    pred = predicted_lb(local, cal)
    shrunk = shrunk_estimate(lb, local, cal)

    assert pred == pytest.approx(0.80)
    assert pred < shrunk < lb, "must move toward the prediction but not all the way"


def test_shrunk_estimate_shrinks_harder_as_the_noise_band_grows():
    base = dict(n_points=6, slope=1.0, intercept=0.0, residual_std=0.05, gap=0.0)
    quiet = Calibration(noise_band=0.005, **base)
    loud = Calibration(noise_band=0.045, **base)

    s_quiet = shrunk_estimate(0.90, 0.80, quiet)
    s_loud = shrunk_estimate(0.90, 0.80, loud)
    assert s_loud < s_quiet < 0.90
    assert abs(s_loud - 0.80) < abs(s_quiet - 0.80)


def test_shrunk_estimate_is_a_no_op_without_measured_noise():
    cal = Calibration(n_points=5, slope=1.0, intercept=0.0, residual_std=0.02, noise_band=0.0)
    assert shrunk_estimate(0.91, 0.80, cal) == pytest.approx(0.91)


def test_shrunk_estimate_returns_the_raw_score_when_uncalibrated():
    cal = Calibration(n_points=0, noise_band=0.02)
    assert shrunk_estimate(0.91, 0.80, cal) == pytest.approx(0.91)


def test_shrunk_estimate_refuses_to_trust_a_two_point_line():
    """Two points fit exactly, so their zero residual is not evidence of anything."""
    cal = Calibration(
        n_points=2, slope=1.0, intercept=0.0, residual_std=0.0, noise_band=0.02, gap=0.0
    )
    assert shrunk_estimate(0.91, 0.80, cal) == pytest.approx(0.91)


def test_shrunk_estimate_falls_back_to_the_prediction_without_a_leaderboard_score():
    cal = Calibration(
        n_points=5, slope=0.9, intercept=-0.05, residual_std=0.02, noise_band=0.01
    )
    assert shrunk_estimate(None, 0.80, cal) == pytest.approx(0.9 * 0.80 - 0.05)


def test_shrunk_estimate_raises_when_there_is_nothing_to_estimate_from():
    with pytest.raises(ValueError):
        shrunk_estimate(None, None, Calibration(n_points=0))


def test_shrinkage_prefers_strong_local_cv_over_a_noise_lucky_lb_leader_winners_curse():
    """The winner's-curse test.

    Candidate A tops the leaderboard on private test A but its local CV is poor,
    so most of its lead is scoring luck. Candidate B is a hair behind on the
    leaderboard but its local CV — measured on far more samples — is much
    stronger. The final ranking re-scores on private test B, so we must submit
    the candidate with the best *expected* score there, which is B.
    """
    cal = Calibration(
        n_points=8, slope=1.0, intercept=0.0, residual_std=0.06, noise_band=0.05, gap=0.0
    )
    a_lb, a_local = 0.900, 0.780  # noise-lucky leader: LB far above its own CV
    b_lb, b_local = 0.890, 0.905  # boring, well-supported by a strong local CV

    assert a_lb > b_lb, "A wins the raw leaderboard argmax"

    a_shrunk = shrunk_estimate(a_lb, a_local, cal)
    b_shrunk = shrunk_estimate(b_lb, b_local, cal)

    assert b_shrunk > a_shrunk, "shrinkage must flip the ranking toward the robust candidate"
    assert a_shrunk < a_lb, "the lucky leader is marked down"
    assert b_shrunk > b_lb, "the well-supported candidate is marked up toward its CV"


# --------------------------------------------------------------------------- #
# ExperimentLedger
# --------------------------------------------------------------------------- #
def test_ledger_add_best_and_top_k(tmp_path: Path):
    bb = make_bb(tmp_path)
    led = ExperimentLedger(bb)
    led.add(Experiment(description="baseline", local_score=0.70, candidate_id="c1"))
    led.add(Experiment(description="better", local_score=0.82, candidate_id="c2"))
    led.add(Experiment(description="worse", local_score=0.61, candidate_id="c3"))
    led.add(Experiment(description="crashed", local_score=None, candidate_id="c3"))

    best = led.best()
    assert best is not None and best.description == "better"
    assert [e.description for e in led.top_k(2)] == ["better", "baseline"]
    assert len(led.all()) == 4
    assert len(led.scored()) == 3
    # the add() call mirrors into the audit log
    assert bb.events_path.exists()


def test_ledger_best_is_none_on_an_empty_ledger(tmp_path: Path):
    assert ExperimentLedger(make_bb(tmp_path)).best() is None


def test_ledger_as_table_is_markdown_and_quotes_reasons(tmp_path: Path):
    bb = make_bb(tmp_path)
    led = ExperimentLedger(bb)
    led.add(
        Experiment(
            description="tfidf + logreg | with a pipe",
            local_score=0.70,
            local_std=0.02,
            folds=[0.68, 0.71, 0.70, 0.72, 0.69],
            runtime_s=120.0,
            accepted=True,
            candidate_id="c1",
            role="tuner",
        )
    )
    led.add(
        Experiment(
            description="deeper net",
            local_score=0.701,
            accepted=False,
            reject_reason="rejected: gain inside the noise band",
            candidate_id="c2",
        )
    )
    table = led.as_table()

    assert table.startswith("| # | exp |")
    assert "|---|" in table
    assert "0.7000" in table and "0.7010" in table
    assert "accepted" in table
    assert "noise band" in table
    # a pipe inside a description must not break the table
    assert "tfidf + logreg / with a pipe" in table
    assert table.count("\n") >= 3


def test_ledger_as_table_handles_an_empty_ledger(tmp_path: Path):
    assert "No experiments" in ExperimentLedger(make_bb(tmp_path)).as_table()


def test_ledger_as_table_truncates_to_the_limit(tmp_path: Path):
    bb = make_bb(tmp_path)
    led = ExperimentLedger(bb)
    for i in range(10):
        led.add(Experiment(description=f"exp{i}", local_score=0.5 + i / 100))
    table = led.as_table(limit=3)
    assert "earlier experiments omitted" in table
    assert "exp9" in table
    assert "exp0" not in table


def test_ledger_summary_stats(tmp_path: Path):
    bb = make_bb(tmp_path)
    led = ExperimentLedger(bb)
    led.add(Experiment(local_score=0.70, accepted=True, runtime_s=60.0, candidate_id="c1"))
    led.add(Experiment(local_score=0.80, accepted=True, runtime_s=120.0, candidate_id="c2"))
    led.add(Experiment(local_score=0.60, accepted=False, runtime_s=60.0, candidate_id="c2"))

    st = led.summary_stats()
    assert st["n_experiments"] == 3
    assert st["n_scored"] == 3
    assert st["n_accepted"] == 2
    assert st["n_candidates"] == 2
    assert st["best_local"] == pytest.approx(0.80)
    assert st["worst_local"] == pytest.approx(0.60)
    assert st["median_local"] == pytest.approx(0.70)
    assert st["acceptance_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert st["total_experiment_minutes"] == pytest.approx(4.0)


def test_ledger_evaluate_uses_the_persisted_calibration(tmp_path: Path):
    bb = make_bb(tmp_path)
    bb.put_calibration(_cal(0.02))
    led = ExperimentLedger(bb)
    led.add(Experiment(local_score=0.80, local_std=0.01, folds=[0.80] * 5, accepted=True))

    ok_small, reason_small = led.evaluate(
        Experiment(local_score=0.805, local_std=0.01, folds=[0.805] * 5)
    )
    ok_big, _ = led.evaluate(Experiment(local_score=0.90, local_std=0.01, folds=[0.90] * 5))

    assert ok_small is False and "noise" in reason_small.lower()
    assert ok_big is True
