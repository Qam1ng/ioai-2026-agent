"""The statistical conscience of the swarm: calibration, significance, shrinkage.

Everything here exists because of one measured failure, recorded in
``memory/lessons/local-cv-optimism.md``: a single 25% stratified holdout read
**0.9156** while the leaderboard read **0.78095**. Two separate lessons hide in
that gap, and this module implements the defence for each.

1. **A local score is not a leaderboard score.** The offset is large, it is not
   noise, and it means the *validation design* is wrong (wrong grouping, leakage,
   or distribution shift). :func:`fit_calibration` measures the offset from real
   submissions and refuses to stay silent about it.

2. **Most "improvements" are noise.** Chasing a phantom 0.0 accuracy on a class
   with one validation sample actively *regressed* later iterations. So a new
   best must clear a noise band **and** improve on most folds before it is
   allowed to consume a submission: :func:`significant_gain`.

A third defence is needed only at the very end. The final ranking re-scores our
best-on-private-A submission on private test **B**. Picking the raw argmax over A
is the classic winner's curse — see :func:`shrunk_estimate`.

Score orientation
-----------------
Every score on the blackboard is stored **maximize-oriented** (higher is better);
a minimize metric such as RMSE must be negated by the role that records it. This
matches :meth:`swarm.bus.Blackboard.best_candidate`, which takes a plain ``max``.
Mixing orientations would silently invert every gate in this file, so the
convention is enforced by documentation and by using ``max`` everywhere here too.

Units
-----
``Calibration.noise_band`` is expressed in **local CV units**, because that is
what it gates: local gains. Leaderboard-side evidence (regression residuals,
noise-floor probes) is converted into local units by dividing by the fitted
slope. :func:`lb_units_noise` converts back for leaderboard-side reasoning.
"""

from __future__ import annotations

import math
from statistics import median, pstdev, stdev
from typing import Any, Iterable, Sequence

from .bus import Blackboard
from .schemas import Calibration, Experiment

# --------------------------------------------------------------------------- #
# Named constants. Every threshold in this file is a judgement call; each one is
# named, defaulted, and justified here rather than buried as a magic number.
# --------------------------------------------------------------------------- #

#: Multiplier on the noise band in :func:`significant_gain`. 1.0 = "the gain must
#: exceed one standard error of the CV mean". Why 1.0 and not 2.0: a 6-hour
#: window yields on the order of tens of experiments on small data, where a 2σ
#: gate rejects essentially every real gain and the pod stalls on its baseline;
#: a 0σ gate reproduces exactly the noise-chasing regression we measured. 1.0 is
#: the smallest gate that still kills single-fold luck, and it is overridable
#: (``SwarmConfig.submit.milestone_min_gain_sigma`` defaults to the same 1.0).
SIGNIFICANCE_K = 1.0

#: mean(local) - mean(lb) above this is reported as a design fault, not noise.
#: 0.05 is ~1/3 of the 0.135 gap we actually observed and far above any plausible
#: scoring wobble for a bounded metric, so it flags real optimism without firing
#: on ordinary train/test variation.
GAP_WARN = 0.05

#: Folds assumed when an experiment reports a std but not its fold vector.
#: 5-fold is the swarm's default validation design.
DEFAULT_FOLDS = 5

#: Below this the fitted slope is treated as uninformative for unit conversion
#: (dividing by a near-zero slope would explode the noise band to infinity).
MIN_ABS_SLOPE = 0.05

#: A gain is "on most folds" when strictly more than this share of comparable
#: folds improve. 0.5 = a strict majority; ties count against the challenger.
FOLD_MAJORITY = 0.5

#: Minimum calibration points before shrinkage is allowed to move a score.
#: A line through 2 points has zero residual degrees of freedom: its residual
#: spread is 0 by construction, which would drive the shrinkage weight to full
#: trust in a line fitted to two submissions. 3 is the first n at which the
#: residual is actually measured rather than assumed.
MIN_SHRINK_POINTS = 3


# --------------------------------------------------------------------------- #
# small numeric helpers (stdlib only: this module must never fail to import)
# --------------------------------------------------------------------------- #
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _std(values: Sequence[float]) -> float | None:
    """Sample std (ddof=1); ``None`` when there is not enough data."""
    vals = [float(v) for v in values if _finite(v)]
    if len(vals) < 2:
        return None
    return stdev(vals)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


# --------------------------------------------------------------------------- #
# noise evidence
# --------------------------------------------------------------------------- #
def probe_noise_floor(bb: Blackboard) -> float | None:
    """Leaderboard noise floor measured by ``noise_floor`` probes, in LB units.

    Two readings are accepted, in order of preference:

    * an explicit number a Prober wrote into ``numeric`` (it may have computed
      the floor from a source we cannot see, e.g. repeated identical kernels);
    * the spread of leaderboard scores across several ``noise_floor`` probes —
      submissions designed to be equivalent, so any difference between them *is*
      the scoring noise.
    """
    probes = [p for p in bb.get_probes() if p.probe_kind == "noise_floor"]
    if not probes:
        return None

    explicit: list[float] = []
    for p in probes:
        for key in ("noise_floor", "lb_std", "std", "band", "spread"):
            v = (p.numeric or {}).get(key)
            if _finite(v):
                explicit.append(abs(float(v)))
                break
    if explicit:
        return max(explicit)

    scores = [float(p.lb_score) for p in probes if _finite(p.lb_score)]
    return _std(scores)


def fold_se_floor(experiments: Iterable[Experiment]) -> float | None:
    """Typical standard error of a CV mean, in local units.

    This is the floor below which *no* local comparison can be trusted, even
    with a perfect leaderboard calibration: it is the wobble of our own
    measuring stick. The median across experiments is used rather than the mean
    so one pathological run cannot inflate the whole band.
    """
    ses: list[float] = []
    for e in experiments:
        folds = [float(f) for f in (e.folds or []) if _finite(f)]
        k = len(folds) if len(folds) >= 2 else DEFAULT_FOLDS
        sd: float | None = float(e.local_std) if _finite(e.local_std) else None
        if sd is None and len(folds) >= 2:
            sd = _std(folds)
        if sd is None or sd <= 0:
            continue
        ses.append(sd / math.sqrt(k))
    if not ses:
        return None
    return median(ses)


def lb_units_noise(cal: Calibration) -> float:
    """``cal.noise_band`` (local units) expressed in leaderboard units."""
    slope = cal.slope if _finite(cal.slope) else None
    scale = abs(slope) if slope is not None and abs(slope) >= MIN_ABS_SLOPE else 1.0
    return float(cal.noise_band) * scale


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def fit_calibration(bb: Blackboard) -> Calibration:
    """Regress leaderboard score on local CV score over every scored submission.

    The fit answers two different questions and it is worth keeping them apart:

    * *Where do we actually stand?* — ``slope``/``intercept`` map a local score
      to an expected leaderboard score.
    * *Is our validation design sound?* — ``gap`` and ``warning``. A large gap is
      not a reason to celebrate a high local score; it is a bug report about the
      CV design, which is precisely the 0.9156 → 0.78095 situation.

    With fewer than two paired points no line exists. Rather than fabricate one,
    the returned :class:`~swarm.schemas.Calibration` carries ``n_points`` and a
    warning; ``noise_band`` is still populated from probe and fold evidence,
    because those do not require a leaderboard fit and the significance gate
    needs *some* band from the first experiment onwards.
    """
    pairs = [
        (float(s.local_score), float(s.lb_score))
        for s in bb.scored_submissions()
        if _finite(s.local_score) and _finite(s.lb_score)
    ]
    experiments = bb.get_experiments()
    probe_nf = probe_noise_floor(bb)  # LB units
    fold_se = fold_se_floor(experiments)  # local units

    n = len(pairs)
    if n < 2:
        band = max(
            [v for v in (probe_nf, fold_se) if v is not None and v > 0] or [0.0]
        )
        warn = (
            f"No local->leaderboard calibration exists yet ({n} paired "
            "submission(s); 2 are needed). Local CV scores are RELATIVE signals "
            "only: use them to rank our own candidates, never as an estimate of "
            "leaderboard standing. Submit early to buy the first calibration "
            "point."
        )
        return Calibration(
            n_points=n,
            slope=None,
            intercept=(pairs[0][1] if n == 1 else None),
            residual_std=None,
            noise_band=round(band, 6),
            gap=(pairs[0][0] - pairs[0][1] if n == 1 else None),
            warning=warn,
        )

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)

    warnings: list[str] = []
    if sxx <= 1e-12:
        # Every submission carried the same local score: the slope is not
        # identifiable. Fall back to "leaderboard is a constant plus noise",
        # which is honest and still usable by shrunk_estimate.
        slope: float | None = None
        intercept: float | None = my
        residual_std: float | None = pstdev(ys) if len(ys) > 1 else 0.0
        warnings.append(
            "All calibration points share one local score, so no slope can be "
            "estimated; only the average offset is known."
        )
    else:
        slope = sxy / sxx
        intercept = my - slope * mx
        resid = [y - (slope * x + intercept) for x, y in pairs]
        # n-2 degrees of freedom for a two-parameter fit. With n == 2 the fit is
        # exact, sse == 0, and residual_std == 0 is a statement about the fit,
        # not about the noise -- hence the explicit warning and the probe/fold
        # floors below.
        dof = max(n - 2, 1)
        residual_std = math.sqrt(sum(r * r for r in resid) / dof)

    gap = mx - my  # positive == local is optimistic

    slope_scale = abs(slope) if slope is not None and abs(slope) >= MIN_ABS_SLOPE else 1.0
    components = [
        (residual_std / slope_scale) if residual_std is not None else None,
        (probe_nf / slope_scale) if probe_nf is not None else None,
        fold_se,
    ]
    band = max([c for c in components if c is not None and c > 0] or [0.0])

    if n == 2:
        warnings.append(
            "Only two calibration points: the line fits them exactly, so the "
            "residual spread is 0 by construction and says nothing about noise. "
            "The noise band falls back to probe and cross-fold evidence."
        )
    if gap > GAP_WARN:
        warnings.append(
            f"Local CV is optimistic by {gap:.4f} (mean local {mx:.4f} vs mean "
            f"leaderboard {my:.4f}). A gap this large is a VALIDATION DESIGN "
            "FAULT, not a good model: suspect leakage, a missing grouping "
            "variable in the folds, or a train/test distribution shift. Fix the "
            "CV design before trusting any further local gain, and never quote "
            "the local score as an expected leaderboard score."
        )
    elif gap < -GAP_WARN:
        warnings.append(
            f"The leaderboard is {abs(gap):.4f} ABOVE local CV. Usually benign "
            "(the hidden test is easier, or full-data retraining helps), but it "
            "means local scores understate us: do not abandon a candidate on a "
            "local score alone."
        )
    if slope is not None and slope <= 0 and n >= 3:
        warnings.append(
            f"Fitted slope is {slope:.3f} (<= 0): local improvements have been "
            "ANTI-correlated with the leaderboard so far. Treat every local gain "
            "as unproven until a submission confirms it."
        )

    return Calibration(
        n_points=n,
        slope=slope,
        intercept=intercept,
        residual_std=residual_std,
        noise_band=round(band, 6),
        gap=gap,
        warning=" ".join(warnings),
    )


def refresh_calibration(bb: Blackboard) -> Calibration:
    """Fit and persist. Convenience for the pod after every score arrives."""
    cal = fit_calibration(bb)
    bb.put_calibration(cal)
    bb.event(
        "calibration",
        n_points=cal.n_points,
        slope=cal.slope,
        gap=cal.gap,
        noise_band=cal.noise_band,
    )
    return cal


# --------------------------------------------------------------------------- #
# significance
# --------------------------------------------------------------------------- #
def _fold_wins(
    new_folds: Sequence[float] | None, best_folds: Sequence[float] | None
) -> tuple[int, int] | None:
    """``(wins, comparable_folds)``, or ``None`` when folds are not comparable.

    Folds are only comparable when both vectors exist and have the same length,
    which is the swarm's proxy for "the same CV split was used". Comparing
    different split designs fold-by-fold would be meaningless.
    """
    if not new_folds or not best_folds:
        return None
    a = [float(v) for v in new_folds if _finite(v)]
    b = [float(v) for v in best_folds if _finite(v)]
    if len(a) != len(b) or not a:
        return None
    return sum(1 for x, y in zip(a, b) if x > y), len(a)


def improves_most_folds(
    new_folds: Sequence[float] | None, best_folds: Sequence[float] | None
) -> bool:
    """Did the challenger beat the incumbent on a strict majority of folds?

    This is the cheap non-parametric companion to the mean test. A mean gain
    driven by one lucky fold — exactly the 3-sample-class artefact we measured —
    fails here even when it passes the mean test.

    When the fold vectors are missing or not comparable this returns ``True``:
    absence of evidence must not by itself veto a candidate, so the decision
    falls back to the mean test alone (and :func:`significant_gain` says so in
    its reason string).
    """
    wins = _fold_wins(new_folds, best_folds)
    if wins is None:
        return True
    n_wins, n_folds = wins
    return n_wins > FOLD_MAJORITY * n_folds


def significant_gain(
    new_score: float | None,
    best_score: float | None,
    new_std: float | None,
    cal: Calibration,
    new_folds: Sequence[float] | None = None,
    best_folds: Sequence[float] | None = None,
    k: float = SIGNIFICANCE_K,
) -> tuple[bool, str]:
    """Decide whether ``new_score`` is a *real* improvement on ``best_score``.

    Two conditions, both required:

    1. the mean gain exceeds ``k`` x the effective noise band;
    2. when fold vectors are comparable, the challenger wins a majority of folds.

    The effective band is the worst (largest) of the calibrated band and this
    candidate's own measurement error, because a candidate with a noisy CV must
    clear a higher bar than the global average implies.

    Returns the decision **and a plain-language reason**: the reason is written
    into the ledger and shown to the Manager, and a Manager that cannot see why
    a gain was rejected will simply propose it again.
    """
    if not _finite(new_score):
        return False, "rejected: the candidate has no local score to compare."
    new_score = float(new_score)

    if not _finite(best_score):
        return True, (
            f"accepted: local {new_score:.4f} is the first scored candidate, so "
            "it becomes the incumbent by default (nothing to compare against)."
        )
    best_score = float(best_score)
    gain = new_score - best_score

    # Own-measurement error: standard error of this candidate's CV mean.
    own_se: float | None = None
    folds_n = len([f for f in (new_folds or []) if _finite(f)])
    if _finite(new_std) and float(new_std) > 0:
        own_se = float(new_std) / math.sqrt(folds_n if folds_n >= 2 else DEFAULT_FOLDS)
    elif folds_n >= 2:
        sd = _std([float(f) for f in new_folds or [] if _finite(f)])
        if sd:
            own_se = sd / math.sqrt(folds_n)

    band = max([v for v in (float(cal.noise_band), own_se) if v is not None] or [0.0])
    threshold = k * band

    parts = [f"noise band {band:.4f}"]
    if cal.n_points >= 2:
        parts.append(f"calibrated on {cal.n_points} leaderboard points")
    else:
        parts.append("uncalibrated: band from fold/probe evidence only")
    if own_se is not None:
        parts.append(f"this candidate's CV standard error {own_se:.4f}")
    basis = "; ".join(parts)

    if gain <= 0:
        return False, (
            f"rejected: local {new_score:.4f} does not beat the incumbent "
            f"{best_score:.4f} (change {gain:+.4f})."
        )

    if band <= 0:
        # No noise evidence at all (first experiment, no folds, no probes). Let a
        # positive gain through, but say loudly that nothing was actually tested.
        return True, (
            f"accepted PROVISIONALLY: local gain {gain:+.4f} over {best_score:.4f}, "
            "but no noise estimate exists yet (no fold std, no probes, no "
            "calibration) so this gain has NOT been tested for significance. "
            "Record folds or run a noise-floor probe."
        )

    if gain <= threshold:
        return False, (
            f"rejected: local gain {gain:+.4f} is inside the noise band "
            f"({k:g} x {band:.4f} = {threshold:.4f}) — indistinguishable from CV "
            f"noise, so submitting it would spend budget on luck. Basis: {basis}."
        )

    wins = _fold_wins(new_folds, best_folds)
    if wins is not None:
        n_wins, n_folds = wins
        if n_wins <= FOLD_MAJORITY * n_folds:
            return False, (
                f"rejected: the mean gain {gain:+.4f} clears the noise band "
                f"({threshold:.4f}) but the candidate wins only {n_wins}/{n_folds} "
                "folds — the gain is concentrated in one or two lucky folds, "
                "which is exactly the pattern that regressed earlier runs."
            )
        fold_note = f" and it wins {n_wins}/{n_folds} folds"
    else:
        fold_note = " (fold vectors not comparable, so the mean test decided alone)"

    return True, (
        f"accepted: local gain {gain:+.4f} over {best_score:.4f} exceeds "
        f"{k:g} x {band:.4f} = {threshold:.4f}{fold_note}. Basis: {basis}."
    )


# --------------------------------------------------------------------------- #
# shrinkage (winner's curse)
# --------------------------------------------------------------------------- #
def shrunk_estimate(
    lb_score: float | None, local_score: float | None, cal: Calibration
) -> float:
    """Expected private-test-B score of a submission, shrunk toward calibration.

    WHY THIS EXISTS
    ---------------
    Final ranking does not use the leaderboard we optimised against: our best
    submission on private test **A** is re-scored on a *different* private test
    **B**. Every observed A-score is ``truth + noise``. Taking the raw argmax
    over A therefore preferentially selects submissions whose noise term was
    large and positive — the winner's curse. Those are precisely the ones that
    regress on B, and the regression is largest exactly when the noise band is
    widest.

    The fix is the James-Stein insight: an extreme observation is best estimated
    not by itself but by pulling it toward an independent prediction. Our
    independent prediction is the calibration line's estimate from local CV,
    ``slope * local + intercept``, which is built from *all* submissions and from
    a validation set with many more samples than the leaderboard's noise allows.

    Decomposition (empirical Bayes)::

        observed residual variance = true signal variance + LB noise variance
        signal_var = max(residual_std^2 - noise^2, 0)
        lambda     = signal_var / (signal_var + noise^2)
        shrunk     = prediction + lambda * (lb - prediction)

    ``lambda`` is the weight kept on the raw leaderboard score, so the shrinkage
    toward the prediction, ``1 - lambda``, grows as the noise band grows —
    exactly the requested behaviour. With zero noise nothing is shrunk; with
    noise dominating the residual spread the leaderboard score is treated as
    pure luck and the prediction wins.

    Degenerate inputs are handled by refusing to invent information: with no
    usable calibration or no local score the raw leaderboard score is returned
    unchanged, and with no leaderboard score the prediction is returned.
    """
    have_lb = _finite(lb_score)
    pred = predicted_lb(local_score, cal)

    if pred is None:
        if have_lb:
            return float(lb_score)  # type: ignore[arg-type]
        raise ValueError("shrunk_estimate needs a leaderboard score or a calibration")
    if not have_lb:
        return float(pred)

    lb = float(lb_score)  # type: ignore[arg-type]
    noise = lb_units_noise(cal)
    resid = float(cal.residual_std) if _finite(cal.residual_std) else None

    if noise <= 0:
        return lb  # no measured scoring noise -> nothing to shrink away
    if resid is None or cal.n_points < MIN_SHRINK_POINTS:
        # Not enough calibration points for the residual spread to mean anything
        # (see MIN_SHRINK_POINTS). Keep the observation rather than shrinking it
        # onto a line we cannot yet vouch for.
        return lb

    signal_var = max(resid * resid - noise * noise, 0.0)
    lam = signal_var / (signal_var + noise * noise) if signal_var > 0 else 0.0
    lam = min(1.0, max(0.0, lam))
    return pred + lam * (lb - pred)


def predicted_lb(local_score: float | None, cal: Calibration) -> float | None:
    """Leaderboard score the calibration predicts for ``local_score``.

    ``None`` when no prediction is possible. When the slope is unidentifiable
    (all calibration points share one local score) the intercept alone is the
    best available prediction: "the leaderboard sits at this level".
    """
    if cal.slope is not None and _finite(cal.slope) and _finite(cal.intercept):
        if not _finite(local_score):
            return None
        return float(cal.slope) * float(local_score) + float(cal.intercept)
    if _finite(cal.intercept) and cal.n_points >= 1:
        return float(cal.intercept)
    return None


# --------------------------------------------------------------------------- #
# the ledger itself
# --------------------------------------------------------------------------- #
class ExperimentLedger:
    """Append-only view of every experiment, with the report/Manager renderings.

    The blackboard already stores experiments; this class adds the *reading* of
    them — ranking, the significance verdict against the current incumbent, and
    the compact markdown table that appears both in Manager context (where token
    budget matters) and in the technical report (where auditability matters).
    """

    def __init__(self, bb: Blackboard):
        self.bb = bb

    # ------------------------------------------------------------- mutation
    def add(self, exp: Experiment) -> Experiment:
        """Append one experiment and mirror it into the event log."""
        self.bb.add_experiment(exp)
        self.bb.event(
            "experiment",
            exp_id=exp.exp_id,
            candidate_id=exp.candidate_id,
            role=exp.role,
            local_score=exp.local_score,
            accepted=exp.accepted,
            reject_reason=exp.reject_reason[:200],
        )
        return exp

    # -------------------------------------------------------------- reading
    def all(self) -> list[Experiment]:
        return self.bb.get_experiments()

    def scored(self) -> list[Experiment]:
        return [e for e in self.all() if _finite(e.local_score)]

    def best(self) -> Experiment | None:
        """Highest-scoring experiment (scores are maximize-oriented)."""
        scored = self.scored()
        if not scored:
            return None
        return max(scored, key=lambda e: float(e.local_score))  # type: ignore[arg-type]

    def top_k(self, k: int = 5) -> list[Experiment]:
        return sorted(
            self.scored(),
            key=lambda e: float(e.local_score),  # type: ignore[arg-type]
            reverse=True,
        )[: max(0, k)]

    # ------------------------------------------------------------- verdicts
    def evaluate(self, exp: Experiment, k: float = SIGNIFICANCE_K) -> tuple[bool, str]:
        """Run the significance gate for ``exp`` against the current incumbent."""
        best = self.best()
        cal = self.bb.get_calibration()
        return significant_gain(
            exp.local_score,
            best.local_score if best else None,
            exp.local_std,
            cal,
            new_folds=exp.folds,
            best_folds=best.folds if best else None,
            k=k,
        )

    # ------------------------------------------------------------ rendering
    def as_table(self, limit: int = 40, desc_chars: int = 64) -> str:
        """Compact markdown table, newest last, oldest rows dropped first.

        Kept deliberately narrow: this string is pasted into Manager context on
        every step, so each extra column costs tokens on every LLM call.
        """
        exps = self.all()
        if not exps:
            return "_No experiments recorded._"
        shown = exps[-limit:] if limit and len(exps) > limit else exps
        head = (
            "| # | exp | candidate | role | description | local | ±std | folds "
            "| min | verdict |\n"
            "|---|---|---|---|---|---|---|---|---|---|"
        )
        rows = [head]
        offset = len(exps) - len(shown)
        for i, e in enumerate(shown, start=offset + 1):
            desc = (e.description or "").replace("|", "/").replace("\n", " ")
            if len(desc) > desc_chars:
                desc = desc[: desc_chars - 1] + "…"
            score = f"{float(e.local_score):.4f}" if _finite(e.local_score) else "unknown"
            std = f"{float(e.local_std):.4f}" if _finite(e.local_std) else "-"
            folds = str(len(e.folds or [])) if e.folds else "-"
            mins = f"{e.runtime_s / 60:.1f}" if e.runtime_s else "-"
            if e.accepted:
                verdict = "accepted"
            else:
                reason = (e.reject_reason or "rejected").replace("|", "/").replace("\n", " ")
                verdict = reason[:80]
            rows.append(
                f"| {i} | {e.exp_id} | {e.candidate_id or '-'} | {e.role or '-'} | "
                f"{desc or '-'} | {score} | {std} | {folds} | {mins} | {verdict} |"
            )
        if offset:
            rows.append(f"| … | _{offset} earlier experiments omitted_ | | | | | | | | |")
        return "\n".join(rows)

    def summary_stats(self) -> dict:
        """Counts the report and the Manager both need. Never estimates."""
        exps = self.all()
        scored = self.scored()
        accepted = [e for e in exps if e.accepted]
        best = self.best()
        locals_ = [float(e.local_score) for e in scored]  # type: ignore[arg-type]
        runtime_s = sum(float(e.runtime_s or 0.0) for e in exps)
        span_s = 0.0
        if len(exps) >= 2:
            times = [float(e.created_at) for e in exps if _finite(e.created_at)]
            if len(times) >= 2:
                span_s = max(times) - min(times)
        return {
            "n_experiments": len(exps),
            "n_scored": len(scored),
            "n_accepted": len(accepted),
            "acceptance_rate": (round(len(accepted) / len(exps), 3) if exps else None),
            "n_candidates": len({e.candidate_id for e in exps if e.candidate_id}),
            "best_local": (float(best.local_score) if best else None),  # type: ignore[arg-type]
            "best_exp_id": (best.exp_id if best else None),
            "best_candidate_id": (best.candidate_id if best else None),
            "median_local": (median(locals_) if locals_ else None),
            "worst_local": (min(locals_) if locals_ else None),
            "total_experiment_minutes": round(runtime_s / 60.0, 1),
            "span_minutes": round(span_s / 60.0, 1),
        }
