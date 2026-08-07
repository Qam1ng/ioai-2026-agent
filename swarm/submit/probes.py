"""Probes: cheap CPU submissions bought for *information*, not for score.

A probe is a deliberately trivial kernel — CPU only, therefore **zero GPU
quota**, and the GPU quota (30 h/week shared by all three problems of a day)
is the binding budget, not the 50 submissions per problem. Spending a
submission slot to learn the shape of the hidden test set is almost always a
better trade than spending it on one more tuning increment.

Why probes are *not* score chasing, stated plainly: the final ranking takes our
best submission on private test A and **re-scores it on private test B**.
Selecting hard on A therefore buys the winner's curse — the submission that
tops A is the one whose A-noise was luckiest, and that luck does not transfer
to B. So probes are aimed at facts that are stable across A and B (test-set
size, base rate, distribution shift, the leaderboard's own noise) and never at
"which of these two near-identical variants scores 0.001 higher".

The five probes:

``constant_baseline``  one constant prediction -> reveals the base rate.
``score_granularity``  several scores -> reveals N via the score quantum 1/N.
``split_shift``        train on half A vs half B -> reveals train/test shift.
``noise_floor``        same pipeline, different seeds -> reveals LB variance,
                       which becomes the significance threshold everything
                       downstream is gated on.
``calibration``        deliberately different-strength runs -> the local<->LB
                       regression, i.e. how optimistic our CV is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..schemas import ProbeResult, TaskCard

# ---------------------------------------------------------------------------
# generated kernel code
# ---------------------------------------------------------------------------

#: Every probe kernel starts here. Paths are DISCOVERED, never hardcoded:
#: competition data mounts at /kaggle/input/competitions/<slug>/, one level
#: deeper than most examples assume (memory: kaggle-kernel-input-paths).
_PREAMBLE = '''\
"""Probe kernel — {kind}.

Hypothesis under test: {hypothesis}
This kernel exists to buy information about the hidden test set, not score.
CPU only; no internet; stdlib + pandas/numpy/sklearn only.
"""
import os, sys, json, random
import numpy as np
import pandas as pd

OUT = "/kaggle/working/submission.csv"
ROOT = "/kaggle/input"


def find_files():
    """Locate the sample submission and any train table under /kaggle/input."""
    sample, train = None, None
    for root, _dirs, files in os.walk(ROOT):
        for f in files:
            low = f.lower()
            p = os.path.join(root, f)
            if sample is None and low in ("sample_submission.csv", "submission.csv",
                                          "sampleentry.csv", "sample_submission.csv.gz"):
                sample = p
            if train is None and low in ("train.csv", "train.csv.gz", "training.csv"):
                train = p
    if sample is None:
        raise RuntimeError("no sample submission found under " + ROOT)
    return sample, train


SAMPLE, TRAIN = find_files()
sub = pd.read_csv(SAMPLE)
train_df = pd.read_csv(TRAIN) if TRAIN else None
TARGET = sub.columns[-1]
print("sample:", SAMPLE, "rows:", len(sub), "target col:", TARGET)
print("train:", TRAIN, "rows:", (0 if train_df is None else len(train_df)))
'''

_WRITE_OUT = '''

sub.to_csv(OUT, index=False)
print("wrote", OUT, "rows:", len(sub))
print("PROBE_META", json.dumps({{"n_rows": int(len(sub)), "kind": "{kind}"}}))
'''

_CONSTANT_BODY = '''
CONST = {const!r}
if CONST is None and train_df is not None and TARGET in train_df.columns:
    # Majority class (classification) or mean (regression): the score this
    # earns IS the test-set base rate for an accuracy-like metric.
    col = train_df[TARGET]
    if col.dtype.kind in "ifc" and col.nunique() > 20:
        CONST = float(col.mean())
    else:
        CONST = col.mode().iloc[0]
if CONST is None:
    CONST = 0
print("constant prediction:", CONST)
sub[TARGET] = CONST
'''

#: Shared fallback model. Deliberately dumb and fast — a probe must not need a
#: GPU, and its absolute score is uninteresting; only score *differences*
#: between probe variants carry the signal.
_FIT_BODY = '''
FRACTION = {fraction!r}
HALF = {half!r}
SEED = {seed!r}

random.seed(SEED); np.random.seed(SEED)


def constant_fallback(reason):
    print("falling back to constant prediction:", reason)
    if train_df is not None and TARGET in train_df.columns:
        col = train_df[TARGET]
        if col.dtype.kind in "ifc" and col.nunique() > 20:
            sub[TARGET] = float(col.mean())
        else:
            sub[TARGET] = col.mode().iloc[0]
    else:
        sub[TARGET] = 0


def select_rows(df):
    """Half-split for split_shift, fractional subsample for calibration."""
    idx = np.arange(len(df))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    if HALF == "first":
        idx = idx[: len(idx) // 2]
    elif HALF == "second":
        idx = idx[len(idx) // 2 :]
    if FRACTION is not None and 0 < FRACTION < 1:
        idx = idx[: max(1, int(len(idx) * FRACTION))]
    return df.iloc[np.sort(idx)]


def run():
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    if train_df is None or TARGET not in train_df.columns:
        return constant_fallback("no usable train table")
    id_col = sub.columns[0]
    feats = [c for c in train_df.columns if c not in (TARGET, id_col)]
    if not feats:
        return constant_fallback("no feature columns")
    tr = select_rows(train_df)
    y = tr[TARGET]
    X = pd.get_dummies(tr[feats], dummy_na=True)
    # Test features come from the same table when the probe has no test frame:
    # the point of these probes is the DIFFERENCE between variants, so an
    # imperfect feature build is acceptable as long as it is identical across
    # variants.
    test_src = None
    for root, _dirs, files in os.walk(ROOT):
        for f in files:
            if f.lower() in ("test.csv", "test.csv.gz"):
                test_src = os.path.join(root, f)
                break
        if test_src:
            break
    if test_src is None:
        return constant_fallback("no test table")
    te = pd.read_csv(test_src)
    Xte = pd.get_dummies(te[[c for c in feats if c in te.columns]], dummy_na=True)
    X, Xte = X.align(Xte, join="left", axis=1, fill_value=0)
    is_clf = y.dtype.kind not in "fc" or y.nunique() <= 20
    model = (HistGradientBoostingClassifier(random_state=SEED, max_iter=60)
             if is_clf else
             HistGradientBoostingRegressor(random_state=SEED, max_iter=60))
    model.fit(X.astype(float), y)
    sub[TARGET] = model.predict(Xte.astype(float))[: len(sub)]


try:
    run()
except Exception as exc:  # a probe must always produce a valid submission
    constant_fallback(repr(exc))
'''


# ---------------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------------
def infer_n_from_scores(
    scores: list[float], max_n: int = 100000, tol: float = 5e-6
) -> int | None:
    """Smallest test-set size N consistent with every observed score.

    An accuracy-like metric on N test rows can only take the values k/N, so a
    handful of distinct leaderboard scores pins N down exactly. Knowing N is
    worth a probe: it tells us how many rows one flipped prediction is worth
    (1/N), which is the true resolution of the leaderboard and therefore the
    smallest gain worth submitting for.

    ``tol`` defaults to 5e-6 because Kaggle reports scores rounded to five
    decimals — a tighter tolerance rejects the true N.

    Returns ``None`` when the scores carry no information (all 0/1) or when no
    denominator up to ``max_n`` explains them (metric is not of the k/N form).

    A continued-fraction shortcut (each score's own minimal denominator must
    divide N) is tempting and much faster, but Kaggle rounds to five decimals,
    which destroys the exact denominators and makes the shortcut return a wrong
    multiple. The straight denominator sweep is ~1e5 cheap float checks and is
    correct under rounding, so that is what runs.
    """
    vals = [float(s) for s in scores if s is not None]
    if not vals:
        return None
    if all(abs(v - round(v)) <= tol for v in vals):
        return None  # only 0s and 1s: every N fits, nothing learned
    if any(v < -tol or v > 1 + tol for v in vals):
        return None  # not a bounded rate; the k/N argument does not apply

    for n in range(1, int(max_n) + 1):
        if all(abs(v * n - round(v * n)) <= tol * n + 1e-9 for v in vals):
            return n
    return None


def linear_calibration(points: list[tuple[float, float]]) -> dict[str, float]:
    """Least-squares LB ~ a + b * local over ``(local, lb)`` pairs."""
    pts = [(float(a), float(b)) for a, b in points if a is not None and b is not None]
    n = len(pts)
    if n < 2:
        return {"n_points": n}
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    if sxx <= 0:
        return {"n_points": n, "gap": mx - my}
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [p[1] - (intercept + slope * p[0]) for p in pts]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return {
        "n_points": n,
        "slope": slope,
        "intercept": intercept,
        "residual_std": var**0.5,
        "gap": mx - my,  # positive == local CV is optimistic
    }


def _std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


# ---------------------------------------------------------------------------
# probe definition
# ---------------------------------------------------------------------------
@dataclass
class Probe:
    """One information-buying submission recipe.

    ``build_code(task_card, **opts)`` returns a complete kernel ``.py``.
    ``interpret(lb_score, context)`` turns the returned leaderboard score into
    a :class:`~swarm.schemas.ProbeResult` whose ``inference`` is a sentence a
    human (or the Manager) can act on without rederiving the arithmetic.
    """

    kind: str
    hypothesis: str
    build_code: Callable[..., str]
    interpret: Callable[..., ProbeResult]
    accelerator: str = "cpu"  # probes are CPU by definition: zero GPU quota
    purpose: str = ""
    options: dict[str, Any] = field(default_factory=dict)


def _kernel(kind: str, hypothesis: str, body: str) -> str:
    return (
        _PREAMBLE.format(kind=kind, hypothesis=hypothesis)
        + body
        + _WRITE_OUT.format(kind=kind)
    )


# ----------------------------------------------------------- constant_baseline
def _constant_code(task_card: TaskCard | None = None, const: Any = None, **_: Any) -> str:
    return _kernel(
        "constant_baseline",
        "a single constant prediction scores the test-set base rate",
        _CONSTANT_BODY.format(const=const),
    )


def _constant_interpret(lb_score: float | None, context: dict | None = None) -> ProbeResult:
    ctx = context or {}
    local = ctx.get("local_base_rate")
    numeric: dict[str, Any] = {"lb_score": lb_score}
    if lb_score is None:
        inference = "constant probe not scored yet; base rate unknown"
    elif local is None:
        inference = (
            f"a constant prediction scores {lb_score:.5f} on the leaderboard, so the "
            f"test-set majority share is about {lb_score:.3f} — any model below this "
            "is worse than guessing"
        )
    else:
        drift = lb_score - float(local)
        numeric["local_base_rate"] = float(local)
        numeric["base_rate_drift"] = drift
        inference = (
            f"test base rate {lb_score:.5f} vs train {float(local):.5f} "
            f"(drift {drift:+.5f}); "
            + (
                "class balance is comparable, stratified CV is trustworthy"
                if abs(drift) < 0.02
                else "class balance differs between train and test — reweight or "
                "recalibrate thresholds"
            )
        )
    return ProbeResult(
        probe_kind="constant",
        hypothesis="a constant prediction scores the test-set base rate",
        lb_score=lb_score,
        inference=inference,
        numeric=numeric,
    )


# ------------------------------------------------------------ score_granularity
def _granularity_code(task_card: TaskCard | None = None, const: Any = None, **_: Any) -> str:
    # Granularity needs *several distinct* scores; the cheapest source is a
    # constant kernel with a different constant each time.
    return _kernel(
        "score_granularity",
        "leaderboard scores are multiples of 1/N, so N is recoverable",
        _CONSTANT_BODY.format(const=const),
    )


def _granularity_interpret(
    lb_score: float | None, context: dict | None = None
) -> ProbeResult:
    ctx = context or {}
    scores = [s for s in (ctx.get("scores") or []) if s is not None]
    if lb_score is not None and lb_score not in scores:
        scores = scores + [lb_score]
    n = infer_n_from_scores(scores, max_n=int(ctx.get("max_n", 100000)))
    numeric: dict[str, Any] = {"scores": scores, "n_scores": len(scores), "n_test": n}
    if n is None:
        inference = (
            f"{len(scores)} score(s) do not pin down a test-set size — the metric is "
            "probably not of the k/N form (AUC, RMSE, log loss); treat leaderboard "
            "resolution as unknown"
        )
    else:
        numeric["score_quantum"] = 1.0 / n
        inference = (
            f"test set has about N={n} rows: every observed score is a multiple of "
            f"1/{n}={1.0/n:.6f}. One flipped prediction moves the leaderboard by "
            f"{1.0/n:.6f}, so any 'improvement' smaller than that is not real"
        )
    return ProbeResult(
        probe_kind="granularity",
        hypothesis="leaderboard scores are multiples of 1/N",
        lb_score=lb_score,
        inference=inference,
        numeric=numeric,
    )


# ------------------------------------------------------------------ split_shift
def _split_shift_code(
    task_card: TaskCard | None = None, half: str = "first", seed: int = 0, **_: Any
) -> str:
    if half not in ("first", "second"):
        raise ValueError("split_shift half must be 'first' or 'second'")
    return _kernel(
        "split_shift",
        "the two halves of the training data score differently on the test set",
        _FIT_BODY.format(fraction=None, half=half, seed=seed),
    )


def _split_shift_interpret(
    lb_score: float | None, context: dict | None = None
) -> ProbeResult:
    ctx = context or {}
    a = ctx.get("first_half_lb")
    b = ctx.get("second_half_lb", lb_score)
    band = float(ctx.get("noise_band", 0.0) or 0.0)
    numeric: dict[str, Any] = {"first_half_lb": a, "second_half_lb": b, "noise_band": band}
    if a is None or b is None:
        inference = "split-shift needs both halves scored; only one is in yet"
    else:
        delta = float(b) - float(a)
        numeric["delta"] = delta
        if abs(delta) <= max(band, 1e-9):
            inference = (
                f"halves agree ({float(a):.5f} vs {float(b):.5f}, delta {delta:+.5f} "
                f"within the {band:.5f} noise band): no detectable train/test shift, "
                "random CV folds are sound"
            )
        else:
            inference = (
                f"halves disagree ({float(a):.5f} vs {float(b):.5f}, delta {delta:+.5f} "
                f"exceeds the {band:.5f} noise band): the training data is not "
                "exchangeable — fold by the grouping variable and prefer the half "
                "that resembles test"
            )
    return ProbeResult(
        probe_kind="split_shift",
        hypothesis="training halves score differently, revealing distribution shift",
        lb_score=lb_score,
        inference=inference,
        numeric=numeric,
    )


# ------------------------------------------------------------------ noise_floor
def _noise_floor_code(
    task_card: TaskCard | None = None, seed: int = 0, **_: Any
) -> str:
    return _kernel(
        "noise_floor",
        "identical pipeline, different seed — the score spread is the LB's own noise",
        _FIT_BODY.format(fraction=None, half="all", seed=seed),
    )


def _noise_floor_interpret(
    lb_score: float | None, context: dict | None = None
) -> ProbeResult:
    ctx = context or {}
    scores = [float(s) for s in (ctx.get("scores") or []) if s is not None]
    if lb_score is not None and lb_score not in scores:
        scores = scores + [float(lb_score)]
    sigma = _std(scores)
    spread = (max(scores) - min(scores)) if len(scores) >= 2 else 0.0
    numeric: dict[str, Any] = {
        "scores": scores,
        "n_seeds": len(scores),
        "noise_std": sigma,
        "spread": spread,
        # The band the milestone gate uses. Std of a 2-3 point sample is a poor
        # estimator, so the full observed spread is the honest floor.
        "noise_band": max(sigma, spread / 2.0),
    }
    if len(scores) < 2:
        inference = "noise floor needs at least two seeds; only one scored so far"
    else:
        band = numeric["noise_band"]
        inference = (
            f"{len(scores)} identical runs span {spread:.5f} (sd {sigma:.5f}); the "
            f"leaderboard's own noise band is {band:.5f}. Do not spend a submission "
            f"on a local gain smaller than this — it is indistinguishable from luck"
        )
    return ProbeResult(
        probe_kind="noise_floor",
        hypothesis="repeat runs measure the leaderboard's own variance",
        lb_score=lb_score,
        inference=inference,
        numeric=numeric,
    )


# ------------------------------------------------------------------ calibration
def _calibration_code(
    task_card: TaskCard | None = None, fraction: float = 1.0, seed: int = 0, **_: Any
) -> str:
    if not 0 < float(fraction) <= 1:
        raise ValueError("calibration fraction must be in (0, 1]")
    return _kernel(
        "calibration",
        "deliberately weaker and stronger runs map local CV onto the leaderboard",
        _FIT_BODY.format(fraction=float(fraction), half="all", seed=seed),
    )


def _calibration_interpret(
    lb_score: float | None, context: dict | None = None
) -> ProbeResult:
    ctx = context or {}
    pairs = [
        (float(a), float(b))
        for a, b in (ctx.get("pairs") or [])
        if a is not None and b is not None
    ]
    local = ctx.get("local_score")
    if local is not None and lb_score is not None:
        pairs = pairs + [(float(local), float(lb_score))]
    fit = linear_calibration(pairs)
    numeric: dict[str, Any] = {"pairs": pairs, **fit}
    if fit.get("n_points", 0) < 2:
        inference = (
            "calibration needs at least two (local, leaderboard) pairs of clearly "
            "different strength; one point cannot separate offset from slope"
        )
    else:
        slope = fit.get("slope")
        gap = fit.get("gap", 0.0) or 0.0
        resid = fit.get("residual_std", 0.0) or 0.0
        direction = "optimistic" if gap > 0 else "pessimistic"
        inference = (
            f"local CV is {direction} by {abs(gap):.5f} on average; "
            f"LB ~ {fit.get('intercept', 0.0):.5f} + {slope:.3f} x local "
            f"(residual sd {resid:.5f}). "
            + (
                "slope near 1 means local gains transfer roughly one-for-one"
                if slope is not None and 0.7 <= slope <= 1.3
                else "slope far from 1 means local gains do NOT transfer "
                "proportionally — trust the leaderboard over local deltas"
            )
        )
    return ProbeResult(
        probe_kind="calibration",
        hypothesis="local CV and leaderboard are linearly related",
        lb_score=lb_score,
        inference=inference,
        numeric=numeric,
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
PROBES: dict[str, Probe] = {
    "constant_baseline": Probe(
        kind="constant_baseline",
        hypothesis="a single constant prediction scores the test-set base rate",
        build_code=_constant_code,
        interpret=_constant_interpret,
        purpose="learn the majority-class share / target mean of the hidden test set",
    ),
    "score_granularity": Probe(
        kind="score_granularity",
        hypothesis="leaderboard scores are multiples of 1/N, so N is recoverable",
        build_code=_granularity_code,
        interpret=_granularity_interpret,
        purpose="learn the hidden test-set size, hence the smallest real improvement",
    ),
    "split_shift": Probe(
        kind="split_shift",
        hypothesis="the two halves of the training data score differently on test",
        build_code=_split_shift_code,
        interpret=_split_shift_interpret,
        purpose="detect train/test distribution shift before trusting random CV folds",
    ),
    "noise_floor": Probe(
        kind="noise_floor",
        hypothesis="identical pipeline, different seed — the spread is the LB's noise",
        build_code=_noise_floor_code,
        interpret=_noise_floor_interpret,
        purpose="measure the leaderboard's own variance; sets the significance threshold",
    ),
    "calibration": Probe(
        kind="calibration",
        hypothesis="local CV and the leaderboard are linearly related",
        build_code=_calibration_code,
        interpret=_calibration_interpret,
        purpose="fit local->LB so local gains can be translated into expected LB gains",
    ),
}


def get_probe(kind: str) -> Probe:
    try:
        return PROBES[kind]
    except KeyError:
        raise KeyError(f"unknown probe {kind!r}; known: {sorted(PROBES)}") from None


def list_probes() -> list[str]:
    return sorted(PROBES)


def probe_plan(task_card: TaskCard | None = None) -> list[tuple[str, dict]]:
    """The default probe sequence, cheapest-and-most-informative first.

    Ordering is deliberate: the noise floor has to exist before any milestone
    gate can be meaningful, and the granularity probe reuses the constant
    probes' scores for free rather than buying its own.
    """
    return [
        ("constant_baseline", {}),
        ("constant_baseline", {"const": 0}),
        ("score_granularity", {"const": 1}),
        ("noise_floor", {"seed": 0}),
        ("noise_floor", {"seed": 1}),
        ("split_shift", {"half": "first"}),
        ("split_shift", {"half": "second"}),
        ("calibration", {"fraction": 0.25}),
        ("calibration", {"fraction": 1.0}),
    ]
