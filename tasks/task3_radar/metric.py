"""Best-effort replication of the radar competition's ``RadarMetric``.

READ THIS BEFORE TRUSTING A NUMBER FROM THIS FILE.
==================================================

**KNOWN** — established from the baseline notebook and from the data itself:

* The task is dense per-pixel multi-class classification over a
  ``50 x 181`` map — 9,050 predictions per sample.
* The label channel on disk holds ``-1, 0, 1, 2, 3``.  The baseline shifts by
  ``+1`` to get ``0..4`` for ``CrossEntropyLoss``, the model's last conv has
  ``out_channels=5``, and inference shifts back by ``-1`` before writing the
  submission.  So there are **5 classes**, and class ``0`` (raw ``-1``) is the
  empty/background class.
* Submission format is ``filename,pixel_0,...,pixel_9049`` in **raw** label
  space, one row per test file.
* Class balance, measured over 30 training files: class 0 ≈ **97.66%**,
  class 1 ≈ 0.15%, class 2 ≈ 0.17%, class 3 ≈ 0.08%, class 4 ≈ 1.94%.
  A "predict background everywhere" submission therefore reaches ≈0.977 pixel
  accuracy while being completely useless.

**ASSUMED** — the competition's ``RadarMetric`` is a *custom* metric and its
source was not published with the mirror.  The aggregation is a guess:

* Whether it is mean IoU, macro-F1, or something else entirely.
* Whether the background class is included in the average.  This matters
  enormously: including class 0 lifts every score by ~0.2 and compresses the
  differences between models, because class 0 is trivially easy.
* Whether the average is over pixels pooled across all files (micro), or
  computed per file and then averaged (macro-over-files).

So this module computes **all of them** and exposes them side by side.  The
:data:`PRIMARY` constant names the one we optimise until we know better.

.. warning::
   :data:`PRIMARY` is a *working assumption*, not a fact.  Before it is trusted,
   spend one **calibration submission**: submit a prediction whose value under
   each candidate aggregation is known and different (e.g. all-background, then
   a fixed single-class map), and compare the leaderboard score against the
   table :func:`calibration_table` prints.  That is one submission out of 50 to
   remove a systematic risk of optimising the wrong objective for six hours —
   the cheapest information the swarm can buy.  Record the outcome as a
   ``ProbeResult`` with ``probe_kind="calibration"``.

Pure numpy, no torch: the whole point is that scoring works wherever the swarm
runs, not only where the model trains.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

N_CLASSES = 5
BACKGROUND_CLASS = 0

#: What we optimise until a calibration submission says otherwise.
#:
#: macro-F1 over the four foreground classes: it ignores the trivially-large
#: background, weights the rare classes equally with the common ones, and moves
#: when a model actually finds objects.  Mean IoU ranks models almost
#: identically here, so an aggregation mistake costs ranking accuracy, not
#: direction — but see the warning above.
PRIMARY = "macro_f1_fg"

#: Every aggregation this module can report, so a caller can log all of them
#: and re-decide later without re-running anything.
AGGREGATIONS = (
    "pixel_accuracy",
    "mean_iou",
    "mean_iou_fg",
    "macro_f1",
    "macro_f1_fg",
    "weighted_f1",
)


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES
) -> np.ndarray:
    """``(n_classes, n_classes)`` int64 counts, rows = truth, cols = prediction.

    Labels are expected in **training space** (``0..4``).  Values outside the
    range are dropped rather than crashing, and the count of dropped pixels is
    recoverable as ``y_true.size - cm.sum()`` — a silent shift-by-one bug is
    the single most likely error in this task, and it shows up here as a
    non-zero drop count.
    """
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_pred).ravel().astype(np.int64)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: truth {t.shape} vs prediction {p.shape}")
    ok = (t >= 0) & (t < n_classes) & (p >= 0) & (p < n_classes)
    idx = t[ok] * n_classes + p[ok]
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """Intersection-over-union per class; ``nan`` for classes absent from both.

    ``nan`` rather than 0 is deliberate: a class that appears in neither truth
    nor prediction has an undefined IoU, and averaging a fake 0 into the mean
    punishes a model for a class the sample never contained.
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """F1 per class; ``nan`` for classes absent from both truth and prediction."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    return f1


def _nanmean(values: np.ndarray) -> float:
    vals = values[~np.isnan(values)]
    return float(vals.mean()) if vals.size else 0.0


def radar_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = N_CLASSES,
) -> dict[str, Any]:
    """Score dense predictions under every candidate aggregation at once.

    Inputs are in **training space** (``0..4``); shift raw ``-1..3`` labels by
    ``+1`` first (``data.load_sample`` already does).  Shapes may be
    ``(H, W)``, ``(N, H, W)`` or flat — everything is raveled, so this is the
    *micro* (pixel-pooled) view.  Use :func:`per_sample_metric` for the
    macro-over-files variant.

    Returns every entry of :data:`AGGREGATIONS`, plus ``primary`` (the value of
    :data:`PRIMARY`), the per-class arrays, the confusion matrix as a list, and
    ``support`` — the true pixel count per class, which is what tells you a
    per-class score of ``nan`` is "class absent" and not "model broken".
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    iou = per_class_iou(cm)
    f1 = per_class_f1(cm)
    support = cm.sum(axis=1).astype(np.int64)
    total = int(cm.sum())

    fg = [c for c in range(n_classes) if c != BACKGROUND_CLASS]
    weights = support.astype(np.float64)
    valid = ~np.isnan(f1)
    wsum = weights[valid].sum()

    out: dict[str, Any] = {
        "pixel_accuracy": float(np.diag(cm).sum() / total) if total else 0.0,
        "mean_iou": _nanmean(iou),
        "mean_iou_fg": _nanmean(iou[fg]),
        "macro_f1": _nanmean(f1),
        "macro_f1_fg": _nanmean(f1[fg]),
        "weighted_f1": float((f1[valid] * weights[valid]).sum() / wsum) if wsum else 0.0,
        "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
        "per_class_f1": [None if np.isnan(v) else float(v) for v in f1],
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "n_pixels": total,
        "n_dropped": int(np.asarray(y_true).size - total),
    }
    out["primary"] = out[PRIMARY]
    out["primary_name"] = PRIMARY
    out["caveat"] = (
        "aggregation ASSUMED; verify against the leaderboard with one "
        "calibration submission before trusting the absolute value"
    )
    return out


def per_sample_metric(
    y_true: Sequence[np.ndarray] | np.ndarray,
    y_pred: Sequence[np.ndarray] | np.ndarray,
    n_classes: int = N_CLASSES,
) -> dict[str, Any]:
    """Macro-over-files variant: score each sample, then average.

    The other plausible reading of "mean per-class IoU".  It differs from the
    pixel-pooled view whenever samples differ in how much of each class they
    contain — which they do here — so a large gap between the two is itself a
    signal about which one the leaderboard is using.
    """
    trues = list(y_true)
    preds = list(y_pred)
    if len(trues) != len(preds):
        raise ValueError(f"{len(trues)} truth samples vs {len(preds)} prediction samples")
    if not trues:
        return {k: 0.0 for k in AGGREGATIONS} | {"n_samples": 0}

    rows = [radar_metric(t, p, n_classes) for t, p in zip(trues, preds)]
    out: dict[str, Any] = {k: float(np.mean([r[k] for r in rows])) for k in AGGREGATIONS}
    out["n_samples"] = len(rows)
    out["primary"] = out[PRIMARY]
    out["primary_name"] = PRIMARY
    out["aggregation"] = "macro over files"
    return out


def constant_prediction_scores(
    y_true: np.ndarray, n_classes: int = N_CLASSES
) -> dict[int, dict[str, float]]:
    """Score an all-class-``c`` prediction, for every ``c``.

    These are the numbers a **calibration submission** is compared against.
    Predicting a single constant class is free to produce, is guaranteed valid,
    and takes a different value under each candidate aggregation — so one
    leaderboard score for a known constant map tells us which formula the
    grader is running.
    """
    y_true = np.asarray(y_true)
    out: dict[int, dict[str, float]] = {}
    for c in range(n_classes):
        m = radar_metric(y_true, np.full(y_true.shape, c, dtype=np.int64), n_classes)
        out[c] = {k: float(m[k]) for k in AGGREGATIONS}
    return out


def calibration_table(y_true: np.ndarray, n_classes: int = N_CLASSES) -> str:
    """Human-readable table of constant-prediction scores.

    Print this, submit the matching constant map once, and read off which
    column the leaderboard agrees with.
    """
    scores = constant_prediction_scores(y_true, n_classes)
    head = f"{'predict':>8} | " + " | ".join(f"{k:>14}" for k in AGGREGATIONS)
    lines = [head, "-" * len(head)]
    for c, row in scores.items():
        lines.append(
            f"{'all ' + str(c):>8} | " + " | ".join(f"{row[k]:14.6f}" for k in AGGREGATIONS)
        )
    lines.append("")
    lines.append(
        "Submit one constant map, compare the leaderboard score to a column, "
        "then set PRIMARY to the column that matches."
    )
    return "\n".join(lines)


def format_report(metrics: dict[str, Any]) -> str:
    """One-screen summary for the blackboard / technical report."""
    lines = [
        f"PRIMARY ({metrics.get('primary_name', PRIMARY)}) : "
        f"{metrics.get('primary', 0.0):.6f}   [ASSUMED aggregation]",
    ]
    for k in AGGREGATIONS:
        if k in metrics:
            lines.append(f"  {k:<16}: {metrics[k]:.6f}")
    if "per_class_f1" in metrics:
        per = ", ".join(
            f"{i}:{'--' if v is None else format(v, '.3f')}"
            for i, v in enumerate(metrics["per_class_f1"])
        )
        lines.append(f"  per-class F1     : {per}")
        lines.append(f"  support          : {metrics.get('support')}")
    if metrics.get("n_dropped"):
        lines.append(
            f"  !! {metrics['n_dropped']} pixels out of range 0..{N_CLASSES - 1} — "
            f"check the +1/-1 label shift"
        )
    return "\n".join(lines)
