#!/usr/bin/env python3
"""Independent scoring — a solver's self-reported number is never believed.

Every candidate must write `out/oof.npy`: out-of-fold predictions, one row per
entry of `folds.json`, in that order. This script re-computes the official
metric from those predictions itself. If a solver claims 0.94 and this says
0.86, 0.86 is the number.

`metric.py` is produced once by the evaluator (it needs to *read* the evaluation
page, which is a judgement task), self-tested against a known expected value,
then frozen. From that point on scoring is a script.

Contract for metric.py:
    def score(y_true, y_pred) -> float | dict   # dict must contain "score"

Usage: python -m native.scripts.evaluate --candidate solver_a --workspace .
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_metric(path: Path):
    spec = importlib.util.spec_from_file_location("ioai_metric", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import metric from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "score"):
        raise SystemExit(f"{path} must define score(y_true, y_pred)")
    return mod.score


def as_float(v) -> float:
    return float(v["score"]) if isinstance(v, dict) else float(v)


def evaluate(ws: Path, candidate: str, metric: str = "metric.py") -> dict:
    """Score a candidate from its raw predictions.

    Reads only `out/oof.npy` — never `score.json`, which a solver could simply
    write itself. promote.py calls this rather than trusting any file that lives
    inside a solver's own directory.
    """
    import numpy as np

    ws = Path(ws)
    folds = json.loads((ws / "folds.json").read_text())
    y = np.asarray(folds["labels"])
    fold = np.asarray(folds["fold"])
    fn = load_metric(ws / metric)

    oof = ws / candidate / "out" / "oof.npy"
    if not oof.exists():
        return {"candidate": candidate, "status": "no_oof",
                "detail": f"{oof} missing — candidate does not exist"}
    pred = np.load(oof, allow_pickle=False)
    if len(pred) != len(y):
        return {"candidate": candidate, "status": "shape_mismatch",
                "detail": f"oof has {len(pred)} rows, folds.json has {len(y)}"}

    # Scored folds are whatever the split actually contains. `-1` marks a unit
    # that is training-only and never scored, which is how a single holdout or a
    # competition's own validation split is expressed — the harness does not
    # need to know which of those it is looking at.
    scored = sorted({int(f) for f in fold.tolist() if int(f) >= 0})
    if not scored:
        return {"candidate": candidate, "status": "no_scored_folds",
                "detail": "every unit in folds.json is marked train-only (-1)"}

    per_fold = []
    for k in scored:
        m = fold == k
        try:
            per_fold.append(as_float(fn(y[m], pred[m])))
        except Exception as e:  # noqa: BLE001
            return {"candidate": candidate, "status": "metric_error",
                    "detail": f"fold {k}: {type(e).__name__}: {e}"}

    keep = fold >= 0
    arr = np.asarray(per_fold, dtype=float)
    return {"candidate": candidate, "status": "ok",
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "per_fold": [round(x, 6) for x in per_fold],
            "pooled": round(as_float(fn(y[keep], pred[keep])), 6),
            "n": int(keep.sum()), "n_total": int(len(y)),
            "folds_scored": len(scored)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="solver dir name")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--metric", default="metric.py")
    a = ap.parse_args()

    ws = Path(a.workspace)
    result = evaluate(ws, a.candidate, a.metric)
    # Written for the solver's convenience only; nothing downstream reads it.
    out = ws / a.candidate / "out"
    if out.exists():
        (out / "score.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
