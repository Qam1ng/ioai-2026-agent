#!/usr/bin/env python3
"""Validate a split, however it was produced.

The harness owns one invariant about validation and no more: every candidate in
a run is scored on the same split, and that split does not change once anything
has been scored on it. It has no business owning the *method*.

That distinction was got wrong at first. `folds.py` generated the split itself,
which meant every task whose shape it had not anticipated needed another flag —
continuous targets, then temporal ordering, then a single holdout — and the next
task would have needed a fourth. Detection labels are not one row per image;
segmentation labels are not in a table at all; some competitions ship their own
validation split. No generator covers that, and a generator that tries becomes a
treadmill.

So `folds.py` is now a convenience for the common tabular case, and anything it
cannot express the evaluator writes directly. This file is what the harness
actually enforces.

Contract for `folds.json`:

    ids     list, one entry per evaluation unit, in the order predictions take
    fold    list, same length. >= 0 is a scored fold; -1 means train-only
    labels  list, same length — ground truth passed to metric.score
    scheme  short string naming how it was cut
    reason  why, in one line; required whenever scheme is not a plain k-fold

    python -m native.scripts.checkfolds --workspace .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = ("ids", "fold", "labels", "scheme")


def validate_spec(spec: dict) -> list[str]:
    bad: list[str] = []
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        return [f"missing key(s): {missing}"]

    n = len(spec["ids"])
    for k in ("fold", "labels"):
        if len(spec[k]) != n:
            bad.append(f"'{k}' has {len(spec[k])} entries but 'ids' has {n}")
    if bad:
        return bad
    if n == 0:
        return ["the split is empty"]

    try:
        fold = [int(f) for f in spec["fold"]]
    except (TypeError, ValueError):
        return ["'fold' must be integers"]

    scored = [f for f in fold if f >= 0]
    if not scored:
        bad.append("no unit is scored — every fold is -1")
        return bad

    present = sorted(set(scored))
    if present != list(range(len(present))):
        bad.append(f"scored folds are {present}, expected 0..{len(present) - 1} "
                   f"with no gaps")
    else:
        sizes = [scored.count(f) for f in present]
        if len(present) < 2 and len(scored) == n:
            bad.append("a single fold covering every unit is not a validation "
                       "split — nothing is held out")
        if min(sizes) == 0:
            bad.append("a scored fold is empty")
        if len(present) > 1 and max(sizes) > 0.6 * len(scored):
            bad.append(f"one fold holds {max(sizes)}/{len(scored)} scored units")

    if len(set(map(str, spec["ids"]))) != n:
        bad.append("'ids' contains duplicates — units must be distinguishable")
    # The named built-ins explain themselves. Anything else — a hand-written
    # split for a task whose shape the generator does not cover — has to say
    # why: a split nobody can explain is not one anybody should trust, and it
    # is also the only place where a favourable cut could hide.
    if spec.get("scheme") not in ("kfold", "stratified", "grouped") \
            and not str(spec.get("reason", "")).strip():
        bad.append(f"scheme '{spec.get('scheme')}' needs a 'reason'")
    if any(int(f) < 0 for f in spec["fold"]) \
            and not str(spec.get("reason", "")).strip():
        bad.append("units are marked train-only (-1) with no stated reason; "
                   "holding data out of scoring needs a justification")
    return bad


def describe(spec: dict) -> str:
    fold = [int(f) for f in spec["fold"]]
    scored = [f for f in fold if f >= 0]
    k = len(set(scored))
    held = len(fold) - len(scored)
    return (f"scheme={spec.get('scheme')} units={len(fold)} scored_folds={k} "
            + (f"train_only={held} " if held else "")
            + (f"| {spec.get('reason')}" if spec.get("reason") else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    p = Path(a.file) if a.file else Path(a.workspace) / "folds.json"
    if not p.exists():
        print(json.dumps({"ok": False, "problems": [f"{p} does not exist"]}))
        return 1
    try:
        spec = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False,
                          "problems": [f"unreadable: {type(e).__name__}: {e}"]}))
        return 1
    bad = validate_spec(spec)
    out = {"ok": not bad, "problems": bad}
    if not bad:
        out["summary"] = describe(spec)
    print(json.dumps(out, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
