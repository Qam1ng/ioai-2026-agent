#!/usr/bin/env python3
"""Fold generation — the harness owns the split, solvers do not.

Our worst measured failure was CV optimism: 0.9156 local -> 0.78095 leaderboard.
A solver that invents its own split will, given enough attempts, invent a
favourable one. So the split is generated once, before any solver starts, from
a fixed seed, and every candidate is scored on it.

Usage:
  python -m native.scripts.folds --labels input/train.csv --target label \\
      --id id --k 5 --workspace .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEED = 20260805  # competition day 1; fixed forever, never tuned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="CSV with the training labels")
    ap.add_argument("--target", required=True, help="target column")
    ap.add_argument("--id", default=None, help="id column (default: row index)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--group", default=None,
                    help="group column — rows sharing a value never split across folds")
    ap.add_argument("--custom", default=None,
                    help="JSON file with {'fold': [...], 'reason': '...'} — an "
                         "explicit assignment for schemes the built-ins cannot "
                         "express, e.g. a temporal split for time series")
    ap.add_argument("--reason", default=None,
                    help="why a custom split is needed; recorded in folds.json")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing folds.json (see below — don't)")
    a = ap.parse_args()

    ws_early = Path(a.workspace)
    existing = ws_early / "folds.json"
    if existing.exists() and not a.force:
        # The danger was never the scheme, it was re-cutting the split after
        # seeing which candidate it favours. A custom split chosen up front is
        # honest; the same split rewritten at minute twenty is not, and no
        # scheme restriction would catch that. So the split is immutable
        # instead, and every score in the run is comparable because of it.
        raise SystemExit(
            f"{existing} already exists and the split is frozen once written — "
            f"every candidate in this run is scored on it, so changing it now "
            f"would make the scores incomparable. Pass --force only if nothing "
            f"has been scored yet.")

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                         StratifiedGroupKFold, KFold)

    df = pd.read_csv(a.labels)
    if a.target not in df.columns:
        raise SystemExit(f"target {a.target!r} not in {list(df.columns)}")
    y = df[a.target].to_numpy()
    ids = df[a.id].tolist() if a.id else list(range(len(df)))
    groups = df[a.group].to_numpy() if a.group else None

    # Decide discreteness from the dtype first, then from cardinality. Counting
    # distinct values alone misfires on small continuous targets — fifty rows of
    # floats have fifty distinct values, and a threshold phrased as "at most
    # fifty classes" reads that as a fifty-class problem and hands it to a
    # stratified splitter, which refuses it.
    n_classes = len(pd.unique(y))
    if pd.api.types.is_float_dtype(df[a.target]):
        # Floats are continuous unless the values genuinely repeat as labels.
        discrete = 1 < n_classes <= 20 and n_classes <= len(y) // 5
    else:
        discrete = 1 < n_classes <= max(2, len(y) // 3)

    if a.custom:
        # An escape hatch, because the three built-in schemes cannot express
        # everything: a time series must be cut chronologically, and scoring a
        # temporal task on a shuffled split flatters every model on it. Refusing
        # such a task outright would be worse than allowing a declared split.
        spec = json.loads(Path(a.custom).read_text())
        fold_of = np.asarray(spec["fold"], dtype=int)
        scheme, reason = "custom", (a.reason or spec.get("reason") or "")
        if not reason:
            raise SystemExit("--custom requires --reason; a split nobody can "
                             "explain is not one anybody should trust")
        a.k = int(max([f for f in fold_of.tolist() if f >= 0], default=-1)) + 1
    else:
        # Stratify on discrete targets; plain KFold on continuous ones.
        if groups is not None:
            sp = (StratifiedGroupKFold(a.k, shuffle=True, random_state=SEED)
                  if discrete else GroupKFold(a.k))
            splits = sp.split(np.zeros(len(y)), y, groups)
        elif discrete:
            splits = StratifiedKFold(a.k, shuffle=True,
                                     random_state=SEED).split(np.zeros(len(y)), y)
        else:
            splits = KFold(a.k, shuffle=True,
                           random_state=SEED).split(np.zeros(len(y)))
        scheme = ("grouped" if groups is not None else
                  "stratified" if discrete else "kfold")
        reason = ""
        fold_of = np.full(len(y), -1, dtype=int)
        for i, (_, va) in enumerate(splits):
            fold_of[va] = i

    import time as _time
    ws = Path(a.workspace)
    spec = {
        "seed": SEED, "k": a.k, "n": len(y),
        "target": a.target, "id_col": a.id, "group_col": a.group,
        "scheme": scheme, "reason": reason, "created": int(_time.time()),
        "stratified": scheme == "stratified", "grouped": groups is not None,
        "ids": ids,
        "fold": fold_of.tolist(),
        "labels": [x.item() if hasattr(x, "item") else x for x in y],
    }
    from native.scripts.checkfolds import describe, validate_spec
    if bad := validate_spec(spec):
        raise SystemExit(f"split rejected: {'; '.join(bad)}")
    (ws / "folds.json").write_text(json.dumps(spec))

    scored = [f for f in fold_of.tolist() if f >= 0]
    sizes = [scored.count(f) for f in sorted(set(scored))]
    held = len(fold_of) - len(scored)
    readme = (f"# Validation protocol (frozen)\n\n"
              f"- scheme **{scheme}**"
              + (f" — {reason}\n" if reason else "\n")
              + f"- seed **{SEED}**, n=**{len(y)}**\n"
              f"- grouped: {groups is not None}\n"
              f"- scored fold sizes: {sizes}"
              + (f", train-only units: {held}\n" if held else "\n") + "\n"
              f"`folds.json` carries `ids`, `fold`, `labels` aligned by position.\n"
              f"Write out-of-fold predictions as `out/oof.npy`, one row per entry\n"
              f"in that order. Any other split is not comparable and will not be\n"
              f"scored.\n\nThis file is written once. It is not re-cut after\n"
              f"candidates have been scored on it.\n")
    (ws / "VALIDATION.md").write_text(readme)

    print(f"[folds] {describe(spec)} -> folds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
