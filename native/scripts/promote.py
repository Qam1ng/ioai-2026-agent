#!/usr/bin/env python3
"""No-regression gate + Last-Known-Good.

This is also where "machine-checkable failure criteria" lives. We deliberately
do NOT make solvers declare a structured hypothesis and a predicate before each
experiment — that is a format cage around the agent kernel, and preserving the
kernel is the whole bet. Instead the check happens one level down, at the
*candidate* level, where it is cheap and unambiguous: a candidate replaces LKG
only if this script's independently computed score is higher. How the solver got
there is its own business.

LKG has two tiers, because they fail differently:
  * ``local``     — best script-computed OOF. Cheap, available immediately.
  * ``confirmed`` — best *Kaggle-scored* submission. Slow, but it is the only
                    number that has survived contact with the real test set.

At the deadline the watchdog submits ``confirmed`` if one exists, else
``local``. In a 2-hour window with three problems sharing one GPU quota, the
catastrophic outcome is not a mediocre score — it is zero.

Usage:
  python -m native.scripts.promote --candidate solver_a --workspace .
  python -m native.scripts.promote --confirm solver_a --lb 0.913 --workspace .
  python -m native.scripts.promote --show --workspace .
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from native.scripts.evaluate import evaluate

STATE = "LKG/state.json"


def load(ws: Path) -> dict:
    p = ws / STATE
    return json.loads(p.read_text()) if p.exists() else {"local": None, "confirmed": None}


def save(ws: Path, st: dict) -> None:
    p = ws / STATE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=2))


def snapshot(ws: Path, cand: str, tier: str) -> str:
    """Immutable copy. LKG must not be a pointer into a directory a solver
    is still editing."""
    dst = ws / "LKG" / tier
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    src = ws / cand
    for rel in ("out", "kernel"):
        if (src / rel).exists():
            shutil.copytree(src / rel, dst / rel, dirs_exist_ok=True)
    return str(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--candidate", default=None, help="promote by local OOF score")
    ap.add_argument("--confirm", default=None, help="candidate whose LB score is known")
    ap.add_argument("--lb", type=float, default=None, help="Kaggle-reported score")
    ap.add_argument("--higher-is-better", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    ws = Path(a.workspace)
    st = load(ws)
    better = (lambda new, old: new > old) if a.higher_is_better else (lambda n, o: n < o)

    if a.show:
        print(json.dumps(st, indent=2))
        return 0

    if a.confirm is not None:
        if a.lb is None:
            raise SystemExit("--confirm requires --lb")
        cur = st.get("confirmed")
        if cur is None or better(a.lb, cur["lb"]):
            path = snapshot(ws, a.confirm, "confirmed")
            st["confirmed"] = {"candidate": a.confirm, "lb": a.lb,
                               "ts": int(time.time()), "path": path}
            save(ws, st)
            print(json.dumps({"decision": "PROMOTED", "tier": "confirmed",
                              "candidate": a.confirm, "lb": a.lb,
                              "previous": cur["lb"] if cur else None}))
        else:
            print(json.dumps({"decision": "REJECTED", "tier": "confirmed",
                              "candidate": a.confirm, "lb": a.lb,
                              "incumbent": cur["lb"],
                              "reason": "leaderboard score did not improve"}))
        return 0

    if a.candidate is None:
        raise SystemExit("pass --candidate, --confirm, or --show")

    # Score it here, from out/oof.npy. Deliberately NOT read from score.json:
    # that file lives inside the solver's own directory, so trusting it would
    # make "self-reported scores are never believed" a comment rather than a
    # property. (The self-test catches this: a hand-written score.json claiming
    # 0.99 must not promote.)
    sc = evaluate(ws, a.candidate)
    if sc.get("status") != "ok":
        print(json.dumps({"decision": "REJECTED", "candidate": a.candidate,
                          "reason": f"evaluate status={sc.get('status')}: "
                                    f"{sc.get('detail', '')}"}))
        return 1

    cur = st.get("local")
    if cur is None or better(sc["mean"], cur["mean"]):
        path = snapshot(ws, a.candidate, "local")
        st["local"] = {"candidate": a.candidate, "mean": sc["mean"],
                       "std": sc["std"], "ts": int(time.time()), "path": path}
        save(ws, st)
        print(json.dumps({"decision": "PROMOTED", "tier": "local",
                          "candidate": a.candidate, "mean": sc["mean"],
                          "std": sc["std"],
                          "previous": cur["mean"] if cur else None}))
    else:
        print(json.dumps({"decision": "REJECTED", "tier": "local",
                          "candidate": a.candidate, "mean": sc["mean"],
                          "incumbent": cur["mean"],
                          "reason": "did not beat LKG on the shared folds"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
