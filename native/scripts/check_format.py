#!/usr/bin/env python3
"""Deterministic submission validation — the floor beneath the evaluator's review.

The evaluator can catch things no script would know to look for: on the chicken
task it read the organisers' grading code and found that a single negative pixel
voids the entire submission. That judgement is worth having, which is why it
gates submission.

But a gate that can hang is a way to score zero. So everything here is
mechanical and always runs, whether or not the evaluator answers: shape, columns,
order, ids, and values that no metric can survive. If the review times out the
submitter proceeds on this alone.

    python -m native.scripts.check_format --candidate solver_a --workspace .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_sample(ws: Path) -> Path | None:
    hits = [p for p in (ws / "input").rglob("*.csv")
            if "sample" in p.name.lower() and "sub" in p.name.lower()]
    return hits[0] if hits else None


def check(ws: Path, candidate: str) -> dict:
    """Look at a submission and report what is wrong with it.

    Two kinds of finding, and conflating them cost a whole run. `fatal` is
    something no competition could accept: the file is missing, or unreadable,
    or disagrees with a sample submission that exists. `warnings` are things
    that depend on rules this script does not know — on the radar task every
    pixel column legitimately contains -1, because -1 *is* the background label,
    and treating that as a defect blocked a 0.9816 candidate eighteen times
    while the agent that could have said so was never asked.

    So this returns evidence. The verdict is the evaluator's.
    """
    import numpy as np
    import pandas as pd

    sub = ws / candidate / "out" / "submission.csv"
    fatal: list[str] = []
    warn: list[str] = []
    if not sub.exists():
        return {"candidate": candidate, "ok": False, "fatal": [f"{sub} does not exist"],
                "warnings": [], "problems": [f"{sub} does not exist"]}
    try:
        df = pd.read_csv(sub)
    except Exception as e:  # noqa: BLE001
        m = f"unreadable as CSV: {type(e).__name__}: {e}"
        return {"candidate": candidate, "ok": False, "fatal": [m],
                "warnings": [], "problems": [m]}

    sample_path = find_sample(ws)
    if sample_path is not None:
        # With a sample to compare against, shape disagreements are facts.
        s = pd.read_csv(sample_path)
        if list(df.columns) != list(s.columns):
            fatal.append(f"columns {list(df.columns)[:4]}... != required "
                         f"{list(s.columns)[:4]}... (order matters)")
        if len(df) != len(s):
            fatal.append(f"{len(df)} rows, sample has {len(s)}")
        idc = s.columns[0]
        if idc in df.columns and idc in s.columns:
            missing = set(s[idc]) - set(df[idc])
            extra = set(df[idc]) - set(s[idc])
            if missing:
                fatal.append(f"{len(missing)} ids missing, e.g. "
                             f"{sorted(map(str, missing))[:3]}")
            if extra:
                fatal.append(f"{len(extra)} ids not in the sample, e.g. "
                             f"{sorted(map(str, extra))[:3]}")
    else:
        warn.append("no sample_submission in the data, so column names, order "
                    "and row count could not be checked against anything")

    # NaN and inf break any scorer. A negative value does not: it is the
    # background label on at least one task we have run, so it is reported and
    # left for someone who knows the rules to judge.
    neg = [c for c in df.columns[1:]
           if pd.to_numeric(df[c], errors="coerce").dropna().lt(0).any()]
    nan = [c for c in df.columns if df[c].isna().any()]
    inf = [c for c in df.columns[1:]
           if not np.isfinite(pd.to_numeric(df[c], errors="coerce").dropna()).all()]
    if nan:
        fatal.append(f"{len(nan)} column(s) contain NaN, e.g. {nan[:3]}")
    if inf:
        fatal.append(f"{len(inf)} column(s) contain inf, e.g. {inf[:3]}")
    if neg:
        mn = min(pd.to_numeric(df[c], errors="coerce").min() for c in neg[:20])
        warn.append(f"{len(neg)} column(s) contain negative values (min {mn:.4g}) "
                    f"— fine if this task's label set includes them, fatal if not")

    return {"candidate": candidate, "ok": not fatal,
            "fatal": fatal, "warnings": warn,
            "problems": fatal + warn,          # kept for older callers
            "rows": len(df), "columns": list(df.columns)[:6]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--workspace", default=".")
    a = ap.parse_args()
    r = check(Path(a.workspace), a.candidate)
    print(json.dumps(r, indent=2))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
