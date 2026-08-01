#!/usr/bin/env python3
"""Is the thing we are about to send the thing we measured, and did it learn?

Two checks the rest of the harness cannot make, both borrowed from a teammate's
system after their run notes showed why they matter.

PROVENANCE. `evaluate.py` scores `out/oof.npy`; `check_format.py` inspects
`out/submission.csv`. Nothing has ever tied the two together, so a candidate
could be promoted on one model's out-of-fold predictions and submitted with
another model's test predictions, and no part of the system would notice. This
records what was scored and refuses a submission whose files have moved since.

ORDER DEPENDENCE. On the audio task their strongest public score came from
reading the official submission ordering, inferring that the rows fall in
contiguous class blocks, and Viterbi-decoding the sequence — 0.86 to 0.988
without the model improving at all. They caught it with an ordering-invariance
test and recorded it as audit-only rather than banking it. A pipeline that
exploits row order has not learned the task and will not survive a reshuffled
test set; on IOAI it is also the kind of thing that gets a submission thrown out.
So: look for predictions that track row position, and say so.

    python -m native.scripts.integrity --candidate solver_a --workspace .
    python -m native.scripts.integrity --candidate solver_a --record --workspace .
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEAL = "integrity.json"


def sha(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def files_of(ws: Path, cand: str) -> dict[str, Path]:
    out = ws / cand / "out"
    return {"oof": out / "oof.npy", "submission": out / "submission.csv"}


def record(ws: Path, cand: str, score: float | None = None) -> dict:
    """Seal what was scored, so a later swap is visible."""
    seal = {}
    p = ws / SEAL
    if p.exists():
        seal = json.loads(p.read_text())
    entry = {"score": score}
    for name, f in files_of(ws, cand).items():
        entry[name] = sha(f) if f.exists() else None
    seal[cand] = entry
    p.write_text(json.dumps(seal, indent=2))
    return entry


def check_provenance(ws: Path, cand: str) -> list[str]:
    p = ws / SEAL
    if not p.exists():
        return []
    seal = json.loads(p.read_text()).get(cand)
    if not seal:
        return []
    bad = []
    for name, f in files_of(ws, cand).items():
        was, now = seal.get(name), (sha(f) if f.exists() else None)
        if was and now and was != now:
            bad.append(f"{name} changed after it was scored — the number this "
                       f"candidate was promoted on came from a different file")
        if was and now is None:
            bad.append(f"{name} has disappeared since it was scored")
    return bad


def check_order_dependence(ws: Path, cand: str) -> list[str]:
    """Do the predictions track row position rather than the input?

    Deliberately a report, not a verdict. Genuine tasks do have ordered data —
    what matters is whether the *predictions* carry order information the inputs
    do not, and only someone who knows the task can settle that.
    """
    import numpy as np
    import pandas as pd

    f = files_of(ws, cand)["submission"]
    if not f.exists():
        return []
    try:
        df = pd.read_csv(f)
    except Exception:  # noqa: BLE001
        return []
    if len(df) < 20 or len(df.columns) < 2:
        return []

    notes = []
    col = pd.to_numeric(df[df.columns[1]], errors="coerce")
    if col.notna().sum() < 20:
        return []
    v = col.to_numpy(dtype=float)
    idx = np.arange(len(v))

    r = abs(float(np.corrcoef(idx, np.nan_to_num(v))[0, 1]))
    if r > 0.35:
        notes.append(f"prediction correlates with row position (|r|={r:.2f}) — "
                     f"check the model is not reading the ordering")

    # Contiguous runs of one value are the signature of a sequence decoder run
    # over the submission order, which is what the audio exploit did.
    vals = np.nan_to_num(v)
    if len(np.unique(vals)) <= max(2, len(vals) // 10):
        runs = 1 + int((vals[1:] != vals[:-1]).sum())
        expected = len(vals) * (1 - 1 / max(len(np.unique(vals)), 2))
        if runs < expected * 0.35:
            notes.append(f"predictions fall into {runs} contiguous blocks over "
                         f"{len(vals)} rows — far fewer than shuffled data would "
                         f"give. If the row order carries label structure, this "
                         f"submission is exploiting it rather than predicting")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--score", type=float, default=None)
    a = ap.parse_args()
    ws = Path(a.workspace)
    if a.record:
        print(json.dumps(record(ws, a.candidate, a.score), indent=2))
        return 0
    fatal = check_provenance(ws, a.candidate)
    warn = check_order_dependence(ws, a.candidate)
    print(json.dumps({"candidate": a.candidate, "ok": not fatal,
                      "fatal": fatal, "warnings": warn}, indent=2))
    return 0 if not fatal else 1


if __name__ == "__main__":
    raise SystemExit(main())
