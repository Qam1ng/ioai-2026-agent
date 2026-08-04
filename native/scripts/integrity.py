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
import time
from pathlib import Path

from native import evidence

SEAL = "integrity.json"
CANDIDATE_EVIDENCE = "evidence/candidates"


def sha(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def files_of(ws: Path, cand: str) -> dict[str, Path]:
    out = ws / cand / "out"
    return {
        "oof": out / "oof.npy",
        "submission": out / "submission.csv",
        "folds": ws / "folds.json",
        "metric": ws / "metric.py",
    }


def code_manifest(ws: Path, cand: str) -> dict[str, str]:
    root = ws / cand
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "out" in path.relative_to(root).parts:
            continue
        if path.suffix.lower() not in (".py", ".json", ".yaml", ".yml", ".toml", ".sh"):
            continue
        out[str(path.relative_to(root))] = sha(path)
    return out


def evidence_path(ws: Path, cand: str) -> Path:
    return ws / CANDIDATE_EVIDENCE / f"{cand}.json"


def record(ws: Path, cand: str, score: float | None = None) -> dict:
    """Seal what was scored, so a later swap is visible."""
    seal = {}
    p = ws / SEAL
    if p.exists():
        seal = json.loads(p.read_text())
    entry = {"schema_version": 2, "candidate": cand, "score": score,
             "recorded_at_unix": int(time.time())}
    for name, f in files_of(ws, cand).items():
        entry[name] = sha(f) if f.exists() else None
    entry["code"] = code_manifest(ws, cand)
    contract = ws / evidence.RUN_CONTRACT
    entry["run_contract"] = sha(contract) if contract.exists() else None
    entry["tainted"] = evidence.is_tainted(ws, cand)
    receipt = ws / cand / "out" / "sample_locality_receipt.json"
    entry["sample_locality_receipt"] = sha(receipt) if receipt.exists() else None
    seal[cand] = entry
    evidence.write_json(p, seal)
    evidence.write_json(evidence_path(ws, cand), entry)
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
        if was is None and now is not None:
            bad.append(f"{name} appeared after the candidate was scored")
        elif was and now and was != now:
            bad.append(f"{name} changed after it was scored — the number this "
                       f"candidate was promoted on came from a different file")
        if was and now is None:
            bad.append(f"{name} has disappeared since it was scored")
    if seal.get("code") != code_manifest(ws, cand):
        bad.append("candidate code/config changed after it was scored")
    contract = ws / evidence.RUN_CONTRACT
    now_contract = sha(contract) if contract.exists() else None
    if seal.get("run_contract") != now_contract:
        bad.append("run contract changed after candidate scoring")
    if evidence.is_tainted(ws, cand) or seal.get("tainted"):
        bad.append("candidate process is tainted by forbidden audit access")
    return bad


def check_sample_locality(ws: Path, cand: str, *, required: bool) -> tuple[list[str], list[str]]:
    from native.scripts.sample_locality import check

    result = check(ws, cand)
    if result.get("valid"):
        return [], []
    message = "; ".join(result.get("errors", [])) or "sample-locality failed"
    if required:
        return [message], []
    return [], [message]


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
    ap.add_argument("--require-sample-locality", action="store_true")
    a = ap.parse_args()
    ws = Path(a.workspace)
    if a.record:
        print(json.dumps(record(ws, a.candidate, a.score), indent=2))
        return 0
    fatal = check_provenance(ws, a.candidate)
    warn = check_order_dependence(ws, a.candidate)
    locality_fatal, locality_warn = check_sample_locality(
        ws, a.candidate, required=a.require_sample_locality)
    fatal += locality_fatal
    warn += locality_warn
    print(json.dumps({"candidate": a.candidate, "ok": not fatal,
                      "fatal": fatal, "warnings": warn,
                      "evidence_manifest": str(evidence_path(ws, a.candidate))},
                     indent=2))
    return 0 if not fatal else 1


if __name__ == "__main__":
    raise SystemExit(main())
