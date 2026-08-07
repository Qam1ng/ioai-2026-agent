#!/usr/bin/env python3
"""Reconnaissance — runs BEFORE any solver, posts findings to the facts board.

Deterministic, no LLM. The premise: the biggest score jumps on these tasks come
from noticing something structural about the data, not from a better model. A
solver that starts modelling before anyone has hashed the test set is spending
its 2 hours in the wrong place.

Checks (each degrades to a no-op if it doesn't apply):
  * file inventory — what is actually in input/, and how big
  * exact duplicates BETWEEN train and test (content hash) — the classic leak
  * exact duplicates WITHIN train — inflates CV, changes fold design
  * label distribution — imbalance, tiny classes, unexpected cardinality
  * id/index ↔ target correlation — ordering leakage
  * submission format — exact columns and row count expected

Usage: python -m native.scripts.recon --input input/ --workspace .
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

MEDIA = {".wav", ".mp3", ".flac", ".ogg", ".png", ".jpg", ".jpeg", ".npy", ".tif"}


def _digest(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def _human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def inventory(root: Path, out: list[str]) -> None:
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        out.append(("env", f"input dir {root} is EMPTY — data not downloaded"))
        return
    by_ext = Counter(p.suffix.lower() for p in files)
    total = sum(p.stat().st_size for p in files)
    top = ", ".join(f"{ext or '(none)'}x{n}" for ext, n in by_ext.most_common(6))
    out.append(("data", f"input: {len(files)} files, {_human(total)}, {top}"))


def media_duplicates(root: Path, out: list[str]) -> None:
    """Content-hash every media file; report train/test overlap and dupes."""
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA]
    if not (8 <= len(files) <= 60000):
        return
    digests: dict[str, list[Path]] = {}
    for p in files:
        try:
            digests.setdefault(_digest(p), []).append(p)
        except OSError:
            continue

    def side(p: Path) -> str:
        s = str(p).lower()
        if "test" in s:
            return "test"
        if "train" in s:
            return "train"
        return "?"

    cross = sum(1 for ps in digests.values()
                if len({side(p) for p in ps}) > 1 and "?" not in {side(p) for p in ps})
    within = sum(len(ps) - 1 for ps in digests.values() if len(ps) > 1)
    if cross:
        out.append(("data", f"LEAK: {cross} media files are byte-identical between "
                            f"train and test — labels may be copyable directly"))
    if within and not cross:
        out.append(("data", f"{within} duplicate media files within the data "
                            f"({len(files)} total) — dedupe before building folds"))
    if not within and not cross:
        out.append(("data", f"no byte-identical duplicates among {len(files)} media files"))


def tabular(root: Path, out: list[str]) -> None:
    try:
        import pandas as pd
    except ImportError:
        return
    csvs = sorted(p for p in root.rglob("*.csv") if p.stat().st_size < 500 << 20)
    if not csvs:
        return
    frames: dict[str, "pd.DataFrame"] = {}
    for p in csvs:
        try:
            frames[p.name] = pd.read_csv(p)
        except Exception:  # noqa: BLE001 — malformed csv is itself a finding
            out.append(("format", f"{p.name} failed to parse as CSV"))
    for name, df in frames.items():
        out.append(("data", f"{name}: {len(df)} rows x {len(df.columns)} cols "
                            f"[{', '.join(map(str, df.columns[:8]))}]"))

    sample = next((d for n, d in frames.items() if "sample" in n.lower()
                   and "sub" in n.lower()), None)
    if sample is not None:
        out.append(("format", f"submission must be {len(sample)} rows, columns "
                              f"exactly {list(sample.columns)} in this order"))

    train = next((d for n, d in frames.items() if n.lower().startswith("train")), None)
    test = next((d for n, d in frames.items() if n.lower().startswith("test")), None)

    if train is not None:
        # label distribution — last column not shared with test is the usual target
        cand = [c for c in train.columns if test is None or c not in test.columns]
        for col in cand[:2]:
            vc = train[col].value_counts()
            if 1 < len(vc) <= 60:
                rare = (vc < 5).sum()
                out.append(("data", f"target '{col}': {len(vc)} classes, "
                                    f"min={vc.min()} max={vc.max()}"
                                    + (f", {rare} classes with <5 samples" if rare else "")))
            elif len(vc) > 60:
                out.append(("data", f"'{col}' is continuous-ish "
                                    f"({len(vc)} distinct, min={train[col].min()}, "
                                    f"max={train[col].max()})"))
        # ordering leakage: does row order predict the target?
        for col in cand[:1]:
            try:
                import numpy as np
                y = pd.factorize(train[col])[0].astype(float)
                r = abs(float(np.corrcoef(np.arange(len(y)), y)[0, 1]))
                if r > 0.2:
                    out.append(("data", f"LEAK: row index correlates with '{col}' "
                                        f"(|r|={r:.2f}) — data is ordered by target; "
                                        f"never split without shuffling"))
            except Exception:  # noqa: BLE001
                pass

    if train is not None and test is not None:
        shared = [c for c in train.columns if c in test.columns]
        if shared:
            th = train[shared].astype(str).agg("|".join, axis=1).map(hash)
            sh = test[shared].astype(str).agg("|".join, axis=1).map(hash)
            n = len(set(th) & set(sh))
            if n:
                out.append(("data", f"LEAK: {n} test rows are identical to train rows "
                                    f"on all {len(shared)} shared columns"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--src", default="recon")
    a = ap.parse_args()

    root, ws = Path(a.input), Path(a.workspace)
    out: list[tuple[str, str]] = []
    for check in (inventory, media_duplicates, tabular):
        try:
            check(root, out)
        except Exception as e:  # noqa: BLE001 — a broken check must not block the run
            out.append(("env", f"recon check {check.__name__} failed: "
                               f"{type(e).__name__}: {e}"))

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from native import facts
    b = facts.init(ws)
    for kind, text in out:
        b.post(kind, text, src=a.src, layer="task")

    (ws / "recon.md").write_text(
        "# Reconnaissance (deterministic, pre-modelling)\n\n"
        + "\n".join(f"- **{k}** — {t}" for k, t in out) + "\n")
    print(f"[recon] {len(out)} findings -> facts board + recon.md")
    for k, t in out:
        print(f"  {k:<8} {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
