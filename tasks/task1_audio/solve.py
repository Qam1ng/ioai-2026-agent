"""Task-1 solver v1: freeze encoder + old 16-class head, learn 13 new rows.

Runs on cached embeddings (see features.py). The 29-way head = [frozen old 16
rows ; trainable new 13 rows]. Training uses a 29-way cross-entropy over all
labeled clips, so the new rows learn to (a) win on their own classes and
(b) stay below the frozen old logits on old clips — anti-forgetting in one
logit space, no separate calibration.

Reports the official metric on a stratified local holdout (the cheap verifier),
then retrains on all data and writes submission.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IN = HERE / "input"
CACHE = HERE / "cache"
sys.path.insert(0, str(HERE.parent.parent / "src"))
from metric import score  # noqa: E402  (official 0.5*Acc_old + 0.5*Acc_new)

NUM_OLD, NUM_NEW, NUM_CLASSES = 16, 13, 29


def _rows(name):
    with open(IN / name) as f:
        return list(csv.DictReader(f))


def load_cache():
    embs = np.load(CACHE / "embeddings.npy")
    paths = (CACHE / "paths.txt").read_text().splitlines()
    p2i = {p: i for i, p in enumerate(paths)}
    W_old = np.load(CACHE / "old_head_W.npy")  # [16,768]
    b_old = np.load(CACHE / "old_head_b.npy")  # [16]
    return embs, p2i, W_old, b_old


def build_labeled(embs, p2i):
    X, y = [], []
    for name in ("train.csv", "fine_tune.csv"):
        for r in _rows(name):
            X.append(embs[p2i[r["path"]]])
            y.append(int(r["target"]))
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def stratified_split(y, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac))) if len(idx) > 1 else 0
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])
    return np.array(tr), np.array(va)


def train_new_head(X, y, W_old, b_old, *, steps=400, lr=1e-2, wd=1e-4,
                   new_bias_init=-4.0, seed=0):
    """Freeze old head; train 13 new rows with class-weighted 29-way CE."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, device=dev)
    yt = torch.tensor(y, device=dev)
    Wo = torch.tensor(W_old, device=dev)          # [16,768] frozen
    bo = torch.tensor(b_old, device=dev)          # [16]     frozen

    Wn = torch.zeros(NUM_NEW, X.shape[1], device=dev, requires_grad=True)
    bn = torch.full((NUM_NEW,), new_bias_init, device=dev, requires_grad=True)
    torch.nn.init.normal_(Wn, std=0.02)

    # inverse-frequency class weights (balance old vs new + rare classes)
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    cw = np.where(counts > 0, counts.sum() / (counts + 1e-6), 0.0)
    cw = cw / cw[cw > 0].mean()
    cwt = torch.tensor(cw, dtype=torch.float32, device=dev)

    opt = torch.optim.Adam([Wn, bn], lr=lr, weight_decay=wd)
    for _ in range(steps):
        old_logits = Xt @ Wo.t() + bo            # [N,16] constant
        new_logits = Xt @ Wn.t() + bn            # [N,13]
        logits = torch.cat([old_logits, new_logits], dim=1)
        loss = F.cross_entropy(logits, yt, weight=cwt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return Wn.detach().cpu().numpy(), bn.detach().cpu().numpy()


def predict(X, W_old, b_old, W_new, b_new):
    old = X @ W_old.T + b_old
    new = X @ W_new.T + b_new
    return np.concatenate([old, new], axis=1).argmax(1)


def evaluate_holdout(**hp):
    embs, p2i, W_old, b_old = load_cache()
    X, y = build_labeled(embs, p2i)
    tr, va = stratified_split(y, seed=hp.pop("split_seed", 0))
    Wn, bn = train_new_head(X[tr], y[tr], W_old, b_old, **hp)
    yp = predict(X[va], W_old, b_old, Wn, bn)
    s = score(y[va].tolist(), yp.tolist())
    # baseline: what does the frozen 16-way head alone get on old val clips?
    old_only = (X[va] @ W_old.T + b_old).argmax(1)
    old_mask = y[va] < NUM_OLD
    frozen_acc_old = (old_only[old_mask] == y[va][old_mask]).mean()
    return s, frozen_acc_old


def make_submission(out_path=None, **hp):
    embs, p2i, W_old, b_old = load_cache()
    X, y = build_labeled(embs, p2i)
    Wn, bn = train_new_head(X, y, W_old, b_old, **hp)  # train on ALL labeled
    sub = _rows("submission.csv")
    Xe = np.asarray([embs[p2i[r["path"]]] for r in sub], np.float32)
    preds = predict(Xe, W_old, b_old, Wn, bn)
    out_path = Path(out_path or HERE / "submission.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "target"])
        for r, p in zip(sub, preds):
            w.writerow([r["path"], int(p)])
    print(f"wrote {out_path} ({len(sub)} rows); pred dist:",
          dict(zip(*[x.tolist() for x in np.unique(preds, return_counts=True)])))
    return out_path


if __name__ == "__main__":
    s, frozen_acc_old = evaluate_holdout()
    print("frozen-head acc_old (16-way, ref):", round(float(frozen_acc_old), 4))
    print("HOLDOUT metric:", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in s.items()})
