"""Official Task-1 metric + a local-CV / submission harness.

This is the objective *verifier* for the experiment loop — the piece that
replaces IMO25's LLM judge. It is deliberately data-free and dependency-light
(pure Python/stdlib) so it can be unit-tested without the 2 GB dataset or a GPU.

Task-1 score = 0.5 * Acc_old + 0.5 * Acc_new
  OLD classes = labels 0..15 (16 base classes)
  NEW classes = labels 16..28 (13 fine-tune classes)
Accuracy is measured over test clips grouped by their TRUE label.
"""
from __future__ import annotations

from typing import Sequence

OLD_CLASSES = frozenset(range(0, 16))   # 0..15
NEW_CLASSES = frozenset(range(16, 29))  # 16..28
NUM_CLASSES = 29


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if not y_true:
        return 0.0
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def _subset_acc(y_true, y_pred, label_set) -> float:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in label_set]
    if not pairs:
        return 0.0  # no test clips for this group
    ct, cp = zip(*pairs)
    return accuracy(ct, cp)


def score(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Return the official weighted score plus its components.

    Returns {'score', 'acc_old', 'acc_new', 'acc_overall', 'n_old', 'n_new'}.
    """
    acc_old = _subset_acc(y_true, y_pred, OLD_CLASSES)
    acc_new = _subset_acc(y_true, y_pred, NEW_CLASSES)
    n_old = sum(1 for t in y_true if t in OLD_CLASSES)
    n_new = sum(1 for t in y_true if t in NEW_CLASSES)
    return {
        "score": 0.5 * acc_old + 0.5 * acc_new,
        "acc_old": acc_old,
        "acc_new": acc_new,
        "acc_overall": accuracy(y_true, y_pred),
        "n_old": n_old,
        "n_new": n_new,
    }


def _selftest() -> None:
    import random

    random.seed(0)

    # 1. Perfect predictions -> score 1.0
    yt = list(range(29))
    assert score(yt, yt)["score"] == 1.0, "perfect should be 1.0"

    # 2. The catastrophic-forgetting trap: old clips all wrong, new all right.
    #    Overall accuracy is high (more new clips), but the 50/50 score punishes
    #    forgetting — proving the metric does what the task intends.
    yt = [0] * 16 + list(range(16, 29))          # 16 old clips + 13 new clips
    yp = [16] * 16 + list(range(16, 29))         # every old clip mispredicted as 16
    r = score(yt, yp)
    assert r["acc_old"] == 0.0 and r["acc_new"] == 1.0
    assert abs(r["score"] - 0.5) < 1e-9, r
    # overall accuracy would be 13/29 ~= 0.448, but score is 0.5 — different signal
    assert abs(r["acc_overall"] - 13 / 29) < 1e-9

    # 3. The opposite trap: keep old perfectly, learn nothing new -> also 0.5
    yp = [t if t in OLD_CLASSES else 0 for t in yt]
    r = score(yt, yp)
    assert r["acc_old"] == 1.0 and r["acc_new"] == 0.0
    assert abs(r["score"] - 0.5) < 1e-9, r

    # 4. Random 29-way baseline hovers near 1/29 on both groups.
    yt = [random.randrange(29) for _ in range(2000)]
    yp = [random.randrange(29) for _ in range(2000)]
    r = score(yt, yp)
    assert 0.0 <= r["score"] <= 0.15, r  # ~1/29 ≈ 0.034

    print("metric self-test OK")
    print("  trap (forget old): ", score([0]*16+list(range(16,29)),
                                          [16]*16+list(range(16,29))))
    print("  random 29-way:     ", {k: round(v, 4) if isinstance(v, float) else v
                                     for k, v in r.items()})


if __name__ == "__main__":
    _selftest()
