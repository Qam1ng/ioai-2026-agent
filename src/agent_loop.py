"""Autonomous experiment loop — the agentic core.

The agent (Opus 4.8) is the decision-maker. Each iteration it sees the task,
the objective metric, and the full history of what it has already tried (with
scores + diagnostics), then it *itself* decides the next experiment and writes
the code for it. The harness only:
  1. executes the agent's code against cached frozen-AST features,
  2. scores it with the official metric on a fixed local holdout (the objective
     verifier — no human, no LLM judge),
  3. feeds the result (or the traceback, if it crashed) back to the agent.

No human proposes experiments. The point of the project is to measure how much
the agent lifts the score on its own, so keep humans out of the decision.

Usage:
    python src/agent_loop.py --task task1_audio --iters 6
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metric import score, OLD_CLASSES, NUM_CLASSES  # noqa: E402
from llm import complete, DEFAULT_MODEL  # noqa: E402


# ----------------------------- data plumbing ------------------------------ #
def load_task(task: str):
    tdir = ROOT / "tasks" / task
    cache, IN = tdir / "cache", tdir / "input"
    embs = np.load(cache / "embeddings.npy")
    paths = (cache / "paths.txt").read_text().splitlines()
    p2i = {p: i for i, p in enumerate(paths)}
    old_W = np.load(cache / "old_head_W.npy")
    old_b = np.load(cache / "old_head_b.npy")

    def rows(name):
        with open(IN / name) as f:
            return list(csv.DictReader(f))

    X, y = [], []
    for name in ("train.csv", "fine_tune.csv"):
        for r in rows(name):
            X.append(embs[p2i[r["path"]]]); y.append(int(r["target"]))
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)

    sub = rows("submission.csv")
    Xe = np.asarray([embs[p2i[r["path"]]] for r in sub], np.float32)
    return tdir, X, y, old_W, old_b, sub, Xe


def stratified_split(y, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n = max(1, round(len(idx) * val_frac)) if len(idx) > 1 else 0
        va += list(idx[:n]); tr += list(idx[n:])
    return np.array(tr), np.array(va)


# --------------------------- agent interaction ---------------------------- #
SYSTEM = """You are an autonomous ML engineer competing in a Kaggle class-incremental audio task.
You improve the model by proposing ONE new experiment per turn and writing its code. You are the sole decision-maker — there is no human to consult.

SETUP (all features are precomputed, so iterations are cheap):
- A frozen Audio Spectrogram Transformer encoder produced a 768-d LayerNormed feature per clip.
- `old_W` [16,768], `old_b` [16] are the checkpoint's FROZEN 16-class head (dense layer). `old_logits = feat @ old_W.T + old_b` reproduces the original model's old-class logits EXACTLY.
- Labels: 0..15 = OLD (16 base classes), 16..28 = NEW (13 fine-tune classes). 29 classes total.
- Known diagnostic: the frozen head is excellent on most old classes but SYSTEMATICALLY fails on class 3 (Cow ~0.14), 9 (Mouse Click ~0.14), 11 (Thunderstorm ~0.16). Old-class data is small/imbalanced (as few as 3 samples/class); new-class data has 24-60/class.

OBJECTIVE (maximize on a held-out set):
    score = 0.5 * accuracy_on_OLD_clips + 0.5 * accuracy_on_NEW_clips
Forgetting old is exactly as costly as failing to learn new.

YOUR OUTPUT each turn MUST be:
1. One short paragraph: your decision and why (grounded in the history/diagnostics).
2. A single ```python code block defining EXACTLY this function:

def fit_predict(Xtr, ytr, Xva, old_W, old_b):
    # Xtr [Ntr,768] float32, ytr [Ntr] int in 0..28, Xva [Nva,768]
    # old_W [16,768], old_b [16]  (the frozen old head)
    # return: np.ndarray [Nva] of int predictions in 0..28
    ...

Rules for the code:
- self-contained; may `import numpy as np`, `torch`, `sklearn`, `scipy`.
- must return predictions for ALL 29 classes (argmax over a 29-way decision).
- deterministic (set seeds). Keep training well under ~1 minute on the given arrays.
- Do NOT read files or the network. Only use the arrays passed in.
Ideas you may explore (your call): keep vs. retrain the old head, class-balanced loss, per-class or new-vs-old logit bias/temperature calibration, regularizing retrained old rows toward old_W, prototypes / nearest-class-mean, kNN, logistic regression, a small MLP, feature normalization, targeting the 3 broken old classes specifically. Try to beat the best score in the history."""


def build_user_prompt(history: list[dict], best: dict | None) -> str:
    lines = ["## History of experiments (most recent last)"]
    if not history:
        lines.append("(none yet — this is the first experiment)")
    for h in history:
        if h.get("error"):
            lines.append(f"- iter {h['iter']}: {h['name']!r} -> ERROR: {h['error']}")
        else:
            s = h["score"]
            lines.append(
                f"- iter {h['iter']}: {h['name']!r} -> score={s['score']:.4f} "
                f"(acc_old={s['acc_old']:.3f}, acc_new={s['acc_new']:.3f})"
                + (f" | worst old classes: {h['worst_old']}" if h.get("worst_old") else "")
            )
    if best and not best.get("error"):
        lines.append(f"\nBEST so far: {best['name']!r} score={best['score']['score']:.4f}")
    lines.append("\nPropose the next experiment (rationale + the fit_predict code block). "
                 "Make a decision that is most likely to raise the held-out score.")
    return "\n".join(lines)


CODE_RE = re.compile(r"(?:```|~~~)(?:python)?\s*(.*?)(?:```|~~~)", re.DOTALL)


def parse_response(text: str):
    m = CODE_RE.search(text)
    if m:
        code = m.group(1).strip()
    else:  # no fence — grab from the def to the end (best effort)
        i = text.find("def fit_predict")
        code = text[i:].strip() if i != -1 else None
    # name = first line that mentions the approach, else first prose line
    name = "experiment"
    for ln in text.splitlines():
        s = ln.strip().lstrip("#").strip()
        if s and "fit_predict" not in s and not s.startswith("```"):
            name = s[:80]
            break
    return name, code


def run_code(code: str, Xtr, ytr, Xva, old_W, old_b):
    ns: dict = {"np": np, "__name__": "agent_experiment"}
    exec(code, ns)  # noqa: S102 — our box, our loop
    fn = ns.get("fit_predict")
    if fn is None:
        raise ValueError("code did not define fit_predict")
    preds = np.asarray(fn(Xtr, ytr, Xva, old_W, old_b)).astype(int).ravel()
    if preds.shape[0] != Xva.shape[0]:
        raise ValueError(f"returned {preds.shape[0]} preds, expected {Xva.shape[0]}")
    if preds.min() < 0 or preds.max() >= NUM_CLASSES:
        raise ValueError(f"preds out of range [0,{NUM_CLASSES-1}]: {preds.min()}..{preds.max()}")
    return preds


def worst_old_classes(yva, yp, k=3):
    out = {}
    for c in sorted(OLD_CLASSES):
        m = yva == c
        if m.sum():
            out[c] = round(float((yp[m] == c).mean()), 2)
    return dict(sorted(out.items(), key=lambda kv: kv[1])[:k])


# -------------------------------- main loop ------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1_audio")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tdir, X, y, old_W, old_b, sub, Xe = load_task(args.task)
    tr, va = stratified_split(y, seed=args.seed)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    print(f"task={args.task} model={DEFAULT_MODEL} | train={len(tr)} val={len(va)} "
          f"| eval={len(Xe)}")

    runs_dir = tdir / "runs"; runs_dir.mkdir(exist_ok=True)
    history: list[dict] = []
    best: dict | None = None

    for it in range(1, args.iters + 1):
        prompt = build_user_prompt(history, best)
        t0 = time.time()
        reply = complete(SYSTEM, prompt, max_tokens=8000, effort="high")
        (runs_dir / f"iter_{it}_reply.md").write_text(reply)
        name, code = parse_response(reply)
        rec: dict = {"iter": it, "name": name, "code": code,
                     "llm_secs": round(time.time() - t0, 1)}
        if not code:
            rec["error"] = "no code block found"
        else:
            try:
                yp = run_code(code, Xtr, ytr, Xva, old_W, old_b)
                rec["score"] = score(yva.tolist(), yp.tolist())
                rec["worst_old"] = worst_old_classes(yva, yp)
            except Exception:
                rec["error"] = traceback.format_exc().strip().splitlines()[-1][:200]
        history.append(rec)

        if rec.get("error"):
            print(f"[iter {it}] {name!r}: ERROR {rec['error']}")
        else:
            s = rec["score"]
            better = best is None or best.get("error") or s["score"] > best["score"]["score"]
            tag = "  <-- NEW BEST" if better else ""
            print(f"[iter {it}] {name!r}: score={s['score']:.4f} "
                  f"(old={s['acc_old']:.3f} new={s['acc_new']:.3f}){tag}")
            if better:
                best = rec
        (runs_dir / "history.json").write_text(json.dumps(history, indent=1))

    # ---- leaderboard + submission from the agent's best decision ---- #
    print("\n=== leaderboard (agent's own decisions) ===")
    ranked = sorted([h for h in history if not h.get("error")],
                    key=lambda h: -h["score"]["score"])
    for h in ranked:
        print(f"  {h['score']['score']:.4f}  (old={h['score']['acc_old']:.3f} "
              f"new={h['score']['acc_new']:.3f})  {h['name']}")

    if best and not best.get("error"):
        print(f"\nBest = {best['name']!r} @ {best['score']['score']:.4f}. "
              "Retraining on ALL labeled data for submission...")
        preds = run_code(best["code"], X, y, Xe, old_W, old_b)
        out = tdir / "submission.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["path", "target"])
            for r, p in zip(sub, preds):
                w.writerow([r["path"], int(p)])
        (runs_dir / "best.json").write_text(json.dumps(
            {k: best[k] for k in ("iter", "name", "score", "code")}, indent=1))
        print(f"wrote {out}  (NOT submitted to Kaggle)")


if __name__ == "__main__":
    main()
