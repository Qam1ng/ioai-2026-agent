#!/usr/bin/env python3
"""Standalone entry point — start the autonomous agent WITHOUT Claude Code.

Claude Code was only the development environment. The agent itself is plain
Python that talks to the Anthropic API directly (key from .env). This script
runs the whole pipeline autonomously — the IOAI² "human only initiates" model:

    python run.py --task task1_audio \
        --slug ioai-2026-ai-models-track-practice-task-1 --iters 6 --submit

Steps:
  1. ensure frozen-feature cache exists (extract if missing)
  2. run the agent loop (the agent decides + codes every experiment)
  3. build the Kaggle notebook from the agent's best fit_predict
  4. (--submit) push the notebook, submit its output, poll the leaderboard score

Without --submit it stops after producing the notebook + local best (dry run).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sh(cmd, **kw):
    print("+", " ".join(cmd)); return subprocess.run(cmd, check=True, **kw)


def ensure_features(task: str):
    cache = ROOT / "tasks" / task / "cache" / "embeddings.npy"
    if cache.exists():
        print(f"[features] cache present: {cache}")
        return
    print("[features] extracting (one-time, GPU)...")
    sh([sys.executable, str(ROOT / "tasks" / task / "features.py")])


def run_agent_loop(task: str, iters: int, seed: int):
    print(f"[agent] running loop: {iters} iterations (the agent decides)")
    sh([sys.executable, str(ROOT / "src" / "agent_loop.py"),
        "--task", task, "--iters", str(iters), "--seed", str(seed)])
    best = json.loads((ROOT / "tasks" / task / "runs" / "best.json").read_text())
    print(f"[agent] best: {best['name']!r} holdout={best['score']['score']:.4f}")
    return best


def build_kernel(task: str, best: dict) -> Path:
    kdir = ROOT / "tasks" / task / "kernel"
    template = (kdir / "template.py").read_text()
    marker = ("# (run.py replaces this line with the agent's best fit_predict "
              "definition)\ndef fit_predict(Xtr, ytr, Xva, old_W, old_b):\n"
              '    raise RuntimeError("no agent code injected")')
    if marker not in template:
        raise RuntimeError("template marker not found; kernel template changed?")
    script = template.replace(marker, best["code"].strip())
    (kdir / "ioai_t1_submit.py").write_text(script)
    print(f"[kernel] injected agent best into {kdir/'ioai_t1_submit.py'}")
    return kdir


def kaggle_submit(kdir: Path, slug: str, kernel_ref: str, note: str):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()

    print("[kaggle] pushing notebook (runs on Kaggle GPU)...")
    out = subprocess.run(["kaggle", "kernels", "push", "-p", str(kdir)],
                         capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    ver = None
    for tok in (out.stdout + out.stderr).split():
        if tok.isdigit():
            ver = int(tok)
    print(f"[kaggle] version = {ver}")

    print("[kaggle] waiting for the notebook run to finish...")
    while True:
        st = subprocess.run(["kaggle", "kernels", "status", kernel_ref],
                            capture_output=True, text=True).stdout
        if "RUNNING" in st or "QUEUED" in st:
            time.sleep(20); continue
        print("[kaggle] kernel status:", st.strip()); break
    if "COMPLETE" not in st:
        raise RuntimeError(f"kernel did not complete: {st.strip()}")

    print("[kaggle] submitting notebook output...")
    r = api.competition_submit_code("submission.csv", note, slug,
                                    kernel=kernel_ref, kernel_version=ver)
    print("[kaggle] submit:", r)

    print("[kaggle] polling leaderboard score...")
    for _ in range(60):
        subs = subprocess.run(["kaggle", "competitions", "submissions", "-c", slug],
                              capture_output=True, text=True).stdout
        top = "\n".join(subs.splitlines()[:4])
        if "PENDING" not in top.splitlines()[-1] if len(top.splitlines()) >= 3 else False:
            print(top); return top
        time.sleep(30)
    print("[kaggle] still pending after poll window")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1_audio")
    ap.add_argument("--slug", default="ioai-2026-ai-models-track-practice-task-1")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel-ref", default="qam1ng/ioai-t1-submit")
    ap.add_argument("--submit", action="store_true", help="push + submit to Kaggle")
    args = ap.parse_args()

    ensure_features(args.task)
    best = run_agent_loop(args.task, args.iters, args.seed)
    kdir = build_kernel(args.task, best)
    if args.submit:
        kaggle_submit(kdir, args.slug, args.kernel_ref,
                      f"run.py: {best['name'][:60]} (holdout {best['score']['score']:.4f})")
    else:
        print("[done] notebook built; re-run with --submit to push to Kaggle.")


if __name__ == "__main__":
    main()
