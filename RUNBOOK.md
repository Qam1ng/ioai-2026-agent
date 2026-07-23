# RUNBOOK — how to run the IOAI 2026 agent

This is the operator manual. The system is an **autonomous experiment loop**:
the agent (Opus 4.8) decides every experiment itself and the harness scores it
objectively. A human only sets things up and (optionally) presses the final
"submit" button — the agent makes the ML decisions.

---

## 0. One-time setup

```bash
cd /home/qing/IOAI2026-agent

# Python deps (GPU box; RTX 4080 present). Python 3.14 / miniconda.
export PATH="/home/qing/miniconda3/bin:$PATH"
pip install -r requirements.txt          # anthropic, kaggle, torch, transformers, librosa, sklearn ...

# Anthropic key (the agent's brain) — gitignored, never commit.
cp .env.example .env                     # then paste ANTHROPIC_API_KEY (already set on this box)
python src/llm.py                        # smoke test -> "IOAI-2026 agent online."

# Kaggle token (for data download + submission), IOAI Discord format:
mkdir -p ~/.kaggle
printf 'KGAT_xxxxxxxx' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
kaggle competitions files -c <competition-slug>   # verifies auth
```

Environment expectations: `nvidia-smi` shows a GPU; `python -c "import torch;print(torch.cuda.is_available())"` prints `True`.

---

## 1. Per-task workflow

Everything lives under `tasks/<task>/`. Example: `task1_audio`.

### a) Get the data (once)
```bash
SLUG=ioai-2026-ai-models-track-practice-task-1
IN=tasks/task1_audio/input
kaggle competitions download -c $SLUG -p $IN
unzip -q -o $IN/$SLUG.zip -d $IN        # -> audio/, model/, *.csv
```

### b) Extract frozen features (once, ~3 min on the 4080)
Runs the frozen encoder over every clip and caches 768-d embeddings so the
agent loop iterates in seconds instead of minutes.
```bash
python tasks/task1_audio/features.py     # -> tasks/task1_audio/cache/{embeddings.npy, old_head_*.npy, paths.txt}
```

### c) Run the autonomous agent loop  ← the main event
The agent proposes + codes each experiment; the harness runs it on cached
features and scores it with the official metric on a fixed local holdout.
```bash
python src/agent_loop.py --task task1_audio --iters 6
```
Outputs:
- live log of each decision and its holdout score, marking new bests
- `tasks/task1_audio/runs/history.json` — every experiment (name, code, score, diagnostics)
- `tasks/task1_audio/runs/best.json` — the agent's best decision
- `tasks/task1_audio/submission.csv` — predictions from the best (NOT yet submitted)

Flags: `--iters N` (budget), `--seed S` (holdout split).

### d) Inspect what the agent did
```bash
python - <<'PY'
import json
h=json.load(open("tasks/task1_audio/runs/history.json"))
for r in h:
    if r.get("error"): print(r["iter"], r["name"], "ERROR", r["error"])
    else: print(r["iter"], round(r["score"]["score"],4), r["name"])
PY
```

### e) Submit (the one human-gated step)
Validate that our local holdout tracks the real leaderboard, then submit:
```bash
kaggle competitions submit -c $SLUG \
  -f tasks/task1_audio/submission.csv -m "agent-loop best"
kaggle competitions submissions -c $SLUG | head   # read the public score
```
> In the real IOAI² run the agent submits autonomously via the Kaggle CLI (no
> human). During development we keep this manual so a human can gate it.

---

## 2. What each piece is

| Path | Role |
|---|---|
| `src/llm.py` | Anthropic client (Opus 4.8, adaptive thinking, streaming) |
| `src/metric.py` | Official metric `0.5·acc_old + 0.5·acc_new` — the objective verifier |
| `src/agent_loop.py` | The autonomous loop: agent decides → harness executes + scores → feeds back |
| `tasks/<t>/features.py` | One-time frozen-encoder feature extraction + cache |
| `tasks/<t>/solve.py` | A hand-written baseline solver (reference; the loop supersedes it) |
| `tasks/<t>/CARD.md` | Task spec |
| `tasks/<t>/cache/` | Cached embeddings + frozen head |
| `tasks/<t>/runs/` | The agent's experiment history + best |

---

## 3. How the loop makes decisions (design)

Each iteration the agent receives: the task + metric, the feature API
(`fit_predict(Xtr, ytr, Xva, old_W, old_b) -> preds`), and the full history of
its past experiments with their holdout scores and per-class diagnostics. It
returns a rationale + a `fit_predict` code block. The harness `exec`s it, scores
the holdout, and — if the code crashes — feeds the traceback back so the agent
self-corrects next turn (generate → execute → verify-by-metric → correct).

**No human proposes experiments.** The score improvement across iterations is
attributable to the agent, which is the quantity IOAI² measures.

To add a new task: create `tasks/<name>/`, drop the data in `input/`, write a
`features.py` that caches `embeddings.npy` + the frozen head + `paths.txt`, then
run `agent_loop.py --task <name>`.
