# DESIGN — multiagent-sdk branch (AI-Build-AI–style, Claude Agent SDK kernel)

Branch goal: adopt the **AIBuildAI** architecture (arXiv 2604.14455, 2605.27873 —
MLE-bench #1 @ 70.7%) on a **Claude Agent SDK** kernel with **Fable 5 only**,
adapted to IOAI² constraints. `master` (provider-agnostic single-agent) stays as
the fallback; the two are A/B'd before Aug 4.

## What we learned from AIBuildAI (papers + artifacts)

**v1 (2604.14455):**
- **Manager** keeps a *compact* context; reads each repo's state via `read_i`
  tools; **asynchronous parallel tool calls** across **n=7 solution repos**;
  feedback-driven routing: approach flawed → Designer; implementation issue →
  Coder; otherwise Tuner. Kills underperforming or *similar* repos to save
  compute. Runs until budget (24 h) → Aggregator.
- **Designer**: proposes/revises a plan, *no code*; diagnoses underfit /
  overfit / instability from results.
- **Coder**: implements plan → `train.py` + `config.py`; sanity-check run only
  (*no training*); repo-isolated.
- **Tuner**: short preliminary runs first → extend promising configs to full
  training; analyzes convergence/overfit; updates config; writes results.
- **Aggregator**: task-adaptive ensembling (rank-percentile blending for
  classification, weighted prob maps for segmentation, weighted averages for
  regression), fallback = pick best single repo.
- Backbone: Opus 4.6, temp 1.0; sub-agents own their contexts (prevents
  context explosion in the manager).

**v2 (2605.27873):** adds a **hierarchical knowledge system**: L1 ≈ 30
human-authored categories (always in context as an index) + L2 ≈ 1000 curated
docs (retrieved on demand); **self-evolution** — after each run an L2-builder
distills the trace into a takeaway, an L1-builder updates the index; plus a
community-learning stream. OOF-fitted blend weights (hill-climb rank average /
SLSQP) over k-fold base models. Opus 4.7. **This is our memory/skills design,
matured** — validates our approach and gives the growth path.

**Artifacts** (repo `tasks/`): `candidate_i/{train.py, config.py, best_model.pth}`,
`model_designs.json`, `ensemble_search.py`, `ensemble_info.json`, `inference.py`.

## IOAI adaptations (where copying would break rules or budgets)

| AIBuildAI | IOAI² reality | Our adaptation |
|---|---|---|
| 24 h/task, A100 | **2 h/task**, Kaggle P100/T4 (30 h GPU quota/day/acct) | fewer repos (dev 2–3, comp 3–5); Manager must early-stop aggressively; preliminary runs ≤ minutes |
| Submits checkpoints/inference.py | **No checkpoint submission** — a `.py` must *train on the Kaggle kernel* (~10 min) | Aggregator emits a single `submit.py` that retrains in-kernel; ensembling only as *in-kernel* seed/config ensembles, or OOF-fitted blend **weights hard-coded as config** (weights = hyperparameters, allowed) |
| Tuner watches local long trainings | local trainings are short; the *real* long-poll is the **Kaggle kernel queue** | Tuner also owns async kernel monitoring, accelerator choice (CPU/P100/T4), GPU-quota frugality |
| Validation ad-hoc per repo | our known failure: single-holdout optimism (0.916→0.781) | **NEW: Verifier agent** (below) |
| Web search (v1) / knowledge base (v2) | — | keep `memory/lessons` + `skills/` as our L1/L2 seed; post-run lesson distillation = mini self-evolution |
| Cost not reported | **tokens/cost are published & tie-break** | record everything (SDK usage → tracer); performance still first priority |

## The Verifier agent (our addition — jointly agreed)

A dedicated sub-agent that owns *measurement*, so trajectories are judged by
statistics rather than vibes:

1. At SCAFFOLD time, builds the **shared validation harness**: stratified k-fold
   splits mirroring the official metric, fixed seeds, saved to
   `workspace/<slug>/validation/` — **all repos must use the same splits**
   (otherwise cross-repo comparison is meaningless).
2. **Stage-gated verification**: at Manager-chosen checkpoints it re-scores every
   live repo on the shared folds and produces a *verification report*
   (mean ± σ per repo, significance vs current best, overfitting flags,
   local↔LB calibration once real submissions exist).
3. The Manager's kill/extend/submit decisions must cite the latest verification
   report. This operationalizes our 4-tier verification stack inside the
   multi-agent design.

## Architecture on the Claude Agent SDK

- **Kernel**: `claude-agent-sdk` (Python). AIBuildAI itself runs on Claude Code
  credentials — this path is battle-tested.
- **Model**: `claude-fable-5` for **all** agents (user decision). Notes: thinking
  always-on (omit `thinking` config), no sampling params, handle
  `stop_reason: refusal` gracefully (log + retry once with rephrase; no
  cross-model fallback by user decision).
- **Manager = main SDK session**; Setup/Designer/Coder/Tuner/Verifier/Aggregator
  = **SDK subagents** (`agents={...}` AgentDefinitions) — each gets its own
  context window, matching the paper's context-isolation design.
- **Parallel repos**: `workspace/<slug>/repo_i/` dirs; Manager fans out parallel
  subagent calls (SDK parallel tool use). Dev cap: `--repos 2..3`.
- **Harness stays ours** (cannot be bypassed by the LLM):
  - SDK **hooks**: PreToolUse gate on `kaggle_submit` (budget) and on `bash`
    (workspace confinement); PostToolUse → cost/step tracer (`trace.jsonl`).
  - Wall-clock + submission + GPU-quota budgets injected into Manager context
    every turn; hard interrupt at deadline → force Aggregator.
  - Invariant: floor submission early; a valid submission exists at all times.
- **Tools**: reuse the proven kaggle toolchain (download/push/status/log/
  submit/submissions) as SDK custom tools (in-process MCP server), plus
  file/bash tools the SDK already provides.
- **Knowledge**: `skills/` exposed to agents; `memory_recall/write` as tools;
  REPORT phase distills lessons (mini L2-builder).

```
launch (human): python -m magent.main --slug … --brief task.md --repos 3
   └─ MANAGER (Fable 5, SDK session; budget line injected per turn)
        ├─ Setup      → data down, env probe, workspace/repo_i scaffolding
        ├─ Verifier   → shared k-fold harness; stage-gated verification reports
        ├─ per repo_i: Designer → plan_i   (no code)
        │              Coder    → train/config/submit.py sanity-checked (no training)
        │              Tuner    → short probes → extend best; async kernel watch
        ├─ kill / extend / submit decisions ← verification reports + budgets
        └─ Aggregator → single in-kernel submit.py (best single or in-kernel
                        ensemble / OOF-weight blend) → kaggle submit → REPORT
```

## Experiment suite (agreed)

| # | Competition | Notes |
|---|---|---|
| 1 | `ioai-2026-ai-models-track-practice-task-1` | baseline exists: master scored **0.78392** autonomously |
| 2 | `ioai-2026-ai-models-track-at-home-practice-task-2` | robot delivery; generalization test (note different slug pattern) |
| 3 | `radar-ioai-2025` | IOAI 2025 real task |
| 4 | `ventilator-pressure-prediction` | MLE-bench-style; old comp → late submission / local CV |
| 5 | `freesound-audio-tagging-2019` | MLE-bench-style; audio tagging, noisy labels |

Success criteria: ≥ master's score on task 1; completes unseen tasks 2–5 without
human help; verification reports demonstrably drive kill/extend decisions;
full cost accounting per run.

## Decisions on record

- Branch = SDK-only, **Fable 5 only**; master untouched as fallback.
- Cost: **recorded always, optimized never** (competition performance first);
  dev runs use small repo counts to cap spend.
- Repo count is a config, not a constant.
