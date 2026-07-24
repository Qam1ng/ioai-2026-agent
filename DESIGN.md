# DESIGN — Autonomous ML Competition Agent (full architecture)

The deliverable for the IOAI² AI Models Track is a **complete autonomous agentic
system**: a human only launches it; the system reads the task, plans, writes all
code, iterates, submits, and writes the technical report — with no human in the
loop and **no dependency on Claude Code** (which was only our development
environment). The model brain is a **pluggable backend** (Claude today; any
API/local model tomorrow).

Division of labor: the **harness enforces what must never go wrong** (budgets,
submission gates, always-valid-submission, tracing); the **agent decides
everything that needs intelligence** (reading, planning, coding, iterating,
selecting).

```
                        ┌─ the human's ONLY action ─┐
                        │ python agent.py            │
                        │   --slug <competition>     │
                        │   --deadline 6h            │
                        │   --backend claude|openai  │
                        └──────────┬─────────────────┘
                                   ▼
┌───────────────────────── HARNESS (plain Python orchestrator) ─────────────────────────┐
│  ┌────────────┐   ┌───────────────────────────────────────────────────────────┐       │
│  │ LLM Backend │◀──│            AGENT LOOP (tool-use / ReAct)                 │       │
│  │ (pluggable) │──▶│  model sees state → decides → calls tool → observes → …  │       │
│  └────────────┘   └──────┬────────────────────────────────────────────────────┘       │
│                          ▼                                                             │
│  ┌──────────────────────── TOOLS ────────────────────────────┐                        │
│  │ run_python (sandbox+timeout) · files r/w · kaggle.*        │                        │
│  │ (download / push-kernel / submit / read-score)             │                        │
│  │ skill.load · memory.recall/write · web_search (orch side)  │                        │
│  │ submit_gate (budget-checked)                               │                        │
│  └────────────────────────────────────────────────────────────┘                        │
│  ┌─ cross-cutting, enforced by harness (agent cannot bypass) ─┐                        │
│  │ ① budgets: wall-clock / submissions / tokens & cost         │                        │
│  │ ② invariant: a valid submission exists at all times         │                        │
│  │ ③ trace.jsonl: every LLM+tool call (for declaration+report) │                        │
│  │ ④ state.json checkpoints: crash-resume mid-run              │                        │
│  └──────────────────────────────────────────────────────────────┘                       │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    ▼
                submitted solution + auto-generated technical report + full trace
```

---

## Phase state machine (launch → final submission)

The harness drives 9 phases. Inside a phase the agent works freely with tools;
each phase exit has a **hard gate** the harness checks.

### Phase 0 — LAUNCH (human's only action)
`python agent.py --slug … --deadline … --submit-budget 50 --backend claude`
Harness creates `workspace/<slug>/`, `state.json`, `trace.jsonl`, starts budget
clocks. No human after this point.

### Phase 1 — INGEST (read the task)
Tools: `kaggle.download`, `read_file`, `run_python`.
Agent reads overview/data pages, scans the data tree, inspects any provided
checkpoint, reads the sample submission.
**Artifact:** `TASK_CARD.md` (task type, metric formula, data schema, submission
format, constraints). **Gate:** card must contain metric + submission format.

### Phase 2 — ANALYZE (EDA + recall)
Tools: `run_python` (class balance, sizes, checkpoint internals),
`memory.recall` (lessons from similar past tasks), `skill.load` (task-family
playbook: tabular / CV / audio / NLP / RL / class-incremental …).

### Phase 3 — PLAN
Agent writes `PLAN.md`: ranked candidate approaches (strong baseline first),
**verification design** (how local CV must mirror the official metric — the
foundation of everything downstream), budget allocation across phases, risks and
fallbacks. **Gate:** plan must contain a verification design and a fallback.

### Phase 4 — SCAFFOLD  ★ the part that is currently hand-written; the agent must write it
Agent writes: data loader, feature/model pipeline, **local metric
implementation**, **k-fold CV harness**, baseline, submission builder (a Kaggle
notebook for code competitions).
**Gate (verification tier 1):** metric self-test on synthetic labels with known
expected value; submission format validator (rows/columns/order/value range);
end-to-end smoke run on a small batch.

### Phase 5 — BASELINE SUBMIT (insurance, spend 1 real submission early)
Simplest valid solution → submit for real. Purpose: ① shake out the submission
path early (we hit notebook-only submission, `/kaggle/input/competitions/`
nesting, etc. — the agent will too); ② establish the floor: from now on the
**always-valid-submission invariant** holds; ③ first calibration point between
local CV and the real leaderboard.

### Phase 6 — EXPERIMENT LOOP (iterate) — the existing inner loop, generalized
Each iteration:
1. **Hypothesis** — agent picks the next experiment from PLAN + the experiment
   ledger (all past runs: approach, score, variance, per-class diagnostics) +
   the previous traceback if any.
2. **Code** → `run_python` in the sandbox (timeout-protected).
3. **Verification tier 2 — statistical:** **k-fold CV** (not a single holdout)
   → mean score **and cross-fold variance**.
4. **Significance gate:** a new best is accepted only if the mean gain exceeds
   a threshold × std and improves on most folds. This kills the noise-chasing
   failure we observed (a 3-sample class showing 0.0 on a 1-sample holdout).
5. **Errors:** traceback fed back; ≤2 retries per hypothesis, then abandon.
6. **Periodic real submission:** when (significant local gain ∧ submission
   budget ∧ time checkpoint) → submit → **recalibrate** the local↔leaderboard
   mapping with the real score.
7. **memory.write:** distilled lessons ("full retrain hurts old classes", …).
Budget interrupts: at late checkpoints the harness forces convergence (no new
directions, only consolidation).

### Phase 7 — FINALIZE
Retrain the best on all data → final check on a **never-touched fold**
(verification tier 4 — catches optimizing into the validation set) → final
format validation → select the final submission(s).

### Phase 8 — FINAL SUBMIT + REPORT
Final submission with a deadline buffer (e.g. T−20 min). Auto-generate the
**technical report** (required by the rules within 30 min of session end):
approach, experiment table from the ledger, score trajectory, **tool
declaration** (summarized directly from `trace.jsonl` — satisfies the rules'
declaration requirement), efficiency metrics (tokens, cost, iterations — the
rules publish these). Archive the workspace. Exit.

---

## The verification stack (why iterations stay honest)

| Tier | When | Catches |
|---|---|---|
| 1. Structural | after every code write | syntax, crashes, wrong submission format, wrong metric implementation |
| 2. Statistical | every experiment | noise-chasing: k-fold mean ± variance + significance gate |
| 3. Generalization | periodically | local-CV optimism: real Kaggle score recalibrates the local↔LB mapping |
| 4. Final holdout | at finalize | overfitting the validation signal itself |

Mapped to our observed failures: the 0.9156→0.78095 optimism is caught by tier 3;
the iter2→6 regression (fixing phantom 0.0 classes with 1 validation sample) is
caught by tier 2.

---

## Component map

| Component | Implementation | Status |
|---|---|---|
| LLM backend | `providers/base.py` (`generate(messages, tools)`), adapters `anthropic.py` / `openai.py` / `local.py` | 🔨 to build (`src/llm.py` is Claude-hardcoded) |
| Harness | `orchestrator.py`: phase machine + agent loop + budgets + invariants + trace | 🔨 to build |
| Tools | `tools/`: run_python (sandbox), files, kaggle (download/push/submit/score), memory, web_search, submit_gate | 🔶 kaggle path proven this session; wrap as tools |
| Skills | `skills/<family>/SKILL.md` playbooks, loaded on demand | 🔶 task1 experience becomes the first skill |
| Memory | `memory/lessons/`: cross-task lessons (notebook-only submission, `/kaggle/input/competitions/` nesting, sklearn 1.9 API changes, …) | 🔶 seeded from this session's potholes |
| Inner experiment loop | today's `src/agent_loop.py`, generalized = Phase 6 | ✅ exists; needs k-fold + significance gate |
| Report generator | summarize trace + ledger | 🔨 to build |

## Multi-problem scheduling (competition day)

The real session is 6 hours / 3 problems. A thin outer scheduler allocates
~110 min per problem + 30 min slack; a problem can be parked as soon as it has a
floor submission, and remaining time is reinvested by expected gain. Practice
runs are single-problem and skip this layer.

## Rules compliance mapping

- *"Code … written and submitted by AI agents autonomously"* → phases 1–8 run
  unattended; human only launches (Phase 0).
- *Declare tools* → auto-generated from `trace.jsonl`.
- *Efficiency metrics published* → tokens/cost/iterations tracked in trace.
- *Technical report within 30 min* → Phase 8 auto-generation.
- *50 submissions / problem, ~30 min runtime* → enforced by `submit_gate` and
  sandbox timeouts.
- *Model choice* → pluggable backend; awards favor reproducible/open models, so
  the backend abstraction is a requirement, not a nicety.
```
