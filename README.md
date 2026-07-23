# IOAI 2026 — AI Models Track Agent

An autonomous agentic system for the **IOAI² AI Models Track** (IOAI 2026). The
agent must **autonomously write, run, and submit ML code** on expert-designed
olympiad ML tasks, scored on **Kaggle** — no human involvement during the run.

Conceptually adapted from the IMO25 solver's *generate → verify → correct* loop,
but with the key upgrade demanded by IOAI: the **verifier is an objective metric**
(local CV + Kaggle score), not an LLM judge. See the design notes below.

## Status

| Piece | State |
|---|---|
| Claude brain (Opus 4.8, adaptive thinking) | ✅ working — `src/llm.py`, smoke test passes |
| API key handling (gitignored `.env`, `chmod 600`) | ✅ |
| Kaggle submission wiring | ⛔ blocked — needs Kaggle token + the Discord `#resources` "Agent Skill file" |
| Task-1 (Audio Classifier) experiment loop | ⛔ blocked on the above + task data |
| ML stack (numpy/pandas/sklearn/torch) | ⛔ not yet installed (bare env) |

## Layout

```
src/llm.py        Anthropic client wrapper (Opus 4.8, adaptive thinking, streaming)
requirements.txt  Dependencies
.env              (gitignored) ANTHROPIC_API_KEY — never committed
tasks/            One dir per IOAI task (data, work, submission) — gitignored contents
skills/           Reusable ML playbooks the agent loads per task family
```

## Design (target architecture)

The core loop the agent will run per problem, within the Kaggle budget
(50 submissions/problem, <30 min/run):

```
read task (ipynb + data) → EDA → baseline submission (guarantee a score)
        ↓
build a LOCAL CV harness that mirrors the official metric   ← the cheap verifier
        ↓
loop under budget:
  coder(Opus 4.8) writes/edits code → execute in sandbox
      → local CV score → if better, spend 1 Kaggle submission → read true score
      → record to experiment leaderboard → decide next move
        ↓
keep best submission + auto-write the required technical report
```

What carries over from IMO25: the self-iterating loop, best-of-N, memory/
checkpointing, structured prompting. What changes: verify = real metric (not
LLM judge); correct = edit code / retrain (not edit prose).

## Setup

```bash
cp .env.example .env   # then paste the dedicated ANTHROPIC_API_KEY
pip install -r requirements.txt
python src/llm.py      # smoke test: should print "IOAI-2026 agent online."
```

## Competition facts (from rules PDF + Discord)

- Sessions: **4 & 6 Aug 2026**, 6-hour window, 3 problems each, on Kaggle.
- Registration: email `adivekar@utexas.edu` by **31 Jul 2026** (lab name in subject).
- At Home practice tasks (don't affect ranking; for wiring up Kaggle submission):
  Task 1 = Audio Classifier, Task 2 = Robot Delivery Academy.
- All tools must be declared; humans may only initiate the run; efficiency
  (tokens, $, iterations) is reported and breaks ties.
