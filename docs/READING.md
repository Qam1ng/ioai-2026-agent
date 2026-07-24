# Team Reading List — IOAI 2026 AI Models Track

Onboarding material for the team, roughly in reading order.

## 1. The competition

- **IOAI official site** — what the olympiad is, dates, tracks
  https://ioai-official.org/
- **IOAI²: AI Models Track — Rules & Format (PDF)** — *the* governing document.
  Key constraints: humans may only initiate the run; all tools must be declared
  (internet, code execution, retrieval, libraries, multi-agent, scaffolding);
  runs on Kaggle, 6-hour windows on 4 & 6 Aug 2026, 3 problems/window,
  50 submissions/problem, ~30 min per submission run; efficiency metrics
  (tokens, cost, time) are published; a technical report per problem is
  mandatory; awards only for models the jury can reproduce.
  https://ioai-official.org/wp-content/uploads/2026/06/IOAI%C2%B2-AI-Models-Track_Rules-Format.pdf

## 2. At-Home practice tasks (private Kaggle invites)

Purpose: wire your agentic system into the Kaggle submission flow. Scores do
not affect final ranking.

- **Practice Task 1 — Audio Classifier** (class-incremental learning; our system
  is validated end-to-end on this one, see README)
  https://www.kaggle.com/t/e463b7687e2142fb86f3bda61c1d8837
- **Practice Task 2 — Robot Delivery Academy** (untouched so far — the real
  generalization test for the outer agent)
  https://www.kaggle.com/t/10a6b578135e4cb49fe3865bd879c6dd

## 3. Background papers & benchmarks

- **"Gemini 2.5 Pro Capable of Winning Gold at IMO 2025"** (Huang & Yang, 2025)
  — the IMO25 agent whose *generate → verify → correct* loop inspired our inner
  loop. Note the key difference: IMO has no ground truth, so they used an
  LLM-as-judge verifier; IOAI tasks have objective metrics, so our verifier is
  real measurement (k-fold CV + leaderboard).
  https://arxiv.org/pdf/2507.15855
- **OpenAI MLE-bench** — the closest public benchmark to what IOAI² measures:
  agents solving Kaggle-style ML engineering tasks end-to-end. Useful for
  harness design ideas, failure taxonomies, and baseline expectations.
  https://github.com/openai/mle-bench

## 4. Past IOAI problems (what the olympiad actually asks)

Each task = notebook + training data + scoring script. Skim several to build a
task-family taxonomy (tabular / CV / audio / NLP / semi-supervised / RL-ish) —
this is what our `skills/` playbooks should cover.

- **IOAI 2025** (At-Home Round, Individual Contest, GAITE)
  https://github.com/IOAI-official/IOAI-2025
- **IOAI 2024** (On-Site Round)
  https://github.com/IOAI-official/IOAI-2024

## How this maps to our repo

| Reading | Where it lands here |
|---|---|
| Rules PDF | `DESIGN.md` (rules-compliance mapping), harness budgets/trace |
| Practice tasks | `workspace/<slug>/` runs; `tasks/task1_audio/` legacy pipeline |
| IMO25 paper | inner experiment loop (`src/agent_loop.py`, EXPERIMENT phase) |
| MLE-bench | outer agent design benchmarking (`agent/`) |
| Past IOAI repos | `skills/` playbooks per task family, `memory/lessons/` |
