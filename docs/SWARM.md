# The swarm: multi-agent architecture

The single-agent harness in `agent/` proved the path end to end: read a task,
write code, push a kernel, get a score. This layer answers the next question —
what do you do with six hours, three problems, and effectively unlimited
compute on your own side of the wire.

The reference point is [AIBuildAI](https://arxiv.org/html/2604.14455) (63.1%
medal rate, first on MLE-bench as of March 2026): a manager agent emitting one
action at a time over designer / coder / tuner sub-agents, with several
solution candidates alive at once and an aggregator at the end. We take its
control structure and change what the competition forces us to change.

---

## What the rules actually constrain

From `docs/QA-NOTES.md` (organizer answers — these override any guess):

| Resource | Limit | Is it binding? |
|---|---|---|
| Wall clock | 6 h per day, 3 problems | **Yes** — the hard one |
| Submissions | 50 per problem | No, if runs are cheap. Plan to spend all of them |
| Kaggle GPU quota | 30 h/week, **all 3 problems share one account** | **Yes** — the real scarcity |
| Kernel runtime | no enforced clock guard | Soft; discipline still expected |
| Agent-side compute | unrestricted | **Not a constraint — it is our lever** |

Two consequences the architecture is built around.

**The submitted artifact is a `.py` kernel that trains on Kaggle.** No uploaded
checkpoints, no internet, fixed package list. So "a candidate" is not a trained
model — it is a self-contained training script that must fit the kernel budget.
Everything the swarm does locally is in service of writing that script.

**GPU quota, not the submission count, is what runs out.** A CPU kernel costs
zero GPU quota. That single fact creates the two-lane submission economy below.

### Why 50 submissions is a target, not a ceiling

The naive estimate — six hours divided by a 30-minute run — gives about a
dozen, and that is what a first reading suggests. It is wrong, because
per-submission runtime is a variable *we choose*, not a constant the organizers
impose:

- A probe kernel that emits a constant prediction finishes in well under a
  minute and costs no GPU quota. Throughput is then bounded by Kaggle's
  concurrency, not by our budget.
- A full training kernel at ~20 min against a ~10 GPU-hour per-problem share
  allows roughly 30 GPU submissions before quota bites.

So the plan is to spend all 50, with the split enforced by
`swarm/submit/broker.py`: a **fast CPU lane** for information and a **heavy GPU
lane** for genuine candidates.

### The one place where more submissions hurt

Final ranking takes our **best submission on private test A and re-scores it on
private test B**. Selection therefore happens by `argmax` over a noisy
measurement. Flooding the leaderboard with near-identical variants raises the
chance that the argmax lands on a *lucky* one, which then regresses on B — the
winner's curse.

The resolution is not to submit less, it is to submit *differently*:

- Spend the fast lane freely on probes that are **structurally informative**
  (base rate, test-set size, distribution shift, LB noise floor, calibration).
  These do not compete for the top of the leaderboard.
- Keep the heavy lane for candidates that are **genuinely different**, not
  reseeds of one idea.
- Choose the final entry by a **shrunk** estimate — leaderboard score pulled
  toward what the local-CV calibration predicts, in proportion to the LB noise
  band (`swarm/ledger.py::shrunk_estimate`) — rather than raw argmax.

---

## Topology

```
 Portfolio (one competition day, one Kaggle account, one GPU quota pool)
   │  policy: insurance first, then feed the margin
   ├── Pod: problem 1 ─┐
   ├── Pod: problem 2 ─┼── each owns a 6h clock, 50 submissions, its own blackboard
   └── Pod: problem 3 ─┘

 Inside a pod
   Profiler ──▶ task card (type, metric, constraints quoted verbatim, CV grouping)
   Designer ──▶ K plans, forced to span different method families
        │
   Manager ──▶ ONE action per step: design | code | tune | verify | probe |
        │      aggregate | retire | wait | stop
        ├── Coder   × N in parallel   (headless Claude Code, sandboxed per candidate)
        ├── Tuner                      (headless Claude Code)
        ├── Verifier                   (API path, adversarial: sees artifacts, not reasoning)
        ├── Compliance                 (API path, read-only, quotes the rule it enforces)
        └── Prober                     (designs the cheap CPU probes)
   Aggregator ──▶ final entry (no retraining from scratch)
   Reporter   ──▶ technical report + tools/efficiency declaration
```

**Coding roles run as real headless Claude Code processes** (`swarm/runners/`),
one per candidate, each confined to its candidate directory. A coding agent
that already owns file editing, a shell and its own test-and-fix loop beats a
hand-rolled tool loop at the one job that decides the score. Judgement roles
(Manager, Verifier, Compliance, Designer) stay on the plain API path where the
context is controlled and auditable — and where any model can serve, which is
what lets the runtime brain switch from Gemini 3.1 Pro in development to
Fable 5 / Opus 5 on the day without touching code.

---

## Division of authority

**The pod enforces what must never go wrong. The Manager decides everything
that needs intelligence.**

Deterministic, not negotiable:

| Gate | Default | Why it is not the Manager's call |
|---|---|---|
| Floor submission | T+45 min | Scores are normalised within the AI track; a zero cannot be bought back elsewhere. If the Manager has not submitted, the pod submits the trivial valid kernel itself |
| Freeze | T+5 h 15 m | No new directions; consolidate |
| Final buffer | T−20 min | Kernels queue; a submission started too late scores nothing |
| Significance gate | gain > noise band | See below |
| GPU reservation | before push | Three pods must not each believe they hold the last GPU-hour |

The asymmetry justifies the split: a Manager that picks a mediocre model loses
a few points, while a Manager that talks itself out of submitting scores zero.

## The significance gate

`memory/lessons/local-cv-optimism.md` records what this system already paid
for: a single 25% stratified holdout read **0.9156** while the leaderboard said
**0.78095**, and later iterations *regressed* by chasing phantom failures in
classes with one validation sample.

So a milestone submission requires the local gain over the current best to
exceed the calibration's noise band (`swarm/ledger.py`). The Verifier reports
mean **and** cross-fold variance, never a single holdout, and grouped folds
where the task card names a grouping variable. The calibration itself is
re-fit from every scored submission, and a large local-minus-LB gap raises a
warning that says the *validation design* is wrong — not that the model is good.

---

## Running it

```bash
python -m swarm.cli doctor                      # pre-flight
python -m swarm.cli run --slug <comp> --dry-run # local rehearsal, never touches Kaggle
python -m swarm.cli run --slug <comp>           # one problem
python -m swarm.cli day --slugs a,b,c           # a full competition day
python -m swarm.cli status --slug <comp>        # blackboard snapshot while it runs
```

`run` is safe to invoke twice — state lives on disk and the budget clock does
not reset, which matches the organizers' start-prompt / continuation-prompt
protocol and prevents a resumed run from silently claiming a second budget.

Model routing lives in `configs/default.yaml`; environment variables win, so
competition day is `SWARM_BACKEND` / `SWARM_MODEL` / `SWARM_BASE_URL` and
nothing else.
