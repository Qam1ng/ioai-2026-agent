---
name: ioai-high-score-agent
description: >
  Run an autonomous, multi-task IOAI AI Models Track session using shallow
  multi-branch ML search, skeptical validation, strict live-contract submission
  custody, dynamic time allocation, and complete trace preservation.
compatibility: Formal runs require the pinned Codex launcher and live official submission adapter. Local provider pilots may use the sealed Claude Code runner; Practice submission additionally requires Kaggle CLI authentication.
metadata:
  author: local
---

# IOAI High-Score Agent

Use this skill for IOAI practice, historical tasks, and formal AI Models Track
sessions. Resolve the submission backend from each live task. For current Practice
Kaggle mutations, read and obey `kaggle-cli-submission`. For the formal event,
require the official MCP starter-kit adapter and fail closed if it is absent.

This policy is a preinstalled, human-written scaffold. Record it in the trace. At a
formal session, the human may provide only IOAI's exact short starting or
continuation prompt; never ask the human to paste this policy or fill runtime
strategy fields.

## 1. Freeze the contract before solving

Create a session state file before code changes. Record:

- wall-clock start, hard stop, and finalization gates;
- every competition slug and working directory;
- current rules, metrics, allowed resources, daily/session submission limits, and
  code-submission mechanism;
- hardware, agent-slot, Kaggle-run, and network constraints;
- trace location and last-known-good candidate for every task.

Never import a rule from an earlier task. Read each live task independently.

Create the trace directory before any model/tool action. Preserve timestamped model
inputs/outputs, tool-call inputs/outputs, tool definitions, all subagent traces,
code-execution outputs, installed-library/environment manifests, and every human
system/launch prompt. Seal only after the root rollout and every successfully
spawned subagent rollout are present. Hash the sealed trace. Redact credentials and local IP
addresses before sharing it; never omit them by silently dropping the surrounding
event.

For a local Claude Code model-provider pilot, use
`tools/claude_three_route_runner.py`; do not replace the formal launcher in place.
It must authenticate only through `ANTHROPIC_API_KEY`, set
`CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`, capture the resolved model from the root
stream, and archive `root.jsonl` plus `trace_1.jsonl`, `trace_2.jsonl`, and
`trace_3.jsonl`. The subagent architecture is root-relay: sibling routes do not
communicate directly. Record semantic adoption, rejection, conflicts, and
cross-route relays in `route_coordination.json`. A Claude provider pilot has no
submission authority; only a separately valid live task contract and existing
machine submission gate may authorize its exact sealed candidate.

## 2. Use all four slots intentionally

The root is the Session Director. Spawn at most three Task Leads, one per formal
task. Each Task Lead uses three modes:

- `Planner`: proposes a falsifiable hypothesis and validation plan.
- `Coder`: implements one atomic change in an isolated run directory.
- `Executor`: runs preflight, training, evaluation, and output checks.

Do not let the same reasoning pass both propose and approve a risky candidate. A Task
Lead waiting on a remote run must perform an independent critic pass for another
task when useful.

## 3. Mandatory task audit

Before selecting a model, inspect all official inputs, not only CSV columns:

- label support, missing values, split/group/time/subject structure;
- file order, IDs, naming patterns, archive layout, and sample-submission order;
- raw media headers, shapes, durations, channels, sample rates, resolutions, and
  acquisition artifacts;
- exact duplicates, near duplicates, source families, and train/test distribution
  shifts;
- supplied checkpoint architecture, preprocessing, training args, label mapping,
  parameter groups, and provenance;
- metric direction, group weighting, class weighting, and final ranking mechanism.

Run leakage and compliance review on every discovered shortcut. Competition-provided
structure may be modeled only when the rules allow it. Never hard-code inferred test
labels or use human labeling.

## 4. Baseline before search

Produce one cheap, valid, end-to-end candidate per task. It must:

- run from a clean process;
- use only allowed data/models/dependencies;
- generate the required output inside the live submission runtime;
- pass IDs, order, columns, shape, type, label/range, NaN, and duplicate checks;
- be completed, retrieved for verification when supported, and submitted according
  to the live official adapter;
- be stored as the task's immutable last-known-good candidate.

## 5. Directed multi-trace search

Use the SHA-pinned `config/forge_v2_policy.json` sidecar controller. Start exactly
three forced-diverse traces whose task-specific root axes and mechanism families are
declared in the active task config. Each proposal must state a diagnosis, causal
hypothesis, one atomic change, expected observables, and a falsifier. ReAct or
Plan/Execute may implement and debug a selected proposal, but they do not choose
which experiment receives the next shared-compute minute.

Every execution returns structured feedback: score and robust-utility changes,
per-group failure deltas, variance, runtime, code/output bindings, compliance and
leakage status, plus a typed textual gradient explaining the observed failure,
causal update, next atomic change, expected observable, and falsifier. Only
`solver_shared` memory may cross traces; audit-only test structure never may.

The controller uses a cost-aware contextual-UCB priority rather than raw score
alone. It protects one full-fidelity champion, keeps at most two near-Pareto
challengers, and may keep at most two time-limited stepping stones that improve a
specific failure group despite a bounded robust-score regression. A stepping stone
expires after the configured number of rounds. It is evidence for one further
child, not a promotion.

Use `smoke -> proxy -> full` multi-fidelity screening. Full fidelity is required for
the champion, finalist, or promotion request. Hyperparameter search is a child
operation inside an already viable pipeline and should use early stopping or
successive-halving logic rather than a full grid.

FORGE v3 is an opt-in search-policy experiment, not yet the formal-session default.
Its schema-2 controller profiles are `aide_tree`, `mcgs_graph`, and `pes`. Compare
them as separate treatment arms under
`tools/agent_system_benchmark_validator.py` before selecting or combining them.
The model, effort, tools, task bundle, evaluator, wall/token/compute limits, and
blindness contract must be identical; only the policy/implementation/adapter and
isolated runtime paths may differ. A contaminated task is audit-only and cannot
rank agent systems.

FORGE v4 is a separate schema-3 treatment arm in
`tools/forge_v4_controller.py`; it does not modify v2/v3 behavior. It combines a
small signature-deduplicated DAG with a behavior-cell archive and an
operator-level UCB scheduler. Every proposal declares one operator and one
behavior cell (`representation`, `validation_data`, `objective_calibration`, or
`inference_ensemble`) at its current fidelity. Operator credit is measured as
hard-gate-passing robust gain per minute; invalid runs receive negative credit.

Two-parent fusion is exceptional, not a default branching trick. Both parents
must be ledger-bound proxy-or-better results on the same split, have distinct
prediction hashes, traces, and behavior mechanisms, and provide a hash-bound
independent review showing sufficient error disagreement, bounded residual
correlation, and positive expected complementarity. The controller also refuses
to admit work whose p95 runtime plus queue, retry, and finalization reserves no
longer fit the deadline.

Treat v4 as experimental until a seal-before-unblind comparison shows a robust
gain. A single fresh split on a previously studied task is only a reset pilot;
promotion to the formal default requires balanced-order tests on multiple tasks
that were not used to design v4.

FORGE v5 is a separate schema-4 treatment arm in
`tools/forge_v5_controller.py`; it preserves schema-3 behavior and adds an
evidence firewall around v4 search. Every proposal/result must declare a clean,
unique process and trace. If a process touches forbidden audit data, run
`taint-process` immediately: that process, its trace, descendants, parent use,
review use, and shared-memory influence are quarantined.

Record leaderboard outcomes only with `record-reveal`. A negative outcome blocks
blind full-fidelity siblings in the same `candidate_family`; `submission-gate`
can lift that block only with independent positive validation bound to the
candidate's raw OOF prediction hash and frozen split. Never copy a leaderboard
score into clean metrics or operator credit.

Operators that retrieve or apply a template prior require a hash-bound,
sample-local receipt passing permutation, strict-subset, and rebatch checks.
Fusion additionally requires raw OOF hashes/scores for both parents, cross-fitted
base features, no same-fold tuning of base features, and a clean independent
reviewer. Use `config/forge_v5_policy.json` and the `templates/forge_v5_*`
payloads. V5 remains a treatment arm until it wins seal-before-unblind,
balanced-order tests on untouched tasks; it cannot replace the existing machine
finalist/submission gate.

Append executable nodes only through `tools/ledger_append.py`, using
the exact `templates/experiment_ledger.csv` header. Every row must be bound to the
supplied task config by both competition name and `task_config_sha256`; completed
rows also carry code/split/prediction/output hashes. Promotion/finalist/submission
rows require finite primary, secondary, normalized-utility, variance, runtime,
memory, token, and cost values; independent review; compliance/output passes; a
completed run; a canonical `seed_set`; a sealed trace hash; and hash-bound output,
evaluator, review, and promotion-evidence receipts. The promotion evidence must bind
the exact node, code, split, task config, metrics, predictions, reviewers, and
decision, so it cannot authorize a different ledger row. The v2 controller is a
read-only ledger consumer and cannot execute code, write the ledger, authorize a
finalist, or submit. Submitted rows additionally
require the live backend, immutable run reference, and backend-specific version
identifiers.

After two causally redundant non-improving children, stop, pivot, or fuse the
branch. Do not continue cosmetic tuning. A non-improving child with a verified
failure-group gain may receive the strictly bounded stepping-stone exception.

## 6. Skeptical validation

The evaluator must be harder to fool than the model search:

- use task-appropriate group/time/source/subject splits;
- separate model selection from calibration when feasible;
- require multiple seeds for finalists;
- report per-class and per-risk-group metrics, not aggregate only;
- veto class collapse, data leakage, test-fitted preprocessing, unused official
  inputs, and prediction/output duplicates;
- use teacher agreement or invariants when labeled validation is contaminated by
  checkpoint provenance;
- rank by a robust lower bound, not one best fold.

Public leaderboard results are sparse evidence. Never optimize every node against
them. Promotion and final selection require `reviewer != proposer`.

For ordinary classification finalists, generate and hash-bind a
`tools/structure_prior_gate.py` receipt. The same sample predictions must survive
permutation, strict-subset, and rebatch checks. A semantic-sequence dependency
requires a separate task-specific contract; any submission/file/path-order or
otherwise dataset-global dependency is audit-only and cannot be promoted.

When a task has been exposed by inferred hidden-label structure or equivalent
contamination, fail closed: set `submission_allowed=false`, isolate the audit
artifact from Planner/Coder context, and use the task only for an explicitly marked
postmortem. A contaminated task cannot be used to compare agent systems.

## 7. Time-aware control

For a six-hour session:

- T+15 minutes: finish global triage.
- T+75 minutes: every task has a valid candidate.
- T-120 minutes: stop creating new root branches and enter finalist review.
- Per task: stop admitting final runs at
  `T - (p95 runtime + queue margin + retry reserve)`.
- T-20 minutes: only submit, poll, recover, and seal evidence.
- T to T+30 minutes: finalize and submit one short technical report per problem.

If a task is blocked, preserve its baseline and reallocate effort. The Director must
not allow one task to starve another.

## 8. Failure containment

- Isolate every candidate in its own directory or branch.
- Set hard timeouts for jobs.
- Allow at most two debug attempts for the same failure.
- Checkpoint after every node.
- Preserve a complete, valid candidate at all times.
- Detect stuck jobs, orphaned processes, exhausted disk/GPU, authentication errors,
  and submission quota exhaustion.
- Roll back instead of rewriting the only working solution.

## 9. Final selection and custody

Select finalists from actual validation evidence. Ensemble only when OOF errors are
complementary and the final code remains compliant and reproducible.

Before submission:

1. freeze code/config/seeds;
2. rerun from a clean process with internet disabled;
3. verify all output and resource contracts;
4. archive code, metadata, logs, predictions, hashes, environment, and trace;
5. submit through the live backend: a completed Kaggle kernel for current Practice,
   or the official IOAI MCP tool for the formal event;
6. poll scoring/status and retain a correction reserve where rules require it.

At hard stop, report only confirmed scores and ranks. Mark private scores or
unavailable signals explicitly.

For every problem report iteration time, dollar cost, input/output tokens,
code-execution tools/environments, Internet, retrieval/search, libraries categorized
as allowed/external/disallowed, multi-agent use, private tools, prompts/scaffolding,
and any other material resource. Start drafting during the session so the final
report can be submitted within the post-session 30-minute deadline.

For tasks requiring one model forward, enforce it structurally: one deterministic
input view per sample, one encoder call, and one module return containing the full
output tensor. Training-only teachers and several heads over the same embedding are
allowed only when live rules permit them. TTA, multiple encoders, dataset-global
test assignment, and row/path-specific prediction overrides fail compliance.
