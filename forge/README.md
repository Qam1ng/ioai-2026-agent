# FORGE v2 — the timed-session lineage

This is not HearSay. `native/` is four Claude Code sessions given hours on a
Kaggle mirror; this is one Codex session given **thirty minutes** on a live IOAI
practice task, under a policy that decides what to spend the next minute on.

It is here because it is what actually ran the official timed slot.

## What it scored

IOAI 2026 AI Models Track, Practice Task 1 (timed dependencies), 2026-08-03.
Window opened `13:59:42Z`, closed `14:29:42Z`.

| submission | node | public |
|---|---|---:|
| 55213834 | fresh 13-row head, one class-stratified holdout | 0.82770 |
| **55213957** | **geometric init, group-balanced objective, three-fold OOF** | **0.83878** |
| 55214130 | holdout-selected duration, full 920-row refit | 0.83232 |

Provisional rank 3 of 9 visible teams at report time. Private score never
published. Direct cost `$0.00` — Kaggle-provided practice compute.

**No clean-benchmark claim is available on this task**, and the run says so
itself: the checkpoint and template hashes match historically exposed practice
material. Historical predictions and historical test structure were not used,
but "we did not use it" is not the same evidence as "we could not have".

## The shape of the thing

Thirty minutes is short enough that scheduling *is* the method. The policy
(`config/forge_v2_policy.json`) fixes:

- **three forced traces**, `trace_1..trace_3`, round-robin until each has been
  tried at least once, so nothing gets one idea and rides it into the wall;
- **three fidelities** — smoke at 60s for contract and gross signal, proxy at
  180s for cheap validation, full at 600s reserved for finalist evidence —
  because a wrong answer at 60s is worth more than a right one after the bell;
- **cost-aware contextual UCB** over what to run next, weighting uncertainty
  0.35, novelty 0.15, urgency 0.10, risk 0.25;
- **an archive** of one champion, two challengers and two stepping stones with a
  two-round TTL, so a promising dead end survives long enough to be recombined
  and no longer;
- **shared memory capped at 24 entries**, successes and failures both, and
  audit-only content forbidden from entering it.

The hard gates are the part that does not bend:

    audit-sensitive content may never be visible        compliance must pass
    output contract must pass                           leakage must pass
    a duplicate prediction cannot promote               full fidelity required for champion
    an existing machine gate is required for a finalist the public leaderboard is not in the inner loop

## What actually ran, and what did not

Be precise about this, because the two are easy to conflate.

`SESSION_STATE.json` pins `search_policy_sha256`, and the file in
`config/` still hashes to it:

    3d093e66b4d927d34b9dde0cd7780bdf70827ced8181080df360971462859572

So the policy was the binding contract for the session, and the run's shape
matches it — `branches/trace_1`, `branches/trace_2`, the 46-column ledger, the
proposer/reviewer split on every scored row.

`forge_v2_controller.py` is the reference implementation of that policy and it
left no artifact in the run directory: no controller state, no archive file.
Read it as the executable specification of the scheduling contract the session
followed, not as a process that was in the loop. The policy itself says
`controller_may_write_experiment_ledger: false`; the ledger is the agent's.

## Layout

    SKILL.md                    the operating contract read before any task action
    SYSTEM_PROMPT.md            preloaded harness policy
    config/forge_v2_policy.json the strategy, hash-pinned by the live run
    forge_v2_controller.py      its reference implementation
    templates/                  the experiment ledger header, 46 columns
    tests/                      7 checks
    evidence/20260803-timed-deps/   the live run, everything but the data

Run the tests:

    python forge/tests/test_forge_v2_controller.py

## The evidence directory

Everything the timed run sealed, minus the 1.5 GB data bundle and the
submission CSVs. `TRACE_MANIFEST.json` binds the selected solution, its output,
its metrics, its review and the final report by hash, and records
`credentials_included: false`. `experiment_ledger.csv` carries both nodes with
their proposer, reviewer, review status, code hash, split hash and output hash.

The three later lineages — `forge_v3` through `v5`, and FORGE-HearSay v6 — are
not here. v6 is what ran the IOAI 2025 Kaggle mirrors the same day, which is a
different format with a different budget and deserves its own branch if anyone
wants it.
