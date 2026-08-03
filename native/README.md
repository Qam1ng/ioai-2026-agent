# HearSay

Three full-power Claude Code sessions solve a Kaggle competition in parallel,
coordinating through one append-only facts board, with a deterministic harness
that owns measurement and submission.

    python -m native.main --mode competition --slug <competition> --solvers 3 \
        --deadline-min 120 --max-cost-usd 100 --max-submissions 4

Results on the IOAI 2025 mirrors, first place on both:

| task | local (OOF) | leaderboard | previous best |
|---|---|---|---|
| `radar-ioai-2025` | 0.98601 | **0.98756** | 0.98723 |
| `ioai-2025-chicken-counting-mirror-unofficial` | 0.918652 | **0.93156** | 0.93002 |

Both margins are small enough to sit inside test-set sampling noise (radar's is
~0.00036 by bootstrap), so read them as "competitive with the best public
entry", not as a demonstrated lead. The reproducibility is the stronger claim:
two independent radar runs landed at 0.98756 and 0.98716, well inside that same
noise band.

## The bet

A tuned agent kernel left intact, inside a hard deterministic harness, beats a
constrained kernel inside a smart soft one.

The previous branch replaced Claude Code's preset system prompt, passed
`setting_sources=[]`, and gave subagents a tool whitelist — three ways of
crippling the thing we were paying for. Here the preset is kept and appended to,
settings load normally, the built-in toolset is untouched, and every constraint
that matters is enforced by a script the model cannot talk its way past.

Concretely: solvers are given no assigned approach, no methodology, and no
submit tool. They read the task, decide where they think it is won, and write
two files. Everything about whether those files are any good is computed
elsewhere.

## Evidence Firewall

The run now starts in one of two frozen modes:

- `competition` permits sparse leaderboard calibration only after a submission
  has been sealed and sent.
- `clean-benchmark` withholds leaderboard and web feedback from every solver;
  an attempted audit read permanently taints that process lineage.

`run_contract.json` pins the mode, model, routes, budget, deadline, and submit
authority. Candidate promotion binds folds, OOF predictions, submission,
code/config hashes, sample-locality probes, and the route-coordination receipt.
Development, clean, and public LKG channels are separate: evidence cannot be
promoted merely because a public score happened to rise.

## Agents

Four independent `ClaudeSDKClient` sessions, each its own process and context.

**`evaluator`** — compiles the ruler once (`metric.py`, self-tested against
hand-computed values; `folds.json`, the split every candidate is scored on),
then stays on as the submission gate. It is the right one to hold that gate
because it is the only agent with no candidate of its own; when solvers held it,
all three bought insurance inside four minutes and spent a whole day's quota
before the metric existed.

**`solver_a/b/c`** — identical contracts differing in nothing at all. Division
of labour is theirs to negotiate on the board. An earlier version assigned one
to the model, one to the data, one to calibration, decided before anybody had
read the task; on chicken counting the win was that the metric is asymmetric and
the target is the density map's sum, which is none of those three.

## The board

`facts.jsonl`, append-only, no LLM. Delivered at every round boundary and
piggybacked onto tool results.

    claim     the angle you are taking, so others take a different one
    adoption  reuse of another route's stable fact ID
    conflict  two findings that need an explicit resolution
    result    a score — evaluator and harness only, never a solver
    decision  evaluator-only resolution bound to evidence
    data      a structural property: a leak, duplicates, a label anomaly
    format    a submission trap
    failure   something confirmed not to work, and why
    env       an environment gotcha

`result` is provenance-gated rather than secret: a number is comparable only if
it came from the shared folds and the frozen metric, and a self-reported one is
not. Everything else is open, including what each direction turned out to be
worth — withholding that only means the others keep drilling dry holes.

Every record has a stable ID and a hash-chain predecessor. `adoption` must point
to another route; self-adoption and invented references are rejected. The final
`route_coordination.json` therefore distinguishes actual cooperation from three
independent end-to-end races instead of inferring teamwork from shared logs.

Two layers. `task` dies with the competition; `day` survives across the three
problems of a competition day.

## Submission

Four gates, and every one of them has a point past which it cannot refuse —
three separate runs were lost to gates that could say no forever.

1. the ruler exists and `--min-candidates` candidates carry a script-computed
   score *(override: past 55% of the window, send the best there is)*
2. quota and throttling; must beat what was already sent
3. mechanical evidence — format, hash provenance, order-dependence
   *(only fatal findings block on their own)*
4. the evaluator's verdict: APPROVE / HOLD / REJECT
   *(override: 90s silence falls back to gate 3; HOLD expires at 55%)*

Gate 3 is evidence handed to gate 4, not a verdict. It used to be a verdict, and
a generic check flagged every pixel column of a radar submission for containing
-1 — where -1 *is* the background label — and blocked a 0.9816 candidate
eighteen times while the one agent that could have said so was never asked.

## Deterministic layer

    recon.py         scans the data before any agent starts
    folds.py         the common tabular case; a convenience, not the definition
    checkfolds.py    the split contract the harness actually enforces
    evaluate.py      scores out/oof.npy — self-reported numbers are never read
    promote.py       no-regression gate and last-known-good
    check_format.py  fatal vs "depends on rules this script does not know"
    integrity.py     evidence-manifest binding and order-dependence detection
    sample_locality.py  permutation, strict-subset, and rebatch invariance
    coordination.py  claims/adoptions/conflicts/results/decisions audit
    system_benchmark.py  equal-budget, reverse-order architecture A/B

`chicken_timestamp_candidate.py` is a concrete task adapter for the updated
policy: a DVR-time route is adopted per capture session only when its frozen
leave-one-out evidence beats the visual incumbent; a failed session keeps the
incumbent. It freezes all 300 official-image hashes, produces duplicate CSVs,
and refuses more than one exploratory submission.

`integrity.py` exists because a teammate's strongest public score on the audio
task came from Viterbi-decoding the submission ordering — 0.86 to 0.988 with no
better model — which they caught with an invariance test and recorded as
audit-only rather than banking. A pipeline exploiting row order has not learned
the task, and on IOAI it is the kind of thing that gets a submission thrown out.

## Supervision

Deterministic, and deliberately not an agent: every question here is a
comparison of an observed number against an expected one.

- **reconciler** — Kaggle's submission count against ours, GPU trespass,
  orphaned processes, cost. Blocks nothing; reports everything. Every
  behavioural fence built for the first two runs was walked around inside one
  run, and none of the breaches surfaced until the logs were read the next day.
- **turn watchdog** — `interrupt()` on four minutes of silence inside one turn,
  retried, then the session is declared unreachable and restarted.
- **lives** — a crashed or hung solver is rebuilt from its files and the board
  rather than lost for the rest of the run.
- **score watcher** — collects the leaderboard score and posts the gap. A solver
  that knows its out-of-fold score runs just under the leaderboard can trust it;
  one told the gap is large and the wrong way round should stop believing its
  own numbers, which is the situation that produced 0.9156 local against
  0.78095 on the board.

## Monitoring

    ./watch                       # most recent workspace
    ./watch <slug> -l             # follow, 20s refresh
    python -m native.monitor --slug <slug> --json

The board first, because that is where everything ends up. A gate that has
refused the same thing three times is surfaced above it — a gate that keeps
refusing is more likely wrong than the run is.

## Tests

    python -m native.selftest     # 163 checks

Against hand-computed values, not against themselves. If `evaluate.py` is wrong
every downstream decision is wrong and nothing else in the system can notice.
The suite exists because it caught `promote.py` trusting a score file that lives
inside the solver's own directory — "self-reported scores are never believed"
was a comment rather than a property.
