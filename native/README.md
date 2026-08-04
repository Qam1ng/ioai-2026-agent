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

## Data synthesis and the ruler

`chicken_data_synthesis.py` owns one component: what rows go into the fit, and
which ruler is allowed to say whether they helped. It does not touch the
estimator. The scaler, Ridge, KNN, log-Ridge, their hyperparameters, the
0.98055 calibration and the 70/30 blend are the incumbent's, and the self-test
asserts each constant so the boundary cannot drift.

**Exact labels, for free.** Every train frame ships a 180x320 density map whose
sum *is* the label, and the frozen pipeline resizes the image to 360x640 —
exactly twice the density grid. Any sub-window therefore carries an exact
count: integrate the map over it. 100 frames become 3,200 exactly labelled rows
with no external byte and no hidden label. Crops are taken at native scale so a
chicken keeps its pixel size and only the field of view shrinks, and the target
is area-normalised, which puts crops and full frames on one regression scale.
Synthesis refuses to run unless its own full-frame path reproduces the sealed
feature cache bit for bit; it does, to `0.0`.

**Then the ruler threw all of it away.** Every recipe scores *below* the
incumbent's 0.920087 at live anchor density — crops −0.0024 to −0.0042, mirrors
−0.0047. Mirroring is the interpretable one: this is a fixed CCTV frame, so a
mirrored coop is a viewpoint the camera cannot produce. The synthesis is real
and the rejection is the finding.

**The ruler is measured, not chosen.** Which holdout to score on is a property
of the data, and it is checkable without a single label: compare how far a
held-out frame sits from its fitting set to how far the live test frames sit
from the whole labelled set.

    live test -> train      median nearest-neighbour distance   52.71
    leave-one-out                                               59.41   gap  6.71
    interleaved 10%                                             59.84   gap  7.13
    grouped 5-fold                                              62.62   gap  9.91
    group holdout, 3 blocks                                     64.59   gap 11.89

Test frames sit *closer* to the labelled set than labelled frames sit to each
other, because 200 test frames are interleaved with 100 train frames across the
same nine capture sessions. No train-only holdout can be as dense as the real
thing, which is why local 0.8908 reads against public 0.93041; leave-one-out is
the closest available and is what the component scores on.

**Protocol parity is the whole point.** The previous chicken candidate was
scored leave-one-out against a baseline scored five-fold grouped, and reported
the difference as improvement:

    candidate, leave-one-out                    0.927325
    baseline, grouped five-fold                 0.890849   -> claimed  +0.036475
    baseline, leave-one-out                     0.920087   -> matched  +0.007238
                                                   inflation removed  +0.029237
    what the board actually paid                                      -0.010850

Eighty percent of that claim was the two sides being measured with different
rulers. The residual +0.0072 is then rejected on its own evidence — block
bootstrap over acquisition blocks puts P(positive) at 0.935 and the 95% lower
bound at −0.0022 — so the same protocol that produces candidates also declines
the one already known to have cost 0.011. A validator that cannot reject a
known-bad candidate has not been shown to work, so that candidate is replayed
as a standing negative control on every run.

**The two rulers were then made to disagree in public.** Retrospective
agreement with a result you already know is weak evidence, so one submission
slot went on the recipe where the rulers point opposite ways. Grouped five-fold
promotes `crop_wide_mid` at +0.00894; leave-one-out rejects it at −0.00341. The
predicted score was written into the sealed receipt and hashed *before* the
file was sent:

    registered prediction        0.92700  (band 0.005)
    public score, 55228496       0.92477
    absolute error               0.00223
    account best it had to beat  0.93041

The board sided with the calibrated ruler, 0.00564 below the incumbent, and the
prediction landed inside its registered band. `seal --falsification-test` is
the only door to that experiment: it refuses a recipe that *passed* the gate,
records `promotion: false`, and will not write a verdict that was not
registered in advance.

Scarce-anchor behaviour is still recorded — six stress scenarios across five
seeds, where crops *do* help by up to +0.021 — but it is diagnostics. Selection
reads the calibrated ruler and nothing else.

    python -m native.scripts.chicken_data_synthesis prepare  --run-root <r> --source-root <s> --timestamps <t>
    python -m native.scripts.chicken_data_synthesis synthesize --run-root <r> --source-root <s>
    python -m native.scripts.chicken_data_synthesis validate --run-root <r>
    python -m native.scripts.chicken_data_synthesis seal     --run-root <r>   # fails closed when nothing clears
    python -m native.scripts.chicken_data_synthesis audit    --run-root <r>
    python -m native.scripts.chicken_data_synthesis record   --run-root <r> --submission-ref <id> --public-score <s>

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
