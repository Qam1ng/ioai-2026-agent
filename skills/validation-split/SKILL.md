# Design the validation split for this task

You decide how the run's one shared split is cut. Nothing here is mandatory —
it is what has gone wrong before and what usually works.

## What the split is for

Score a model on the data it trained on and the number is a memory test. So
hold data out, predict it with a model that never saw it, and score that. Every
candidate in the run is scored on the same split, so the numbers are comparable
and the harness can rank them. That is the whole reason it is yours to fix once
rather than each solver's to choose.

The failure this exists to prevent, measured on our own runs: local CV 0.9156,
leaderboard 0.78095. A split chosen to flatter, or one that leaks, tells
everybody the wrong thing for the rest of the run.

## The contract

`folds.json`, in the workspace root:

    ids     one entry per evaluation unit, in the order predictions take
    fold    same length. >= 0 is a scored fold; -1 means train-only
    labels  same length — ground truth passed to metric.score
    scheme  short name for how you cut it
    reason  one line; required for anything but a plain k-fold

Check it: `python -m native.scripts.checkfolds --workspace .`
It is written once. Do not re-cut it after candidates have been scored.

## Choosing k

Five folds give a stable mean and a spread, at the cost of five training runs.
That is usually right and usually affordable.

It stops being affordable when one training run is slow enough to eat the
window. Then mark most units `-1` and keep one honest holdout — say so, because
a single holdout is much noisier and the people reading the board should know
which kind of number they are looking at. Budget backwards from the deadline:
if five runs do not fit before the first submission is due, do not pretend they
do.

## Choosing how to cut

Ask what the shuffled split would leak, and cut along that instead.

**Time.** If rows are ordered in time and the task is to predict later from
earlier, a shuffled split lets every model see its own future. Cut
chronologically: earliest block trains, later blocks score.

**Repeated subjects.** Frames from one video, images of one patient, rows from
one session — near-identical items on both sides of the split make every model
look better than it is. Group them so they never straddle a fold. Check for this
rather than assuming: hash or correlate the items and look for clusters.

**Rare classes.** With a handful of examples in a class, an unstratified split
can leave a fold with none of them, and the per-fold scores become noise.
Stratify.

**A provided validation set.** Some competitions ship one. Use it — mark the
training rows `-1` and the provided rows fold 0. Inventing your own split
alongside one the organisers designed usually means measuring something else.

**Nothing structural.** Plain shuffled k-fold.

## When the unit is not a table row

The generator script assumes one labelled row per unit. Many tasks are not like
that — detection has many boxes per image, segmentation labels are masks,
retrieval is scored per query. Decide what the *evaluation unit* is (whatever
the metric consumes one of), and write `folds.json` directly with one entry per
unit. The script is a convenience for the common case, not the definition.

## Quick path for a labelled table

    python -m native.scripts.folds --labels <csv> --target <col> \
        [--id <col>] [--group <col>] --k 5 --workspace .

    # anything it cannot express:
    python -m native.scripts.folds --labels <csv> --target <col> \
        --custom my_folds.json --reason "why" --workspace .

`--custom` takes `{"fold": [...], "reason": "..."}` and is how temporal splits
and single holdouts get expressed.

## Choosing hyperparameters without spending the split

The folds are what everyone is scored on, so anything chosen by looking at
fold scores is chosen on the test of record. Picking the epoch with the best
fold score is the common version and it is a real leak: the reported number
then includes the best of however many epochs you looked at, which is not what
a fresh run would give.

Fix the schedule before you score. Decide the epoch count from a training
curve, a held-out slice inside the training part, or a fixed budget — then
train that many and report it. If you must search, search inside each training
fold and leave the scored fold untouched. Say on the board which you did; a
number tuned on the folds is not comparable to one that was not.

## Before you call it done

- Fold sizes roughly even, no empty scored fold.
- Nothing in a scored fold has a near-twin in the training part.
- The label distribution per fold looks like the whole.
- Post the scheme and the reason to the facts board. Everyone's numbers come
  from this; they should know what it is.
