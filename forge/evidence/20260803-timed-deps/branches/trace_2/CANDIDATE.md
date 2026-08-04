# trace_2 — geometric group-balanced incremental AST

## Hypothesis

The supplied 16-class AST already contains useful audio embeddings and a calibrated
old-class head. Freezing both, then learning only 13 new dense rows in the supplied
head's post-LayerNorm geometry, should retain old-class behavior while fitting new
classes quickly. A 50:50 old/new objective matches the leaderboard metric more
closely than raw-row training.

## Candidate

- One deterministic 10.2-second center waveform view per sample.
- One supplied AST encoder call per sample.
- Frozen AST encoder and complete original LayerNorm+dense head.
- Geometric new-row initialization: class-centroid directions with the median
  original row norm and center-matched biases.
- Training weights: 70% old/new group-micro balance plus 30% within-group
  class-macro balance.
- Three-fold stratified OOF selection of one shared new-logit scale and shift.
- Final fit uses every labeled old and new row; inference returns one `[B,29]`
  tensor and preserves template path order exactly.

## Live-data audit

- `train.csv`: 296 rows, labels 0–15.
- `fine_tune.csv`: 624 rows, labels 16–28.
- `submission.csv`: 363 unique paths.
- No path overlap among old train, fine-tune, and test CSVs.
- Checkpoint dense row shape: `[16,768]`; classifier also has a frozen 768-wide
  LayerNorm.

## Validation performed locally

- Python 3 syntax compilation passed.
- CSV label ranges, row counts, path disjointness, and test-path uniqueness passed.
- Safetensors-header architecture/shape audit passed.
- No model execution was attempted locally because the live contract has no local
  GPU budget and the offline Kaggle environment is installed only inside the
  submitted kernel.

## Runtime estimate and risks

Expected T4 time is 4–8 minutes: about 40 seconds for official offline-wheel setup,
then one frozen AST pass over 1,283 samples; head fitting is only 768×29 linear
optimization and should be seconds. Main risks are CPU audio preprocessing on
unusually high-sample-rate WAV files, possible batch-16 T4 memory pressure, and
OOF-to-full-fit calibration drift. Center cropping is deterministic and bounded but
can miss a short event in unusually long recordings.

This branch does not push a Kaggle kernel or submit predictions.
