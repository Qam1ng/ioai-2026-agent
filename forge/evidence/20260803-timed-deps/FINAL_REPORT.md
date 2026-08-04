# IOAI 2026 AI Models Track — timed dependencies practice

## Outcome

- Competition: `ioai-2026-ai-models-track-practice-task-1-timed-deps`
- Participation status: joined, accepted the competition terms, and submitted through the official Kaggle notebook backend.
- Best confirmed submission at report draft time: `55213957`
- Best confirmed public score: `0.83878`
- Leaderboard snapshot at `2026-08-03T14:25:53Z`: provisional rank 3 of 9 visible teams.
- Private/final score: not published; public score only.
- Clean benchmark claim: prohibited because this task's checkpoint/template hashes match historically exposed practice material. Historical predictions and historical test structure were not used in this run.

## Contract

- Supplied AST checkpoint only; no external pretrained audio model.
- Frozen AST encoder and immutable original 16-class head.
- One deterministic waveform view and one AST encoder call per sample.
- Complete 29-class output from one shared representation.
- Offline Kaggle kernel on one Nvidia T4.
- Exact `submission.csv` contract: 363 rows, columns `path,target`, template path order unchanged, integer labels 0 through 28.

## Submitted candidates

1. `55213834`: fresh 13-row head with one deterministic class-stratified holdout and a train-only group offset. Status `COMPLETE`, public score `0.82770`.
2. `55213957`: geometric initialization, group-balanced objective, three-fold OOF calibration, and full-data refit. Status `COMPLETE`, public score `0.83878`. Selected incumbent.
3. `55214130`: holdout-selected training duration and offset followed by a full 920-row refit. Status `COMPLETE`, public score `0.83232`; rejected relative to the incumbent.

## Validation evidence

- Candidate 2 three-fold OOF balanced accuracy: `0.7806003`.
- Candidate 2 OOF old accuracy: `0.6621622`.
- Candidate 2 OOF new accuracy: `0.8990384`.
- Candidate 2 runtime after environment setup: `44.17 s`; Kaggle log reached output at approximately `111.50 s`.
- Candidate 2 output SHA-256: `4333c03f55b562bcc30237e0873621abf0baf74c2b2071c72d405b057f32c8e7`.
- Candidate 2 code SHA-256: `8e4a14339423b87578ff8a39e0d7b97f4362f472e6c4ac4440a8117e0d718071`.
- Independent static review passed for the original frozen-AST baseline. Every submitted output separately passed the machine output-contract validator.

## Resources and limits

- Kaggle GPU runs: 3 on Nvidia T4.
- Kaggle submissions: 3.
- Direct monetary cost: USD `0.00` (Kaggle-provided practice compute).
- Agent input/output tokens: not exposed by the current Codex runtime, recorded as unavailable rather than estimated.
- No credentials or authentication material were written into the run artifacts.

## Decision

Retain submission `55213957`, which has the highest confirmed public score of the three completed candidates. Do not infer a private score or final rank from the public leaderboard snapshot.
