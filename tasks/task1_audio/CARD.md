# Task 1 — Audio Classifier (At Home Practice) · task card

URL: https://www.kaggle.com/competitions/ioai-2026-ai-models-track-practice-task-1
Type: **class-incremental / continual learning**, single-label audio classification.
Format: Kaggle "Community Code Competition", PRIVATE. Submit a `submission.csv`.

## The problem
Start from a **provided AST checkpoint** (`model/`) trained on **16 base classes**
(labels 0–15). Extend it to **29 classes** (add 13 new: labels 16–28) in a
**single forward pass**, WITHOUT catastrophic forgetting of the 16 old classes.

Rules:
- ✅ Must reuse the checkpoint's encoder weights. May add params (extra
  classifier rows, adapters) and fine-tune any subset.
- ❌ No training from scratch; no other pre-trained audio models.
- ⏱️ ~10 min single-GPU training budget.

## Metric (custom)
`Score = 0.5 * Acc_old + 0.5 * Acc_new`
- Acc_old = accuracy over test clips whose TRUE label ∈ {0..15}
- Acc_new = accuracy over test clips whose TRUE label ∈ {16..28}
Forgetting old is exactly as costly as failing to learn new. → strong pressure
against catastrophic forgetting; naive fine-tuning on only new data will tank Acc_old.

## Data (2.03 GB total)
- `audio/`      — 1,283 `.wav` clips
- `model/`      — the AST checkpoint (encoder + 16-class head)
- `train.csv`   — 296 rows, labels 0–15. cols: path, split(train), target, category
- `fine_tune.csv` — 624 rows, labels 16–28. same schema. (combine both to train)
- `submission.csv` — 363 rows, cols: path, target. Replace target with predictions.
  **Must keep same columns and row order as submission.csv.**

## Class labels (0–28)
0 Dog · 1 Rooster · 2 Pig · 3 Cow · 4 Frog · 5 Cat · 6 Hen · 7 Sheep · 8 Crow ·
9 Mouse Click · 10 Keyboard Typing · 11 Thunderstorm · 12 Sea Waves ·
13 Bird Chirping · 14 Wolf Howl · 15 Rain  ‹— OLD (base)
16 Crackling Fire · 17 Axe · 18 Chainsaw · 19 Generator · 20 Hand Saw ·
21 Vehicle Engine · 22 Helicopter · 23 Gunshot · 24 Firework ·
25 Pterophylla Camellifolia · 26 Cicada Orni · 27 Gryllus Campestris ·
28 Tettigonia Viridissima  ‹— NEW (fine-tune)

OLD = 0..15 (16 classes), NEW = 16..28 (13 classes).

## Likely-strong approach (for the solver to explore)
- AST is a HuggingFace `ASTModel` (spectrogram transformer). Extract encoder
  features, keep the frozen 16-way head rows, add 13 new head rows.
- Freeze encoder (or light adapters) → train only the new head / a small
  classifier on pooled features → old classes preserved, new classes learned
  within the 10-min budget. Prototype/linear-probe on frozen features is a
  natural first baseline. Class imbalance across the 13 new classes exists.

## What's needed to actually run it here
1. Kaggle API token (to download the 2 GB data + checkpoint) — NOT yet provided.
2. torch + torchaudio + transformers + a GPU — env currently has none.
3. Submission mechanism (Kaggle CLI / the Discord #resources Agent Skill file).
