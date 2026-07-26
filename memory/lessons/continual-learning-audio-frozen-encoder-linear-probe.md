# Class-incremental fine-tuning (e.g. AST audio, "add N new classes")

When a task says "extend a pretrained classifier to new classes WITHOUT
forgetting old ones" and metric = 0.5*Acc_old + 0.5*Acc_new:

- **Frozen encoder + linear probe (StandardScaler + LogisticRegression,
  class_weight='balanced') over the FULL class space** is a strong, robust
  floor. Freezing the backbone preserves Acc_old *by construction* — this is
  the dominant risk in these tasks (catastrophic forgetting).
- Cache embeddings to .npy once, then CV iteration is instant. Extraction is
  the runtime bottleneck, not the classifier.
- Trainable heads / full fine-tune can OVERFIT few-shot old classes: observed
  local holdout 0.9156 → public LB 0.78095 (~0.13 gap). Small test sets +
  few-shot classes make CV very optimistic — trust LB direction.
- Runtime: resample to the model's expected SR and TRUNCATE audio (e.g. 10.24s)
  or the offline kernel will time out (35min→). Batch inference on GPU.
- Kernel gotcha: find data root with recursive glob `/kaggle/input/**/train.csv`
  not a hardcoded path.
- Get a valid submission early; only replace it if a candidate beats it by a
  margin bigger than the known CV→LB gap.
