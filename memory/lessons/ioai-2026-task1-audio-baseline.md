# IOAI 2026 Task1 (AST audio, 16 old + 13 new classes, Score=0.5*Acc_old+0.5*Acc_new)

## Key gotchas
- **Kaggle data root**: don't hardcode; use `glob.glob('/kaggle/input/**/train.csv', recursive=True)`.
  The competition_sources path may differ from the slug name.
- **Runtime killer**: some fine_tune insect clips are multi-minute. Always load
  with `librosa.load(path, sr=16000, mono=True, duration=10.24)` (10.24s = AST
  max_length 1024). Without truncation the mel-spectrogram of full audio makes
  a 1283-clip embed run take 50+ min; with truncation still ~35 min (embedding
  is the bottleneck — cache embeddings!).
- Floor that works: frozen `model.audio_spectrogram_transformer(feats).pooler_output`
  embeddings + StandardScaler + balanced LogisticRegression, 29-way. CV~0.90.
- **CV overstates LB badly**: holdout/CV ~0.91 → public LB 0.781. Gap ~0.13 due
  to few-shot old classes + small test. Trust LB direction.
- Kernel commit runs are slow (~35 min for this embedding pipeline). Budget wait
  time accordingly; poll with sleeps. kaggle_kernel_log returns the LAST COMPLETE
  version's log, not the currently running one (stale/misleading).
