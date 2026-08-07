# IOAI 2026 Task1 - AST 29-class extension (audio, code comp, offline kernel)

Task: extend pretrained ASTForAudioClassification (16 old classes 0-15) to 29 classes
(add 13 new 16-28) in a single forward pass. Metric = 0.5*Acc_old + 0.5*Acc_new
(GROUP-balanced accuracy over true-old vs true-new clips, NOT plain accuracy).

## What worked (strong floor, cheap)
Frozen AST encoder as feature extractor: `model.audio_spectrogram_transformer(feats).pooler_output`
gives [N,768]. Train StandardScaler + LogisticRegression(class_weight='balanced', max_iter=2000)
on 29-way (train.csv old + fine_tune.csv new). 5-fold stratified CV:
**Score 0.903 (Acc_old 0.895, Acc_new 0.910)**. Backbone is very discriminative; a linear
probe recovers old classes well even though the pretrained head is discarded.

## Key facts
- Audio sample rates wildly heterogeneous (8k..384k) -> ALWAYS librosa.load(sr=16000, mono=True).
- AST feature extractor: 128 mel, max_length 1024 (~10s), mean=-4.2677 std=4.5690. Insect
  classes (25-28) are long (up to 120s) -> truncated to first 10s, fine.
- Tiny old classes (Sheep=3, KbdTyping=3, Rain=7) -> StratifiedKFold(5) warns, noisy Acc_old.
- Embedding all 920 train files = ~116s on GPU (kernel-feasible). Cache to .npy.
- Env local: transformers 5.14.1, torch 2.13. Kernel offline; model checkpoint IS in comp data.
  Load with ASTForAudioClassification.from_pretrained(root/model). No torchaudio -> librosa.
- Kaggle user: qam1ng. Code competition -> push kernel dir, output /kaggle/working/submission.csv.

## Next ideas to beat floor
- Head+low-LR backbone finetune with rehearsal of old data to avoid forgetting.
- Fold LR into model head to satisfy 'single forward pass' rule cleanly.
- Logit-bias calibration to trade Acc_old<->Acc_new for the 0.5/0.5 objective.
