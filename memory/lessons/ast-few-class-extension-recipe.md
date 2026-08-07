When a task ships a fine-tuned HF audio checkpoint (e.g. ASTForAudioClassification on N old classes) plus a small fine_tune.csv adding M new classes:

1. Cache features once. `ASTFeatureExtractor` in transformers 5.x has a pure-numpy path, so it works without torchaudio. Resample to 16 kHz with `scipy.signal.resample_poly`; mono = channel mean. Keep the extractor's default ZERO padding for clips shorter than the model window — cyclically tiling the clip to fill the window is measurably WORSE (linear probe 0.904 vs 0.913). Cut longer clips into ~3 evenly spaced windows; random window at train time, softmax-average at inference.

2. A frozen-embedding logistic regression on the checkpoint's pooler output (L2-normalised, C=10) is a strong, ~2-minute baseline and lands within ~1.5 points of a full fine-tune. Build it first; it is the safety candidate.

3. For the fine-tune, widen the head to N+M and COPY the original N rows of weight/bias into the new head so the old classes are not forgotten. Backbone lr 2e-5, head lr 10x, OneCycle, label smoothing 0.1, SpecAugment + mixup, ~6 epochs. More epochs hurt on ~900 samples (14 epochs scored below 6).

4. The frozen heads and the fine-tune make different errors: an EQUAL-weight softmax average of the fine-tune plus 2-3 frozen-head variants (mean-pool, mean+max-pool, per-window-expanded) beat every single model and every weighting I fitted. Prefer the untuned equal weight — fitted blend weights on ~900 OOF rows differ by a handful of samples and are noise.

Do not trust the shipped checkpoint's own head as a signal of feature quality: it scored only 0.65 on its own class subset here while its embeddings supported 0.91.
