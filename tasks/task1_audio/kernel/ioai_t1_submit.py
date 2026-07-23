"""IOAI 2026 Task-1 submission notebook (Kaggle Code Competition).

Runs end-to-end on Kaggle: loads the provided AST checkpoint, extracts frozen
768-d LayerNormed features, then reproduces the agent's best offline strategy —
augment features with the frozen 16-class head logits and fit a class-balanced
LogisticRegression over all 29 classes — and writes /kaggle/working/submission.csv.

Starts from the provided checkpoint, reuses its encoder, adds new-class capacity
via the linear head; no training from scratch, no other pretrained audio models.
"""
import os
import csv
import numpy as np
import torch
import librosa
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

INPUT = "/kaggle/input/ioai-2026-ai-models-track-practice-task-1"
OUT = "/kaggle/working/submission.csv"
SR, BATCH = 16000, 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rows(name):
    with open(os.path.join(INPUT, name)) as f:
        return list(csv.DictReader(f))


def main():
    fe = ASTFeatureExtractor.from_pretrained(os.path.join(INPUT, "model"))
    model = ASTForAudioClassification.from_pretrained(
        os.path.join(INPUT, "model")).to(DEV).eval()
    old_W = model.classifier.dense.weight.detach().cpu().numpy().astype(np.float64)
    old_b = model.classifier.dense.bias.detach().cpu().numpy().astype(np.float64)

    train, ft, sub = rows("train.csv"), rows("fine_tune.csv"), rows("submission.csv")
    labeled = train + ft
    y = np.array([int(r["target"]) for r in labeled], dtype=np.int64)

    def feats(records):
        paths = [r["path"] for r in records]
        out = np.zeros((len(paths), model.config.hidden_size), dtype=np.float32)
        for i in range(0, len(paths), BATCH):
            waves = [librosa.load(os.path.join(INPUT, p), sr=SR, mono=True)[0]
                     for p in paths[i:i + BATCH]]
            inp = fe(waves, sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                pooled = model.audio_spectrogram_transformer(
                    inp["input_values"].to(DEV)).pooler_output
                feat = model.classifier.layernorm(pooled)
            out[i:i + len(waves)] = feat.cpu().numpy()
        return out.astype(np.float64)

    Xl = feats(labeled)
    Xe = feats(sub)

    def augment(X):
        return np.hstack([X, X @ old_W.T + old_b])   # features + frozen old logits

    scaler = StandardScaler().fit(augment(Xl))
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                             solver="lbfgs", random_state=0)
    clf.fit(scaler.transform(augment(Xl)), y)
    preds = clf.predict(scaler.transform(augment(Xe))).astype(int)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "target"])
        for r, p in zip(sub, preds):
            w.writerow([r["path"], int(p)])
    print("wrote", OUT, "rows:", len(sub), "distinct preds:", len(set(preds.tolist())))


if __name__ == "__main__":
    main()
