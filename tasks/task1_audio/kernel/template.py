"""Kaggle submission notebook TEMPLATE for Task 1.

run.py injects the agent's best `fit_predict` at the marker below, then this
notebook (on Kaggle GPU) extracts frozen AST features and calls it to produce
/kaggle/working/submission.csv. Generic: whatever strategy the agent picked,
this template runs it — no hand-editing per iteration.
"""
import os, csv, numpy as np, torch, librosa
from transformers import ASTFeatureExtractor, ASTForAudioClassification

OUT = "/kaggle/working/submission.csv"
SR, BATCH = 16000, 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def find_input():
    for root, _d, files in os.walk("/kaggle/input"):
        if "submission.csv" in files and "train.csv" in files:
            return root
    raise RuntimeError("input dir not found under /kaggle/input")


def find_model(inp):
    for root, _d, files in os.walk(inp):
        if "config.json" in files and any(f.endswith(".safetensors") for f in files):
            return root
    raise RuntimeError("model dir not found under " + inp)


INPUT = find_input(); MODEL = find_model(INPUT)
print("INPUT =", INPUT, "| MODEL =", MODEL, "| device =", DEV)


# ============================= AGENT_FIT_PREDICT =============================
# (run.py replaces this line with the agent's best fit_predict definition)
def fit_predict(Xtr, ytr, Xva, old_W, old_b):
    raise RuntimeError("no agent code injected")
# ===========================================================================


def rows(name):
    with open(os.path.join(INPUT, name)) as f:
        return list(csv.DictReader(f))


def main():
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).to(DEV).eval()
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

    Xl, Xe = feats(labeled), feats(sub)
    preds = np.asarray(fit_predict(Xl, y, Xe, old_W, old_b)).astype(int).ravel()

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "target"])
        for r, p in zip(sub, preds):
            w.writerow([r["path"], int(p)])
    print("wrote", OUT, "rows:", len(sub), "distinct preds:", len(set(preds.tolist())))


if __name__ == "__main__":
    main()
