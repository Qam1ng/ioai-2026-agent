"""Extract & cache frozen AST embeddings for Task 1.

The AST encoder forward is the expensive part; we run it ONCE over every clip
(train + fine_tune + eval) and cache 768-d pooled embeddings to disk. All head
training / calibration then happens on the cached embeddings, so the
experiment loop iterates in seconds instead of minutes.

Also saves the frozen 16-class classifier head (W_old, b_old) so the solver can
reproduce the checkpoint's exact old-class logits without re-forwarding.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IN = HERE / "input"
CACHE = HERE / "cache"
MODEL_DIR = IN / "model"
SR = 16000
BATCH = 16


def _read_rows(csv_path: Path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def all_paths() -> list[str]:
    """Every audio path referenced by train, fine_tune, and submission."""
    paths: list[str] = []
    seen = set()
    for name in ("train.csv", "fine_tune.csv", "submission.csv"):
        for r in _read_rows(IN / name):
            p = r["path"]
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def extract() -> None:
    import torch
    import librosa
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    CACHE.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    fe = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
    model = ASTForAudioClassification.from_pretrained(MODEL_DIR).to(device).eval()

    # The head is ASTMLPHead = LayerNorm -> Linear(768->16). We cache the
    # LAYERNORMED feature (the dense layer's input) so old logits = feat @ W_old
    # reproduce the checkpoint exactly, and new rows share the same normalized
    # feature space. Save the frozen dense weights (W_old [16,768], b_old [16]).
    head = model.classifier            # ASTMLPHead
    np.save(CACHE / "old_head_W.npy", head.dense.weight.detach().cpu().numpy())
    np.save(CACHE / "old_head_b.npy", head.dense.bias.detach().cpu().numpy())
    print(f"old head dense: {tuple(head.dense.weight.shape)}")

    paths = all_paths()
    print(f"extracting embeddings for {len(paths)} clips...")
    embs = np.zeros((len(paths), model.config.hidden_size), dtype=np.float32)

    t0 = time.time()
    for i in range(0, len(paths), BATCH):
        batch_paths = paths[i : i + BATCH]
        waves = []
        for p in batch_paths:
            y, _ = librosa.load(IN / p, sr=SR, mono=True)
            waves.append(y)
        inputs = fe(waves, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out = model.audio_spectrogram_transformer(
                inputs["input_values"].to(device)
            )
            # LayerNormed feature = the dense head's input (see head note above)
            feat = model.classifier.layernorm(out.pooler_output)  # [B, 768]
        embs[i : i + len(batch_paths)] = feat.cpu().numpy()
        if (i // BATCH) % 10 == 0:
            done = i + len(batch_paths)
            print(f"  {done}/{len(paths)}  ({done/(time.time()-t0):.1f} clips/s)")

    np.save(CACHE / "embeddings.npy", embs)
    with open(CACHE / "paths.txt", "w") as f:
        f.write("\n".join(paths))
    print(f"saved {embs.shape} embeddings in {time.time()-t0:.1f}s -> {CACHE}")


if __name__ == "__main__":
    extract()
