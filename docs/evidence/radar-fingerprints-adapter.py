"""Extract the radar kNN fingerprint for train and test, for a ruler audit.

The fingerprint is the one the frozen kNN parents actually search in:
5x5 average pooling of the 6-channel 50x181 signal, width edge-padded to 185,
giving 6*10*37 = 2220 dimensions, no standardisation.
Labels are never read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DATA = Path(
    "/Users/bytedance/Documents/IOAI/ioai-historical/fresh_bases/"
    "20260802-radar-t2-fable5-v2/data"
)
OUT = Path(__file__).resolve().parent
HEIGHT, WIDTH = 50, 181


def fingerprint(signal: np.ndarray) -> np.ndarray:
    if signal.shape != (6, HEIGHT, WIDTH):
        raise ValueError(f"unexpected signal shape {signal.shape}")
    signal = np.asarray(signal, dtype=np.float32)
    padded = np.pad(signal, ((0, 0), (0, 0), (0, 4)), mode="edge")
    pooled = padded.reshape(6, 10, 5, 37, 5).mean(axis=(2, 4))
    return pooled.reshape(-1).astype(np.float32, copy=False)


def collect(directory: Path, channels: int) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(directory.glob("*.mat.pt"), key=lambda p: int(p.name.split(".", 1)[0]))
    features = np.empty((len(paths), 6 * 10 * 37), dtype=np.float32)
    ids = np.empty(len(paths), dtype=np.int64)
    for index, path in enumerate(paths):
        value = torch.load(path, map_location="cpu", weights_only=True).numpy()
        if value.shape[0] != channels:
            raise ValueError(f"{path}: expected {channels} channels, got {value.shape[0]}")
        features[index] = fingerprint(value[:6])
        ids[index] = int(path.name.split(".", 1)[0])
    return features, ids


def main() -> None:
    train, train_ids = collect(DATA / "training_set/training_set", 7)
    test, test_ids = collect(DATA / "test_set/test_set", 6)
    np.savez_compressed(OUT / "radar_train_fingerprints.npz", x=train, ids=train_ids)
    np.savez_compressed(OUT / "radar_test_fingerprints.npz", x=test, ids=test_ids)
    print("train", train.shape, "test", test.shape)


if __name__ == "__main__":
    main()
