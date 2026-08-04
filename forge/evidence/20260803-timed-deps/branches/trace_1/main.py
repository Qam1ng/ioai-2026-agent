"""Compliant head-expansion baseline for IOAI 2026 Practice Task 1.

The supplied AST checkpoint and its original 16-class head are immutable.  We
cache one encoder embedding for each labelled example, train only a new 13-row
linear head, and concatenate the frozen and new logits.  At test time the
wrapper makes exactly one encoder call and returns all 29 logits together.
"""

from __future__ import annotations

import csv
import gc
import math
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path


COMPETITION = "ioai-2026-ai-models-track-practice-task-1-timed-deps"
NUM_OLD_CLASSES = 16
NUM_NEW_CLASSES = 13
NUM_CLASSES = NUM_OLD_CLASSES + NUM_NEW_CLASSES
SEED = 20260803
SAMPLE_RATE = 16_000
BATCH_SIZE = 16
HEAD_EPOCHS = 300


def ensure_transformers() -> None:
    """Use the attached wheel dataset only if the base image lacks packages."""
    try:
        import safetensors  # noqa: F401
        import transformers  # noqa: F401
        return
    except ImportError:
        pass

    wheel_root = Path("/kaggle/input/ioai-2026-wheel-dataset")
    wheel_files = list(wheel_root.rglob("*.whl")) if wheel_root.exists() else []
    if not wheel_files:
        raise RuntimeError("transformers is unavailable and no offline wheels were found")

    wheel_dirs = sorted({str(path.parent) for path in wheel_files})
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-cache-dir",
    ]
    for directory in wheel_dirs:
        command.extend(["--find-links", directory])
    command.extend(["transformers", "safetensors"])
    subprocess.check_call(command)


ensure_transformers()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import ASTForAudioClassification, AutoFeatureExtractor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_competition_root() -> Path:
    input_root = Path("/kaggle/input")
    preferred = input_root / COMPETITION
    required = ("train.csv", "fine_tune.csv", "submission.csv")
    if all((preferred / name).is_file() for name in required):
        return preferred

    candidates = []
    for submission_path in input_root.glob("*/submission.csv"):
        candidate = submission_path.parent
        if all((candidate / name).is_file() for name in required):
            if (candidate / "model" / "config.json").is_file():
                candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one competition input directory, found {candidates}")
    return candidates[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_audio(path: Path) -> np.ndarray:
    sample_rate, waveform = wavfile.read(path)
    if waveform.ndim == 2:
        waveform = waveform.astype(np.float32).mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError(f"Unsupported waveform shape {waveform.shape}: {path}")

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        scale = float(max(abs(info.min), info.max))
        waveform = waveform.astype(np.float32) / scale
    else:
        waveform = waveform.astype(np.float32)

    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    return np.ascontiguousarray(waveform)


class ExpandedAST(nn.Module):
    """Frozen supplied AST plus exactly 13 trainable classifier rows."""

    def __init__(self, supplied_model: ASTForAudioClassification) -> None:
        super().__init__()
        self.encoder = supplied_model.audio_spectrogram_transformer
        self.old_layernorm = supplied_model.classifier.layernorm
        self.old_dense = supplied_model.classifier.dense
        self.new_dense = nn.Linear(self.old_dense.in_features, NUM_NEW_CLASSES)

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.old_layernorm.parameters():
            parameter.requires_grad_(False)
        for parameter in self.old_dense.parameters():
            parameter.requires_grad_(False)

        with torch.no_grad():
            old_weight = self.old_dense.weight.float()
            nn.init.normal_(
                self.new_dense.weight,
                mean=float(old_weight.mean()),
                std=float(old_weight.std()),
            )
            self.new_dense.bias.fill_(float(self.old_dense.bias.float().mean()))

    def encode_once(self, input_values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_values=input_values, return_dict=True)
        pooled = (encoded.last_hidden_state[:, 0] + encoded.last_hidden_state[:, 1]) / 2
        return self.old_layernorm(pooled)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # The only encoder invocation for this batch.  Both heads consume the
        # same embedding and the single return tensor contains all 29 scores.
        embedding = self.encode_once(input_values)
        old_logits = self.old_dense(embedding)
        new_logits = self.new_dense(embedding)
        logits = torch.cat((old_logits, new_logits), dim=-1)
        if logits.shape[-1] != NUM_CLASSES:
            raise RuntimeError(f"Expected {NUM_CLASSES} logits, got {logits.shape}")
        return logits


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def preprocess_batch(
    rows: list[dict[str, str]],
    root: Path,
    extractor: AutoFeatureExtractor,
) -> torch.Tensor:
    waveforms = [load_audio(root / row["path"]) for row in rows]
    features = extractor(
        waveforms,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    return features["input_values"]


def cache_labelled_embeddings(
    rows: list[dict[str, str]],
    root: Path,
    extractor: AutoFeatureExtractor,
    model: ExpandedAST,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    old_scores: list[torch.Tensor] = []
    labels: list[int] = []
    model.eval()

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]
        input_values = preprocess_batch(batch_rows, root, extractor).to(device)
        with torch.inference_mode(), autocast_context(device):
            embedding = model.encode_once(input_values)
            old_logits = model.old_dense(embedding)
        embeddings.append(embedding.float().cpu())
        old_scores.append(old_logits.float().cpu())
        labels.extend(int(row["target"]) for row in batch_rows)
        if start % (BATCH_SIZE * 10) == 0:
            print(f"cached labelled embeddings: {min(start + BATCH_SIZE, len(rows))}/{len(rows)}")

    return (
        torch.cat(embeddings, dim=0),
        torch.cat(old_scores, dim=0),
        torch.tensor(labels, dtype=torch.long),
    )


def train_new_rows(
    model: ExpandedAST,
    embeddings: torch.Tensor,
    old_logits: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> None:
    embeddings = embeddings.to(device)
    old_logits = old_logits.to(device)
    labels = labels.to(device)

    counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    if torch.any(counts == 0):
        raise RuntimeError(f"Every class must be represented; counts={counts.tolist()}")
    class_weights = torch.sqrt(counts.mean() / counts).to(device)

    optimizer = torch.optim.AdamW(
        model.new_dense.parameters(),
        lr=0.02,
        weight_decay=0.01,
    )
    model.new_dense.train()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(HEAD_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        new_logits = model.new_dense(embeddings)
        logits = torch.cat((old_logits, new_logits), dim=1)
        loss = F.cross_entropy(
            logits,
            labels,
            weight=class_weights,
            label_smoothing=0.01,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.new_dense.parameters(), max_norm=5.0)
        optimizer.step()

        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.new_dense.state_dict().items()
            }
        if epoch % 50 == 0 or epoch == HEAD_EPOCHS - 1:
            with torch.no_grad():
                accuracy = float((logits.argmax(1) == labels).float().mean().cpu())
            print(f"head epoch={epoch:03d} loss={value:.5f} train_accuracy={accuracy:.4f}")

    if best_state is None:
        raise RuntimeError("Head optimization did not produce a finite state")
    model.new_dense.load_state_dict(best_state)
    model.new_dense.eval()


def predict(
    rows: list[dict[str, str]],
    root: Path,
    extractor: AutoFeatureExtractor,
    model: ExpandedAST,
    device: torch.device,
) -> list[int]:
    predictions: list[int] = []
    model.eval()

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]
        input_values = preprocess_batch(batch_rows, root, extractor).to(device)
        with torch.inference_mode(), autocast_context(device):
            logits = model(input_values)  # one wrapper/encoder forward, 29 scores
        batch_predictions = logits.float().argmax(dim=1).cpu().tolist()
        predictions.extend(int(value) for value in batch_predictions)
        if start % (BATCH_SIZE * 10) == 0:
            print(f"predicted test rows: {min(start + BATCH_SIZE, len(rows))}/{len(rows)}")
    return predictions


def write_and_verify_submission(
    sample_rows: list[dict[str, str]], predictions: list[int]
) -> Path:
    if len(predictions) != len(sample_rows):
        raise RuntimeError("Prediction count does not match sample submission")
    if len({row["path"] for row in sample_rows}) != len(sample_rows):
        raise RuntimeError("Sample submission contains duplicate paths")
    if any(not isinstance(value, int) or not 0 <= value < NUM_CLASSES for value in predictions):
        raise RuntimeError("Prediction labels must be integers in [0, 28]")

    output_path = Path("/kaggle/working/submission.csv")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "target"])
        writer.writeheader()
        for row, prediction in zip(sample_rows, predictions):
            writer.writerow({"path": row["path"], "target": prediction})

    written = read_csv(output_path)
    if [row["path"] for row in written] != [row["path"] for row in sample_rows]:
        raise RuntimeError("Output path order differs from sample submission")
    if [int(row["target"]) for row in written] != predictions:
        raise RuntimeError("Output targets differ from predictions")
    print(f"wrote verified submission: {output_path} rows={len(written)}")
    return output_path


def main() -> None:
    seed_everything(SEED)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")

    root = locate_competition_root()
    train_rows = read_csv(root / "train.csv")
    fine_tune_rows = read_csv(root / "fine_tune.csv")
    sample_rows = read_csv(root / "submission.csv")
    labelled_rows = train_rows + fine_tune_rows

    observed_labels = sorted({int(row["target"]) for row in labelled_rows})
    if observed_labels != list(range(NUM_CLASSES)):
        raise RuntimeError(f"Expected labels 0..28, found {observed_labels}")
    if {row["path"] for row in train_rows} & {row["path"] for row in fine_tune_rows}:
        raise RuntimeError("train.csv and fine_tune.csv paths overlap")
    labelled_paths = {row["path"] for row in labelled_rows}
    if labelled_paths & {row["path"] for row in sample_rows}:
        raise RuntimeError("Labelled and test paths overlap")

    model_dir = root / "model"
    extractor = AutoFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    supplied = ASTForAudioClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    if supplied.classifier.dense.out_features != NUM_OLD_CLASSES:
        raise RuntimeError("Supplied checkpoint is not the expected 16-class AST")
    model = ExpandedAST(supplied).to(device)
    del supplied
    gc.collect()

    embeddings, old_logits, labels = cache_labelled_embeddings(
        labelled_rows, root, extractor, model, device
    )
    train_new_rows(model, embeddings, old_logits, labels, device)
    del embeddings, old_logits, labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    predictions = predict(sample_rows, root, extractor, model, device)
    write_and_verify_submission(sample_rows, predictions)


if __name__ == "__main__":
    main()
