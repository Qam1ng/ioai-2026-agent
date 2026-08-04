"""IOAI Task 1 trace_2: frozen-AST geometric incremental classifier.

Compliance properties:
- only the supplied checkpoint and labeled competition CSVs are used;
- exactly one deterministic waveform view and one AST encoder call per sample;
- the original 16 classifier rows and the AST encoder remain immutable;
- predictions are sample-local (no test-order/global assignment or path overrides).
"""

# Keep offline environment setup before third-party imports.
import os
import re
import subprocess
import sys
from pathlib import Path

WHEEL_DATASET = "ioai-2026-wheel-dataset"
_LOCAL_VER = re.compile(
    r"^(?P<name>[^-]+)-(?P<ver>\d[\d.]*)(?:\+?cu\d+)-(?P<rest>.+\.whl)$"
)


def _find_wheels(search_root="/kaggle/input", depth=7):
    base = Path(search_root)
    for d in range(depth):
        hits = sorted(base.glob("/".join(["*"] * d + ["ioai_env-*.whl"])))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        f"IOAI wheels not found under {base}; attach {WHEEL_DATASET!r}"
    )


def _normalise_wheels(wheels):
    needs_fix = [w for w in wheels.glob("*.whl") if _LOCAL_VER.match(w.name)]
    if not needs_fix:
        return wheels
    work = Path("/kaggle/working/_ioai_wheels")
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob("*"):
        stale.unlink()
    for wheel in sorted(wheels.glob("*.whl")):
        match = _LOCAL_VER.match(wheel.name)
        name = (
            f"{match['name']}-{match['ver']}-{match['rest']}"
            if match
            else wheel.name
        )
        (work / name).symlink_to(wheel)
    return work


def setup_ioai_env(search_root="/kaggle/input"):
    wheels = _normalise_wheels(_find_wheels(search_root))
    common = ["--no-index", f"--find-links={wheels}"]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *common, "uv"]
    )
    env = dict(os.environ, UV_SKIP_WHEEL_FILENAME_CHECK="1")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--system",
                *common,
                "ioai-env",
            ],
            env=env,
        )
    except subprocess.CalledProcessError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--only-binary=:all:",
                *common,
                "ioai-env",
            ]
        )
    print(f"IOAI environment ready from {wheels}", flush=True)


setup_ioai_env()

import csv
import gc
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification

SEED = 20260803
N_OLD = 16
N_NEW = 13
N_CLASSES = N_OLD + N_NEW
TARGET_SR = 16_000
MAX_SECONDS = 10.20
BATCH_SIZE = 16
HEAD_STEPS = 260
ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def find_competition_csv(name):
    candidates = [
        p for p in ROOT.rglob(name)
        if p.is_file() and "competitions" in p.parts
    ]
    if not candidates:
        candidates = [p for p in ROOT.rglob(name) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(name)
    # The task archive directory is the one that also contains audio/.
    candidates.sort(key=lambda p: (not (p.parent / "audio").is_dir(), str(p)))
    return candidates[0]


def find_model_dir():
    for config in sorted(ROOT.rglob("config.json")):
        if not (config.parent / "model.safetensors").is_file():
            continue
        try:
            payload = json.loads(config.read_text())
        except Exception:
            continue
        if payload.get("model_type") == "audio-spectrogram-transformer":
            return config.parent
    raise FileNotFoundError("supplied AST checkpoint")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_wave(path):
    """One deterministic center view, reading at most 10.2 seconds from disk."""
    with sf.SoundFile(str(path)) as handle:
        sr = int(handle.samplerate)
        frames = min(len(handle), max(1, int(round(sr * MAX_SECONDS))))
        handle.seek(max(0, (len(handle) - frames) // 2))
        wave = handle.read(frames, dtype="float32", always_2d=True).mean(axis=1)
    if sr != TARGET_SR:
        divisor = math.gcd(sr, TARGET_SR)
        wave = resample_poly(
            wave, TARGET_SR // divisor, sr // divisor
        ).astype(np.float32, copy=False)
    if not np.isfinite(wave).all():
        wave = np.nan_to_num(wave, copy=False)
    return wave


def batches(rows, archive_dir, extractor, batch_size=BATCH_SIZE):
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            paths = [archive_dir / row["path"] for row in chunk]
            waves = list(pool.map(read_wave, paths))
            values = extractor(
                waves, sampling_rate=TARGET_SR, return_tensors="pt"
            ).input_values
            yield chunk, values


class IncrementalAST(nn.Module):
    """One encoder call; the public forward always returns [B, 29]."""

    def __init__(self, checkpoint):
        super().__init__()
        base = ASTForAudioClassification.from_pretrained(
            checkpoint, local_files_only=True
        )
        if not hasattr(base.classifier, "dense") or not hasattr(base.classifier, "layernorm"):
            raise TypeError("expected the supplied AST LayerNorm+dense classifier head")
        if base.classifier.dense.out_features != N_OLD:
            raise ValueError(
                f"expected {N_OLD} old rows, got {base.classifier.dense.out_features}"
            )
        self.encoder = base.audio_spectrogram_transformer
        self.old_classifier = base.classifier
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.old_classifier.parameters():
            parameter.requires_grad_(False)
        hidden = self.old_classifier.dense.in_features
        self.new_weight = nn.Parameter(torch.zeros(N_NEW, hidden), requires_grad=False)
        self.new_bias = nn.Parameter(torch.zeros(N_NEW), requires_grad=False)
        self.register_buffer("new_scale", torch.tensor(1.0))
        self.register_buffer("new_shift", torch.tensor(0.0))

    def _one_embedding(self, input_values):
        # Exactly one AST encoder invocation.
        return self.encoder(input_values=input_values).pooler_output

    def encode_training_rows(self, input_values):
        embedding = self._one_embedding(input_values)
        # New rows share the supplied head's frozen LayerNorm geometry.
        head_features = self.old_classifier.layernorm(embedding)
        old_logits = self.old_classifier.dense(head_features)
        return head_features.float(), old_logits.float()

    def set_new_head(self, weight, bias, scale, shift):
        self.new_weight.copy_(weight.to(self.new_weight))
        self.new_bias.copy_(bias.to(self.new_bias))
        self.new_scale.fill_(float(scale))
        self.new_shift.fill_(float(shift))

    def forward(self, input_values):
        embedding = self._one_embedding(input_values)
        head_features = self.old_classifier.layernorm(embedding)
        old_logits = self.old_classifier.dense(head_features).float()
        new_logits = F.linear(
            head_features.float(), self.new_weight.float(), self.new_bias.float()
        )
        new_logits = self.new_scale * new_logits + self.new_shift
        logits = torch.cat([old_logits, new_logits], dim=1)
        if logits.shape[1] != N_CLASSES:
            raise RuntimeError(f"bad output shape {tuple(logits.shape)}")
        return logits


@torch.inference_mode()
def encode_labeled(model, rows, archive_dir, extractor, device):
    all_embeddings, all_old_logits, all_targets = [], [], []
    for batch_rows, input_values in batches(rows, archive_dir, extractor):
        values = input_values.to(device, non_blocking=True)
        if device.type == "cuda":
            values = values.half()
        embeddings, old_logits = model.encode_training_rows(values)
        all_embeddings.append(embeddings.cpu())
        all_old_logits.append(old_logits.cpu())
        all_targets.extend(int(row["target"]) for row in batch_rows)
    return (
        torch.cat(all_embeddings).float(),
        torch.cat(all_old_logits).float(),
        torch.tensor(all_targets, dtype=torch.long),
    )


def stratified_folds(targets, folds=3):
    assignment = np.empty(len(targets), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    y = np.asarray(targets)
    for label in range(N_CLASSES):
        indices = np.flatnonzero(y == label)
        if len(indices) < folds:
            raise ValueError(f"class {label} has only {len(indices)} rows")
        rng.shuffle(indices)
        assignment[indices] = np.arange(len(indices)) % folds
    return assignment


def objective_weights(targets):
    """70% group-micro + 30% class-macro, each 50:50 old/new."""
    y = targets.detach().cpu().numpy()
    weights = np.zeros(len(y), dtype=np.float32)
    for group_labels in (range(0, N_OLD), range(N_OLD, N_CLASSES)):
        mask = np.isin(y, list(group_labels))
        group_count = int(mask.sum())
        n_labels = len(list(group_labels))
        for label in group_labels:
            label_mask = y == label
            count = int(label_mask.sum())
            weights[label_mask] = (
                0.70 * 0.5 / group_count + 0.30 * 0.5 / (n_labels * count)
            )
    weights /= weights.mean()
    return torch.from_numpy(weights)


def geometric_initialization(embeddings, targets, old_weight, old_bias):
    center = embeddings.mean(dim=0)
    old_norm = old_weight.float().norm(dim=1).median().clamp_min(1e-4)
    median_bias = old_bias.float().median()
    rows, biases = [], []
    for label in range(N_OLD, N_CLASSES):
        direction = embeddings[targets == label].mean(dim=0) - center
        direction = direction / direction.norm().clamp_min(1e-6)
        row = direction * old_norm
        rows.append(row)
        biases.append(median_bias - torch.dot(row, center))
    return torch.stack(rows), torch.stack(biases)


def fit_new_rows(embeddings, old_logits, targets, old_weight, old_bias, steps):
    device = old_weight.device
    z = embeddings.to(device)
    fixed_old = old_logits.to(device)
    y = targets.to(device)
    weights = objective_weights(targets).to(device)
    initial_w, initial_b = geometric_initialization(
        z, y, old_weight.float(), old_bias.float()
    )
    weight = nn.Parameter(initial_w.clone())
    bias = nn.Parameter(initial_b.clone())
    optimizer = torch.optim.AdamW([weight, bias], lr=0.018, weight_decay=2e-4)
    target_norm = old_weight.float().norm(dim=1).median().detach()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        new_logits = F.linear(z, weight, bias)
        logits = torch.cat([fixed_old, new_logits], dim=1)
        ce = F.cross_entropy(logits, y, reduction="none")
        norm_guard = ((weight.norm(dim=1) / target_norm - 1.0) ** 2).mean()
        loss = (ce * weights).mean() + 5e-4 * norm_guard
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weight, bias], 5.0)
        optimizer.step()
        if step in (0, steps - 1):
            print(f"head step={step + 1}/{steps} loss={loss.item():.5f}")
    return weight.detach(), bias.detach()


def balanced_score(logits, targets):
    prediction = logits.argmax(dim=1)
    old = targets < N_OLD
    old_acc = (prediction[old] == targets[old]).float().mean().item()
    new_acc = (prediction[~old] == targets[~old]).float().mean().item()
    return 0.5 * (old_acc + new_acc), old_acc, new_acc


def select_calibration(old_logits, new_logits, targets):
    best = None
    for scale in np.linspace(0.75, 1.25, 7):
        for shift in np.linspace(-5.0, 5.0, 81):
            logits = torch.cat(
                [old_logits, new_logits * float(scale) + float(shift)], dim=1
            )
            score, old_acc, new_acc = balanced_score(logits, targets)
            candidate = (
                score,
                min(old_acc, new_acc),
                -abs(float(scale) - 1.0) - 0.01 * abs(float(shift)),
                float(scale),
                float(shift),
                old_acc,
                new_acc,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    return best


@torch.inference_mode()
def predict(model, rows, archive_dir, extractor, device):
    output = []
    for batch_rows, input_values in batches(rows, archive_dir, extractor):
        values = input_values.to(device, non_blocking=True)
        if device.type == "cuda":
            values = values.half()
        logits = model(values)  # one shared AST forward -> complete [B,29] tensor
        output.extend(logits.argmax(dim=1).cpu().tolist())
    return output


def main():
    started = time.time()
    seed_everything()
    if not torch.cuda.is_available():
        raise RuntimeError("This timed solution requires the configured Kaggle GPU")
    device = torch.device("cuda")

    train_csv = find_competition_csv("train.csv")
    fine_tune_csv = find_competition_csv("fine_tune.csv")
    template_csv = find_competition_csv("submission.csv")
    archive_dir = train_csv.parent
    train_rows = read_rows(train_csv)
    fine_tune_rows = read_rows(fine_tune_csv)
    test_rows = read_rows(template_csv)
    labeled_rows = train_rows + fine_tune_rows

    targets = [int(row["target"]) for row in labeled_rows]
    if sorted(set(targets)) != list(range(N_CLASSES)):
        raise ValueError("expected labels 0..28 exactly")
    if any(not (archive_dir / row["path"]).is_file() for row in labeled_rows + test_rows):
        raise FileNotFoundError("one or more CSV audio paths are missing")

    model_dir = find_model_dir()
    extractor = ASTFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    model = IncrementalAST(model_dir).eval().to(device)
    model.half()
    old_weight_reference = model.old_classifier.dense.weight.detach().clone()
    old_bias_reference = model.old_classifier.dense.bias.detach().clone()

    embeddings, old_logits, y = encode_labeled(
        model, labeled_rows, archive_dir, extractor, device
    )
    print(f"encoded labeled rows: {tuple(embeddings.shape)}")

    fold_id = stratified_folds(y.numpy(), folds=3)
    oof_new = torch.empty(len(y), N_NEW)
    for fold in range(3):
        train_mask = torch.from_numpy(fold_id != fold)
        valid_mask = torch.from_numpy(fold_id == fold)
        weight, bias = fit_new_rows(
            embeddings[train_mask],
            old_logits[train_mask],
            y[train_mask],
            model.old_classifier.dense.weight.float(),
            model.old_classifier.dense.bias.float(),
            HEAD_STEPS,
        )
        oof_new[valid_mask] = F.linear(
            embeddings[valid_mask].to(device), weight, bias
        ).cpu()
        del weight, bias
        gc.collect()
        torch.cuda.empty_cache()

    calibration = select_calibration(old_logits, oof_new, y)
    score, _, _, scale, shift, old_acc, new_acc = calibration
    print(
        f"OOF balanced={score:.5f} old={old_acc:.5f} new={new_acc:.5f} "
        f"new_scale={scale:.4f} new_shift={shift:.4f}"
    )

    final_weight, final_bias = fit_new_rows(
        embeddings,
        old_logits,
        y,
        model.old_classifier.dense.weight.float(),
        model.old_classifier.dense.bias.float(),
        HEAD_STEPS,
    )
    model.set_new_head(final_weight, final_bias, scale, shift)
    model.eval()

    if not torch.equal(model.old_classifier.dense.weight, old_weight_reference):
        raise RuntimeError("old classifier weights changed")
    if not torch.equal(model.old_classifier.dense.bias, old_bias_reference):
        raise RuntimeError("old classifier bias changed")

    predictions = predict(model, test_rows, archive_dir, extractor, device)
    if len(predictions) != len(test_rows):
        raise RuntimeError("prediction count mismatch")
    if not all(isinstance(value, int) and 0 <= value < N_CLASSES for value in predictions):
        raise RuntimeError("invalid prediction value")

    submission_path = WORK / "submission.csv"
    with submission_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "target"])
        writer.writerows(
            (row["path"], int(prediction))
            for row, prediction in zip(test_rows, predictions)
        )

    written = read_rows(submission_path)
    if [row["path"] for row in written] != [row["path"] for row in test_rows]:
        raise RuntimeError("submission paths/order changed")
    elapsed = time.time() - started
    metrics = {
        "candidate": "trace_2_geometry_group_balanced",
        "oof_balanced_accuracy": score,
        "oof_old_accuracy": old_acc,
        "oof_new_accuracy": new_acc,
        "new_scale": scale,
        "new_shift": shift,
        "elapsed_seconds": elapsed,
        "rows": {"old": len(train_rows), "new": len(fine_tune_rows), "test": len(test_rows)},
        "single_forward": True,
        "old_rows_immutable": True,
    }
    (WORK / "trace_2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {submission_path}")


if __name__ == "__main__":
    main()
