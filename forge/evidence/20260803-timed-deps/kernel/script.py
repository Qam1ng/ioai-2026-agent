# ─── IOAI 2026 — environment setup. Keep this at the VERY TOP of your script. ───
# Installs the competition's pinned packages from the mounted wheel files.
# No internet is used. It must run BEFORE you import any of those packages:
# pip cannot change a module that Python has already loaded.
#
# Attach the wheel dataset to your notebook first:
#     Add data -> Datasets -> "ioai-2026-wheel-dataset"
# (or fork the official starter, which already has it attached).
#
# Two steps, because it makes the difference between 40 seconds and 6 minutes:
#   1. install `uv` (a fast installer) with pip — one small wheel, ~6s
#   2. use `uv` to install everything else                        — ~34s
import os
import re
import subprocess
import sys
from pathlib import Path

# torch ships as e.g. torch-2.13.0+cu126-...whl. Kaggle strips the "+" from
# uploaded filenames, leaving torch-2.13.0cu126-... — which is not a parseable
# version, so pip and uv do not recognise the file as torch at all. Normalising
# the name to torch-2.13.0-... fixes that, but then the filename disagrees with
# the wheel's internal metadata, which uv rejects unless told the mismatch is
# intentional. Hence both the rename and UV_SKIP_WHEEL_FILENAME_CHECK below.
WHEEL_DATASET = "ioai-2026-wheel-dataset"

_LOCAL_VER = re.compile(r"^(?P<name>[^-]+)-(?P<ver>\d[\d.]*)(?:\+?cu\d+)-(?P<rest>.+\.whl)$")


def _find_wheels(search_root="/kaggle/input", depth=7):
    """Locate the wheel directory by looking for the meta-wheel.

    Never hard-code the path: Kaggle mounts competition data under
    /kaggle/input/competitions/<slug>/ and datasets under
    /kaggle/input/datasets/<owner>/<slug>/, and the nesting can change.
    """
    base = Path(search_root)
    for d in range(depth):
        hits = sorted(base.glob("/".join(["*"] * d + ["ioai_env-*.whl"])))
        if hits:
            return hits[0].parent
    present = sorted(str(p.relative_to(base)) for p in base.glob("*"))[:20]
    raise FileNotFoundError(
        f"IOAI wheels not found under {base}.\n"
        f"  {base} currently contains: {present or '<nothing>'}\n\n"
        f"  Add the wheel dataset to this notebook:\n"
        f"    Add data -> Datasets -> search '{WHEEL_DATASET}'\n"
        f"  (or fork the official starter notebook, which already has it attached)."
    )


def _normalise(wheels):
    """Return a directory whose wheel filenames all parse.

    Symlinks only — nothing is copied, so this costs no disk and no time.
    If every name is already fine, the original directory is used as-is.
    """
    needs_fix = [w for w in wheels.glob("*.whl") if _LOCAL_VER.match(w.name)]
    if not needs_fix:
        return wheels
    work = Path("/kaggle/working/_ioai_wheels")
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob("*"):
        stale.unlink()
    for w in sorted(wheels.glob("*.whl")):
        m = _LOCAL_VER.match(w.name)
        name = f"{m['name']}-{m['ver']}-{m['rest']}" if m else w.name
        (work / name).symlink_to(w)
    print(f"normalised {len(needs_fix)} wheel filename(s) into {work}")
    return work


def setup_ioai_env(search_root="/kaggle/input"):
    """Install the pinned environment offline. Returns the wheel directory."""
    wheels = _normalise(_find_wheels(search_root))
    common = ["--no-index", f"--find-links={wheels}"]

    # Step 1 — bootstrap uv itself with pip. One wheel, no dependencies.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           *common, "uv"])

    # Step 2 — let uv do the heavy lifting. `-m uv` rather than the `uv`
    # executable, so this does not depend on PATH inside the kernel.
    env = dict(os.environ, UV_SKIP_WHEEL_FILENAME_CHECK="1")
    try:
        subprocess.check_call([sys.executable, "-m", "uv", "pip", "install",
                               "--system", *common, "ioai-env"], env=env)
    except subprocess.CalledProcessError:
        # Falling back is deliberate: a slow run beats a failed submission.
        print("WARNING: uv failed — falling back to pip. This takes ~6 minutes "
              "instead of ~35 seconds, but it will work.", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--only-binary=:all:", *common, "ioai-env"])

    print(f"IOAI environment ready (wheels from {wheels})")
    return wheels


setup_ioai_env()
# ───────────────────────────────────────────────────────────────────────────────

# ─── Your solution below ───────────────────────────────────────────────────────
import csv
import copy
import math
import random

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ASTFeatureExtractor, ASTForAudioClassification


SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def find_input(name, search_root="/kaggle/input"):
    """Find a competition file by name. Never hard-code a path — Kaggle mounts
    competition data under /kaggle/input/competitions/<slug>/ and the nesting
    can change."""
    matches = sorted(Path(search_root).rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {search_root}")
    return matches[0]


def read_rows(name):
    with find_input(name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_audio(path, target_rate=16000):
    wave, rate = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != target_rate:
        divisor = math.gcd(int(rate), target_rate)
        wave = resample_poly(
            wave, target_rate // divisor, int(rate) // divisor
        ).astype(np.float32, copy=False)
    return np.asarray(wave, dtype=np.float32)


train_rows = read_rows("train.csv")
new_rows = read_rows("fine_tune.csv")
test_rows = read_rows("submission.csv")
assert train_rows and new_rows and test_rows
assert {int(row["target"]) for row in train_rows} == set(range(16))
assert {int(row["target"]) for row in new_rows} == set(range(16, 29))

audio_paths = {
    path.name: path
    for path in Path("/kaggle/input").rglob("*.wav")
    if "competitions" in str(path)
}
needed_names = {
    Path(row["path"]).name for row in train_rows + new_rows + test_rows
}
assert needed_names <= set(audio_paths), (
    len(needed_names), len(audio_paths), sorted(needed_names - set(audio_paths))[:5]
)

model_dir = find_input("model.safetensors").parent
processor = ASTFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
base = ASTForAudioClassification.from_pretrained(model_dir, local_files_only=True)
assert base.config.num_labels == 16
for parameter in base.parameters():
    parameter.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "This competition provides a T4 GPU"
base.to(device).eval()


@torch.inference_mode()
def encode(rows, batch_size=12):
    all_features = []
    all_old_logits = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        waves = [load_audio(audio_paths[Path(row["path"]).name]) for row in batch]
        values = processor(
            waves,
            sampling_rate=16000,
            padding="max_length",
            return_tensors="pt",
        )["input_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pooled = base.audio_spectrogram_transformer(values).pooler_output
            features = base.classifier.layernorm(pooled)
            old_logits = base.classifier.dense(features)
        all_features.append(features.float().cpu())
        all_old_logits.append(old_logits.float().cpu())
    return torch.cat(all_features), torch.cat(all_old_logits)


all_rows = train_rows + new_rows
all_targets = torch.tensor([int(row["target"]) for row in all_rows], dtype=torch.long)
print(f"encoding {len(all_rows)} labelled and {len(test_rows)} test clips", flush=True)
features, old_logits = encode(all_rows)
test_features, test_old_logits = encode(test_rows)

# Deterministic within-class holdout. It is used only for model selection and a
# single global old-vs-new logit offset; no test-global statistic is consulted.
train_indices = []
valid_indices = []
for label in range(29):
    members = torch.where(all_targets == label)[0].tolist()
    for position, index in enumerate(members):
        (valid_indices if position % 5 == 0 else train_indices).append(index)
train_indices = torch.tensor(train_indices, dtype=torch.long)
valid_indices = torch.tensor(valid_indices, dtype=torch.long)

train_features = features[train_indices].to(device)
train_old = old_logits[train_indices].to(device)
train_targets = all_targets[train_indices].to(device)
valid_features = features[valid_indices].to(device)
valid_old = old_logits[valid_indices].to(device)
valid_targets = all_targets[valid_indices].to(device)

new_head = torch.nn.Linear(features.shape[1], 13).to(device)
torch.nn.init.normal_(new_head.weight, std=0.02)
torch.nn.init.constant_(new_head.bias, -2.0)
initial_state = copy.deepcopy(new_head.state_dict())

counts = torch.bincount(train_targets, minlength=29).float()
class_weights = counts.sum() / counts.clamp_min(1.0)
class_weights /= class_weights.mean()
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(new_head.parameters(), lr=0.02, weight_decay=0.01)


def balanced_score(logits, targets):
    predictions = logits.argmax(dim=1)
    old_mask = targets < 16
    new_mask = ~old_mask
    old_acc = (predictions[old_mask] == targets[old_mask]).float().mean().item()
    new_acc = (predictions[new_mask] == targets[new_mask]).float().mean().item()
    return 0.5 * old_acc + 0.5 * new_acc, old_acc, new_acc


best_score = -1.0
best_epoch = 0
best_state = None
stale = 0
for epoch in range(301):
    new_head.train()
    optimizer.zero_grad(set_to_none=True)
    logits = torch.cat([train_old, new_head(train_features)], dim=1)
    loss = loss_fn(logits, train_targets)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        new_head.eval()
        with torch.no_grad():
            valid_logits = torch.cat([valid_old, new_head(valid_features)], dim=1)
            score, old_acc, new_acc = balanced_score(valid_logits, valid_targets)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(new_head.state_dict())
            stale = 0
        else:
            stale += 5
        if stale >= 60:
            break

assert best_state is not None
new_head.load_state_dict(best_state)
new_head.eval()

# One train-defined global group calibration parameter. The final classifier is
# still a single 29-way head over one AST embedding.
best_offset = 0.0
best_calibrated = -1.0
with torch.no_grad():
    valid_new = new_head(valid_features)
    for offset in torch.linspace(-6.0, 6.0, 49, device=device):
        candidate = torch.cat([valid_old, valid_new + offset], dim=1)
        score, _, _ = balanced_score(candidate, valid_targets)
        if score > best_calibrated:
            best_calibrated = score
            best_offset = float(offset.item())

print(
    f"best_epoch={best_epoch} raw_valid={best_score:.6f} "
    f"calibrated_valid={best_calibrated:.6f} new_offset={best_offset:.3f}",
    flush=True,
)

# Refit the selected head on every labeled row for the validation-selected
# number of updates. The holdout chose only training duration and calibration;
# the final fit does not inspect any test-derived statistic.
final_head = torch.nn.Linear(features.shape[1], 13).to(device)
final_head.load_state_dict(initial_state)
full_features = features.to(device)
full_old = old_logits.to(device)
full_targets = all_targets.to(device)
full_counts = torch.bincount(full_targets, minlength=29).float()
full_weights = full_counts.sum() / full_counts.clamp_min(1.0)
full_weights /= full_weights.mean()
final_loss_fn = torch.nn.CrossEntropyLoss(weight=full_weights)
final_optimizer = torch.optim.AdamW(final_head.parameters(), lr=0.02, weight_decay=0.01)
final_head.train()
for _ in range(best_epoch + 1):
    final_optimizer.zero_grad(set_to_none=True)
    final_logits = torch.cat([full_old, final_head(full_features)], dim=1)
    final_loss = final_loss_fn(final_logits, full_targets)
    final_loss.backward()
    final_optimizer.step()
final_head.eval()
print(f"refit_rows={len(all_rows)} refit_updates={best_epoch + 1}", flush=True)

with torch.inference_mode():
    test_new_logits = final_head(test_features.to(device)).cpu()
    combined = torch.cat([test_old_logits, test_new_logits + best_offset], dim=1)
    predictions = combined.argmax(dim=1).numpy().astype(int)

assert len(predictions) == len(test_rows)
assert predictions.min() >= 0 and predictions.max() <= 28

SUBMISSION = Path("/kaggle/working/submission.csv")
with SUBMISSION.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "target"])
    for row, prediction in zip(test_rows, predictions):
        w.writerow([row["path"], int(prediction)])

assert SUBMISSION.is_file() and SUBMISSION.stat().st_size > 0
with SUBMISSION.open(newline="") as fh:
    check_rows = list(csv.DictReader(fh))
assert len(check_rows) == len(test_rows)
assert [row["path"] for row in check_rows] == [row["path"] for row in test_rows]
assert all(0 <= int(row["target"]) <= 28 for row in check_rows)
print(f"wrote {SUBMISSION} with {len(check_rows)} rows")
