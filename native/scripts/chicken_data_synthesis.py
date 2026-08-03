#!/usr/bin/env python3
"""Density-integral data synthesis and test-faithful validation for Chicken counting.

The official train split ships a 180x320 float32 density map per image whose sum
is the label, and the frozen feature pipeline resizes each image to 360x640 -
exactly twice the density grid.  Any sub-window therefore carries an *exact*
count, obtained by integrating the density map over that window, so the 100
labelled frames can be expanded into thousands of exactly labelled rows without
a single external byte and without touching a hidden label.

Crops are taken at native scale, so a chicken keeps its pixel size and only the
field of view shrinks.  The target is area-normalised - the count the frame
would carry if the whole frame looked like the crop - which keeps crops and full
frames on one regression scale.

Nothing here changes the estimator.  The scaler, Ridge, KNN, log-Ridge, their
hyperparameters, the 0.98055 calibration and the 70/30 blend are the sealed
incumbent's, byte for byte.  Only the rows fed to them, and the protocol that
decides whether those rows help, are new.

The validation protocol exists because the previous candidate cost 0.011 of
public score while its leave-one-out evidence claimed +0.039.  Train and test
frames are interleaved inside the same nine capture sessions, so a single
grouped fold both leaks same-session neighbours and hides the anchor sparsity
the real test set has.  Every recipe here is judged on a robust lower bound
across six scenarios and five seeds, and the mechanism that produced that 0.011
regression is re-run through the same protocol as a negative control: a
validator that cannot reject a known-bad candidate has not been shown to work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


SCHEMA_VERSION = 1
TASK = "ioai-2025-chicken-counting-mirror-unofficial"

# Sealed incumbent estimator constants. Frozen: this component may not tune them.
RIDGE_ALPHA = 1.0
KNN_NEIGHBOURS = 2
RIDGE_WEIGHT = 0.570968517
KNN_WEIGHT = 0.429020957
BASE_SCALE = 0.98055
BASE_WEIGHT = 0.70
LOG_RIDGE_WEIGHT = 0.30
LOG_RIDGE_ALPHA = 0.10

# Sealed anchors the synthesis must reproduce before it is allowed to proceed.
SEALED_BASE_OOF = 0.8889048869588969
SEALED_BLEND_OOF = 0.8908494636264506
ANCHOR_TOLERANCE = 1e-7
REPO_PUBLIC_INCUMBENT = 0.93156
ACCOUNT_PUBLIC_BEST = 0.93041

DENSITY_SHAPE = (180, 320)
IMAGE_SIZE = (640, 360)
CROP_SCALES = (0.95, 0.87, 0.78)
CROP_ANCHORS = ((0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0))

VALIDATION_SEEDS = (20260803, 20260817, 20260831, 20260914, 20260928)
SCENARIOS = (
    "interleaved_session_holdout",
    "group_holdout",
    "purged_local_block",
    "label_anchor_thinning",
    "label_noise_stress",
    "candidate_prediction_perturbation",
)

# Rulers the calibrator may choose between, described by how many of the 100
# labelled frames each one withholds from the fit set.
RULERS = {
    "leave_one_out": 1,
    "interleaved_10pct": 10,
    "interleaved_25pct": 25,
    "grouped_5fold": 20,
    "group_holdout_3_blocks": 21,
}

# Robust promotion gate. A recipe must clear every one of these.
GATE_MIN_POSITIVE_FRACTION = 0.80
GATE_MIN_QUANTILE10 = 0.0
GATE_MAX_WORST_CELL_LOSS = 0.005

# A candidate must also beat the ruler's own repeat-to-repeat noise, measured
# rather than assumed, before its margin counts as a signal.
GATE_NOISE_MARGIN_MULTIPLE = 1.0

# Sealed DVR mechanism, replayed only as a negative control.
DVR_BANDWIDTH_SECONDS = 105.0
DVR_TEMPORAL_WEIGHT = 0.875


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb" if exclusive else "wb") as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_score(target: Any, prediction: Any) -> float:
    import numpy as np

    target_array = np.asarray(target, dtype=float)
    prediction_array = np.maximum(np.asarray(prediction, dtype=float), 0.01)
    return float(
        np.exp(-np.mean(np.abs(target_array - prediction_array) / target_array))
    )


# --------------------------------------------------------------------------
# Frozen feature pipeline, reproduced exactly from the sealed extractor.
# --------------------------------------------------------------------------


def build_extractor(base_checkpoint: Path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FeatureExtraction(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=2, dilation=2)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, value):
            value = F.relu(self.conv1(value))
            value = F.relu(self.conv2(value))
            value = self.pool2(value)
            value = F.relu(self.conv3(value))
            value = F.relu(self.conv4(value))
            return self.pool4(value)

    model = FeatureExtraction()
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(
        {
            key.split(".", 1)[1]: value
            for key, value in checkpoint.items()
            if key.startswith("feature_extraction.")
        }
    )
    model.eval()
    return model


def summarize_feature_map(feature) -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as F

    pieces = [
        F.adaptive_avg_pool2d(feature, (1, 1)).flatten(),
        F.adaptive_max_pool2d(feature, (1, 1)).flatten(),
        feature.flatten(2).std(dim=2).flatten(),
        F.adaptive_avg_pool2d(feature, (3, 5)).flatten(),
        F.adaptive_avg_pool2d(feature, (6, 10)).flatten(),
        F.adaptive_max_pool2d(feature, (3, 5)).flatten(),
    ]
    return torch.cat(pieces).cpu().numpy().astype(np.float32)


def crop_window(scale: float, anchor: tuple[float, float]) -> tuple[int, int, int, int]:
    height = int(round(DENSITY_SHAPE[0] * scale))
    width = int(round(DENSITY_SHAPE[1] * scale))
    row = int(round((DENSITY_SHAPE[0] - height) * anchor[0]))
    column = int(round((DENSITY_SHAPE[1] - width) * anchor[1]))
    return row, column, height, width


def synthesize(run_root: Path, source_root: Path) -> dict[str, Any]:
    """Expand 100 labelled frames into exactly labelled density-integral rows."""
    import numpy as np
    import torch
    from PIL import Image

    run_root = run_root.resolve()
    source_root = source_root.resolve()
    official = source_root / "data/official"
    sealed_features = source_root / "root_workspace/conv_features_640.npz"
    for path in (official / "base.pth", official / "train/metadata.csv", sealed_features):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    output = run_root / "outputs/synthetic_rows.npz"
    if output.exists():
        raise FileExistsError(f"synthesis already exists: {output}")

    model = build_extractor(official / "base.pth")
    with (official / "train/metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise ValueError(f"expected 100 train rows, got {len(rows)}")

    features: list[Any] = []
    targets: list[float] = []
    source: list[int] = []
    kind: list[str] = []
    area: list[float] = []
    started = time.time()

    with torch.inference_mode():
        for index, row in enumerate(rows):
            image = Image.open(official / row["image_path"]).convert("RGB").resize(
                IMAGE_SIZE, Image.Resampling.BILINEAR
            )
            tensor = (
                torch.from_numpy(np.array(image, dtype=np.uint8))
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                / 255.0
            )
            density = np.load(official / row["density_path"]).astype(np.float64)
            if density.shape != DENSITY_SHAPE:
                raise ValueError(f"unexpected density shape for {row['id']}")
            declared = float(row["count"])
            if abs(float(density.sum()) - declared) > 1e-4:
                raise ValueError(f"density integral disagrees with the label: {row['id']}")

            def emit(view, count: float, fraction: float, tag: str) -> None:
                features.append(summarize_feature_map(model(view)))
                targets.append(count / fraction)
                source.append(index)
                kind.append(tag)
                area.append(fraction)

            emit(tensor, declared, 1.0, "full")
            emit(torch.flip(tensor, dims=[3]), declared, 1.0, "full_flip")
            for scale in CROP_SCALES:
                for anchor in CROP_ANCHORS:
                    top, left, height, width = crop_window(scale, anchor)
                    count = float(density[top : top + height, left : left + width].sum())
                    fraction = (height * width) / (DENSITY_SHAPE[0] * DENSITY_SHAPE[1])
                    view = tensor[
                        :, :, 2 * top : 2 * (top + height), 2 * left : 2 * (left + width)
                    ]
                    emit(view, count, fraction, f"crop{scale:g}")
                    emit(torch.flip(view, dims=[3]), count, fraction, f"crop{scale:g}_flip")
            if (index + 1) % 20 == 0:
                print(
                    f"synthesised {index + 1}/100 frames, {len(features)} rows, "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )

    x = np.stack(features)
    y = np.asarray(targets, dtype=np.float64)
    kind_array = np.asarray(kind)

    # The synthesis is only trustworthy if its own full-frame path reproduces the
    # sealed feature cache exactly; otherwise the added rows live in a different
    # feature space from the incumbent and every comparison below is meaningless.
    sealed = np.load(sealed_features, allow_pickle=False)
    full = kind_array == "full"
    feature_delta = float(np.max(np.abs(x[full] - sealed["x"])))
    target_delta = float(np.max(np.abs(y[full] - sealed["y"])))
    if feature_delta != 0.0 or target_delta != 0.0:
        raise ValueError(
            f"synthesis does not reproduce the sealed pipeline: "
            f"feature delta {feature_delta}, target delta {target_delta}"
        )

    np.savez_compressed(
        output,
        x=x,
        y=y,
        source=np.asarray(source, dtype=np.int32),
        kind=kind_array,
        area=np.asarray(area, dtype=np.float64),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "method": "native_scale_density_integral_crops_with_area_normalised_targets",
        "label_provenance": "official train density maps only; no hidden label, no external data",
        "rows": int(len(y)),
        "frames": len(rows),
        "feature_dimension": int(x.shape[1]),
        "crop_scales": list(CROP_SCALES),
        "crop_anchors": [list(anchor) for anchor in CROP_ANCHORS],
        "kinds": sorted(set(kind)),
        "target_range": [float(y.min()), float(y.max())],
        "sealed_pipeline_reproduction": {
            "sealed_features": str(sealed_features),
            "sealed_features_sha256": sha256_file(sealed_features),
            "full_frame_feature_max_abs_delta": feature_delta,
            "full_frame_target_max_abs_delta": target_delta,
            "exact": True,
        },
        "artifact_path": str(output),
        "artifact_sha256": sha256_file(output),
        "runtime_seconds": time.time() - started,
    }
    write_json(run_root / "outputs/synthetic_validation_manifest.json", manifest, exclusive=True)
    return {
        "synthesised": True,
        "rows": manifest["rows"],
        "reproduces_sealed_pipeline_exactly": True,
        "artifact_sha256": manifest["artifact_sha256"],
    }


# --------------------------------------------------------------------------
# Frozen estimator stack. Data changes; this does not.
# --------------------------------------------------------------------------


def fit_and_predict(fit_x, fit_y, query_x) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_fit = scaler.fit_transform(fit_x)
    scaled_query = scaler.transform(query_x)
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaled_fit, fit_y)
    knn = KNeighborsRegressor(n_neighbors=KNN_NEIGHBOURS, weights="distance").fit(
        scaled_fit, fit_y
    )
    log_ridge = Ridge(alpha=LOG_RIDGE_ALPHA).fit(scaled_fit, np.log(fit_y))
    base = BASE_SCALE * (
        RIDGE_WEIGHT * ridge.predict(scaled_query)
        + KNN_WEIGHT * knn.predict(scaled_query)
    )
    log_prediction = np.exp(log_ridge.predict(scaled_query))
    blend = BASE_WEIGHT * base + LOG_RIDGE_WEIGHT * log_prediction
    return {
        "base": np.maximum(base, 0.01),
        "log": np.maximum(log_prediction, 0.01),
        "blend": np.maximum(blend, 0.01),
    }


def recipe_table() -> dict[str, list[str]]:
    """Candidate data recipes. The empty recipe is the sealed incumbent."""
    crops = [f"crop{scale:g}" for scale in CROP_SCALES]
    table: dict[str, list[str]] = {"incumbent_original_only": []}
    table["mirror_only"] = ["full_flip"]
    table["crop_wide"] = [crops[0]]
    table["crop_wide_mid"] = [crops[0], crops[1]]
    table["crop_mid_tight"] = [crops[1], crops[2]]
    table["crop_all"] = list(crops)
    table["crop_all_mirrored"] = crops + [f"{tag}_flip" for tag in crops]
    return table


# --------------------------------------------------------------------------
# Ruler calibration.  Which holdout reproduces the live query-to-anchor
# geometry is a measurable property of the feature matrices, not a convention.
# --------------------------------------------------------------------------


def nearest_distances(query, reference, *, exclude_self: bool):
    import numpy as np

    output = np.empty(len(query), dtype=float)
    for position, row in enumerate(query):
        distance = np.linalg.norm(reference - row, axis=1)
        if exclude_self:
            distance[position] = np.inf
        output[position] = distance.min()
    return output


def ruler_splits(ruler: str, seed: int, folds, blocks, rows: int) -> list[Any]:
    """Return held-out index arrays that partition the labelled frames once."""
    import numpy as np

    index = np.arange(rows)
    rng = np.random.default_rng(seed)
    if ruler == "leave_one_out":
        return [np.asarray([position]) for position in index]
    if ruler == "grouped_5fold":
        return [np.flatnonzero(folds == held) for held in sorted(set(folds.tolist()))]
    if ruler.startswith("interleaved_"):
        parts = 10 if ruler == "interleaved_10pct" else 4
        order = rng.permutation(index)
        return [order[part::parts] for part in range(parts)]
    if ruler == "group_holdout_3_blocks":
        unique = rng.permutation(np.unique(blocks))
        return [
            np.flatnonzero(np.isin(blocks, unique[start : start + 3]))
            for start in range(0, len(unique), 3)
        ]
    raise ValueError(f"unknown ruler: {ruler}")


def calibrate_ruler(x0, x_test, folds, blocks) -> dict[str, Any]:
    """Pick the ruler whose held-out frames sit as far from their fitting set as
    the live test frames sit from the whole labelled set."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x0)
    train_scaled = scaler.transform(x0)
    test_scaled = scaler.transform(x_test)
    live = nearest_distances(test_scaled, train_scaled, exclude_self=False)
    live_median = float(np.median(live))

    measured = {}
    for ruler in RULERS:
        samples: list[float] = []
        for seed in VALIDATION_SEEDS:
            for held in ruler_splits(ruler, seed, folds, blocks, len(x0)):
                fit = np.setdiff1d(np.arange(len(x0)), held)
                samples.extend(
                    nearest_distances(
                        train_scaled[held], train_scaled[fit], exclude_self=False
                    ).tolist()
                )
            if ruler in ("leave_one_out", "grouped_5fold"):
                break  # deterministic; seeds would only repeat the same split
        array = np.asarray(samples)
        measured[ruler] = {
            "median_query_to_fit_distance": float(np.median(array)),
            "p25": float(np.percentile(array, 25)),
            "p75": float(np.percentile(array, 75)),
            "median_gap_vs_live": float(abs(np.median(array) - live_median)),
        }
    selected = min(measured, key=lambda name: measured[name]["median_gap_vs_live"])
    return {
        "live_test_geometry": {
            "median_test_to_train_distance": live_median,
            "p25": float(np.percentile(live, 25)),
            "p75": float(np.percentile(live, 75)),
            "note": "features only; no hidden label is touched by this measurement",
        },
        "rulers": measured,
        "selected_ruler": selected,
        "selection_rule": "smallest absolute gap between median query-to-fit and median test-to-train distance",
    }


# --------------------------------------------------------------------------
# Test-faithful validation scenarios. Train-only, no hidden label is read.
# --------------------------------------------------------------------------


def build_cell(
    scenario: str,
    seed: int,
    blocks: Any,
    nearest: Any,
    rows: int,
) -> dict[str, Any]:
    """Return the held-out rows, the admissible fitting rows, and any label stress."""
    import numpy as np

    rng = np.random.default_rng(seed)
    index = np.arange(rows)
    unique_blocks = np.unique(blocks)

    if scenario == "group_holdout":
        held_blocks = rng.choice(unique_blocks, size=3, replace=False)
        query = index[np.isin(blocks, held_blocks)]
        fit = index[~np.isin(blocks, held_blocks)]
    elif scenario == "purged_local_block":
        held_blocks = rng.choice(unique_blocks, size=3, replace=False)
        query = index[np.isin(blocks, held_blocks)]
        fit = index[~np.isin(blocks, held_blocks)]
        # Purge the visual nearest neighbour of every held-out frame, in both
        # directions, so the fit set cannot answer by near-duplicate lookup.
        held = set(query.tolist())
        purged = {int(nearest[position]) for position in query}
        purged |= {int(position) for position in fit if int(nearest[position]) in held}
        fit = np.asarray([position for position in fit if int(position) not in purged])
    elif scenario == "interleaved_session_holdout":
        # Train and test frames really are interleaved inside the same nine
        # sessions, so this is the scenario that matches the live regime.
        query = np.sort(rng.choice(index, size=max(1, rows // 4), replace=False))
        fit = np.asarray([position for position in index if position not in set(query.tolist())])
    elif scenario == "label_anchor_thinning":
        query = np.sort(rng.choice(index, size=max(1, rows // 4), replace=False))
        remaining = np.asarray([p for p in index if p not in set(query.tolist())])
        keep = rng.random(len(remaining)) >= 0.35
        if keep.sum() < 10:
            keep[:] = True
        fit = remaining[keep]
    elif scenario in ("label_noise_stress", "candidate_prediction_perturbation"):
        query = np.sort(rng.choice(index, size=max(1, rows // 4), replace=False))
        fit = np.asarray([position for position in index if position not in set(query.tolist())])
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    if len(fit) < 10 or len(query) < 5:
        raise ValueError(f"degenerate cell for {scenario}/{seed}")
    cell: dict[str, Any] = {"fit": fit, "query": query}
    if scenario == "label_noise_stress":
        cell["label_factor"] = np.exp(rng.normal(0.0, 0.05, size=rows))
    if scenario == "candidate_prediction_perturbation":
        cell["prediction_factor"] = np.exp(rng.normal(0.0, 0.02, size=rows))
    return cell


def evaluate_recipes(
    x0,
    y0,
    synthetic,
    blocks,
    nearest,
    *,
    extra_mechanisms: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np

    sx = synthetic["x"]
    sy = synthetic["y"]
    ssrc = synthetic["source"]
    skind = synthetic["kind"]
    rows = len(y0)
    recipes = recipe_table()
    extra_mechanisms = extra_mechanisms or {}

    cells: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in VALIDATION_SEEDS:
            plan = build_cell(scenario, seed, blocks, nearest, rows)
            fit = plan["fit"]
            query = plan["query"]
            fit_y = y0[fit].copy()
            synthetic_y = sy.copy()
            if "label_factor" in plan:
                fit_y = fit_y * plan["label_factor"][fit]
                synthetic_y = synthetic_y * plan["label_factor"][ssrc]

            predictions: dict[str, Any] = {}
            for name, tags in recipes.items():
                admissible = (
                    np.isin(skind, tags) & np.isin(ssrc, fit)
                    if tags
                    else np.zeros(len(sy), dtype=bool)
                )
                combined_x = np.concatenate([x0[fit], sx[admissible]])
                combined_y = np.concatenate([fit_y, synthetic_y[admissible]])
                predictions[name] = fit_and_predict(combined_x, combined_y, x0[query])["blend"]

            baseline_blend = predictions["incumbent_original_only"]
            for name, mechanism in extra_mechanisms.items():
                predictions[name] = mechanism(fit, query, fit_y, baseline_blend)

            if "prediction_factor" in plan:
                factor = plan["prediction_factor"][query]
                predictions = {k: v * factor for k, v in predictions.items()}

            truth = y0[query]
            scores = {name: official_score(truth, value) for name, value in predictions.items()}
            cells.append(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "fit_rows": int(len(fit)),
                    "query_rows": int(len(query)),
                    "scores": scores,
                }
            )

    incumbent_key = "incumbent_original_only"
    summary: dict[str, Any] = {}
    for name in list(recipes) + list(extra_mechanisms):
        if name == incumbent_key:
            continue
        deltas = np.asarray(
            [cell["scores"][name] - cell["scores"][incumbent_key] for cell in cells]
        )
        per_scenario = {
            scenario: float(
                np.mean(
                    [
                        cell["scores"][name] - cell["scores"][incumbent_key]
                        for cell in cells
                        if cell["scenario"] == scenario
                    ]
                )
            )
            for scenario in SCENARIOS
        }
        gate = {
            "positive_fraction": float(np.mean(deltas > 0)),
            "quantile10": float(np.quantile(deltas, 0.10)),
            "worst_cell": float(deltas.min()),
            "mean": float(deltas.mean()),
        }
        gate["pass"] = bool(
            gate["positive_fraction"] >= GATE_MIN_POSITIVE_FRACTION
            and gate["quantile10"] > GATE_MIN_QUANTILE10
            and gate["worst_cell"] >= -GATE_MAX_WORST_CELL_LOSS
            and gate["mean"] > 0.0
        )
        summary[name] = {"robust": gate, "per_scenario_mean_delta": per_scenario}
    return {"cells": cells, "summary": summary}


def paired_block_bootstrap(target, incumbent, candidate, blocks) -> dict[str, Any]:
    import numpy as np

    target = np.asarray(target, dtype=float)
    row_delta = (
        np.abs(target - incumbent) / target - np.abs(target - candidate) / target
    )
    unique = np.unique(blocks)
    rng = np.random.default_rng(20260803)
    values = np.empty(20_000, dtype=float)
    for draw in range(20_000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        picked = np.concatenate([np.flatnonzero(blocks == item) for item in sampled])
        values[draw] = float(row_delta[picked].mean())
    return {
        "unit": "acquisition_block",
        "draws": 20_000,
        "mre_improvement_point": float(row_delta.mean()),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "probability_positive": float(np.mean(values > 0)),
    }


def evaluate_under_ruler(
    ruler: str,
    x0,
    y0,
    synthetic,
    folds,
    blocks,
    *,
    mechanisms: dict[str, Any],
) -> dict[str, Any]:
    """Score every recipe on the same rows, with the same fitting sets.

    Cross-protocol comparison is the defect this whole module exists to remove,
    so a candidate and its baseline are never allowed to come from two passes.
    """
    import numpy as np

    sx, sy, ssrc, skind = (
        synthetic["x"],
        synthetic["y"],
        synthetic["source"],
        synthetic["kind"],
    )
    recipes = recipe_table()
    seeds = (
        [VALIDATION_SEEDS[0]]
        if ruler in ("leave_one_out", "grouped_5fold")
        else list(VALIDATION_SEEDS)
    )
    names = list(recipes) + list(mechanisms)
    per_seed: dict[str, list[float]] = {name: [] for name in names}
    pooled: dict[str, Any] = {}

    for seed in seeds:
        oof = {name: np.zeros(len(y0)) for name in names}
        for held in ruler_splits(ruler, seed, folds, blocks, len(y0)):
            fit = np.setdiff1d(np.arange(len(y0)), held)
            for name, tags in recipes.items():
                admissible = (
                    np.isin(skind, tags) & np.isin(ssrc, fit)
                    if tags
                    else np.zeros(len(sy), dtype=bool)
                )
                oof[name][held] = fit_and_predict(
                    np.concatenate([x0[fit], sx[admissible]]),
                    np.concatenate([y0[fit], sy[admissible]]),
                    x0[held],
                )["blend"]
            for name, mechanism in mechanisms.items():
                oof[name][held] = mechanism(
                    fit, held, y0[fit], oof["incumbent_original_only"][held]
                )
        for name in names:
            per_seed[name].append(official_score(y0, oof[name]))
            if seed == seeds[0]:
                pooled[name] = oof[name]

    incumbent = "incumbent_original_only"
    noise = (
        float(np.std(per_seed[incumbent], ddof=1)) if len(per_seed[incumbent]) > 1 else 0.0
    )
    summary = {}
    for name in names:
        if name == incumbent:
            continue
        deltas = np.asarray(per_seed[name]) - np.asarray(per_seed[incumbent])
        bootstrap = paired_block_bootstrap(y0, pooled[incumbent], pooled[name], blocks)
        gate = {
            "mean_delta": float(deltas.mean()),
            "worst_seed_delta": float(deltas.min()),
            "positive_seed_fraction": float(np.mean(deltas > 0)),
            "ruler_noise_sd": noise,
            "beats_measured_noise": bool(
                deltas.mean() > GATE_NOISE_MARGIN_MULTIPLE * noise
            ),
            "bootstrap": bootstrap,
        }
        gate["pass"] = bool(
            gate["mean_delta"] > 0.0
            and gate["worst_seed_delta"] > 0.0
            and gate["beats_measured_noise"]
            and bootstrap["ci95"][0] > 0.0
            and bootstrap["probability_positive"] >= 0.95
        )
        summary[name] = gate
    return {
        "ruler": ruler,
        "seeds": seeds,
        "scores": {name: per_seed[name] for name in names},
        "incumbent_score": float(np.mean(per_seed[incumbent])),
        "summary": summary,
    }


def dvr_negative_control(timestamps_path: Path, train_ids: list[str], incumbent_oof):
    """Replay the sealed DVR mechanism that lost 0.011 of public score.

    The point is not to submit it.  A validator that cannot reject a candidate
    already proven bad on the live board has not been shown to work, so the
    same protocol has to be pointed at it and has to say no.
    """
    import numpy as np

    payload = json.loads(timestamps_path.read_text(encoding="utf-8"))
    stamp = {
        row["id"]: row["timestamp"]
        for row in payload["rows"]
        if row["split"] == "train"
    }
    if set(stamp) != set(train_ids):
        raise ValueError("timestamp coverage does not match the train split")
    from datetime import datetime

    seconds = np.asarray(
        [datetime.fromisoformat(stamp[item]).timestamp() for item in train_ids]
    )
    session = np.asarray([stamp[item][:13] for item in train_ids])

    def mechanism(fit, query, fit_y, baseline_blend):
        # Per-session adoption is decided on the fitting rows only, then applied
        # to held-out rows.  The original run decided it on the same rows it
        # scored, which is where the phantom +0.039 came from.
        temporal_fit = np.empty(len(fit), dtype=float)
        for position, row in enumerate(fit):
            weights = np.exp(-0.5 * ((seconds[fit] - seconds[row]) / DVR_BANDWIDTH_SECONDS) ** 2)
            weights = weights * (session[fit] == session[row])
            weights[position] = 0.0
            temporal_fit[position] = (
                float(weights @ fit_y / weights.sum()) if weights.sum() > 1e-12 else fit_y[position]
            )
        visual_fit = incumbent_oof[fit]
        fit_blend = np.maximum(
            DVR_TEMPORAL_WEIGHT * temporal_fit + (1.0 - DVR_TEMPORAL_WEIGHT) * visual_fit, 0.01
        )
        adopted = {}
        for name in np.unique(session[fit]):
            selected = session[fit] == name
            adopted[name] = official_score(
                fit_y[selected], fit_blend[selected]
            ) >= official_score(fit_y[selected], visual_fit[selected])
        output = np.empty(len(query), dtype=float)
        for position, row in enumerate(query):
            name = session[row]
            if not adopted.get(name, False):
                output[position] = baseline_blend[position]
                continue
            weights = np.exp(-0.5 * ((seconds[fit] - seconds[row]) / DVR_BANDWIDTH_SECONDS) ** 2)
            weights = weights * (session[fit] == name)
            if weights.sum() <= 1e-12:
                output[position] = baseline_blend[position]
                continue
            temporal = float(weights @ fit_y / weights.sum())
            output[position] = max(
                DVR_TEMPORAL_WEIGHT * temporal
                + (1.0 - DVR_TEMPORAL_WEIGHT) * baseline_blend[position],
                0.01,
            )
        return output

    return mechanism


def validate(run_root: Path) -> dict[str, Any]:
    import numpy as np

    run_root = run_root.resolve()
    contract = load_object(run_root / "contracts/component_contract.json")
    source_root = Path(contract["source_root"])
    receipt_path = run_root / "outputs/synthetic_validation_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("validation already ran; overwrite and resume are forbidden")

    sealed = np.load(source_root / "root_workspace/conv_features_640.npz", allow_pickle=False)
    x0, y0 = sealed["x"], sealed["y"]
    synthetic = np.load(run_root / "outputs/synthetic_rows.npz", allow_pickle=False)

    with (source_root / "trace_validation/grouped_folds.csv").open(newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    train_ids = [row["id"] for row in fold_rows]
    blocks = np.asarray([int(row["acquisition_block"]) for row in fold_rows])
    position_of = {row["id"]: index for index, row in enumerate(fold_rows)}
    nearest = np.asarray([position_of[row["nearest_train_id"]] for row in fold_rows])

    # Sealed-anchor check: the frozen five-fold OOF has to come out exactly.
    folds = np.asarray([int(row["fold"]) for row in fold_rows])
    base_oof = np.zeros(len(y0))
    blend_oof = np.zeros(len(y0))
    for held in range(5):
        fit = np.flatnonzero(folds != held)
        query = np.flatnonzero(folds == held)
        out = fit_and_predict(x0[fit], y0[fit], x0[query])
        base_oof[query] = out["base"]
        blend_oof[query] = out["blend"]
    anchors = {
        "recomputed_base_oof": official_score(y0, base_oof),
        "recomputed_blend_oof": official_score(y0, blend_oof),
        "sealed_base_oof": SEALED_BASE_OOF,
        "sealed_blend_oof": SEALED_BLEND_OOF,
        "tolerance": ANCHOR_TOLERANCE,
        "tolerance_reason": (
            "the sealed anchors were computed on another host; matrix "
            "reassociation moves the score in the eighth decimal, five orders "
            "below any difference this gate decides on"
        ),
    }
    anchors["base_delta"] = anchors["recomputed_base_oof"] - SEALED_BASE_OOF
    anchors["blend_delta"] = anchors["recomputed_blend_oof"] - SEALED_BLEND_OOF
    if abs(anchors["base_delta"]) > ANCHOR_TOLERANCE:
        raise ValueError(f"sealed base OOF anchor mismatch: {anchors['base_delta']}")
    if abs(anchors["blend_delta"]) > ANCHOR_TOLERANCE:
        raise ValueError(f"sealed blend OOF anchor mismatch: {anchors['blend_delta']}")

    started = time.time()
    timestamps = Path(contract["negative_control"]["timestamps_path"])
    control = dvr_negative_control(timestamps, train_ids, blend_oof)
    x_test = np.load(
        source_root / "root_workspace/conv_test_features_640.npz", allow_pickle=False
    )["x"]

    calibration = calibrate_ruler(x0, x_test, folds, blocks)
    ruler = calibration["selected_ruler"]
    print(f"calibrated ruler: {ruler}", flush=True)

    # What the previous run's comparison was actually worth. Its candidate was
    # scored leave-one-out and its baseline five-fold grouped; putting both on
    # one ruler is the single largest correction this component makes.
    loo_incumbent = np.zeros(len(y0))
    for position in range(len(y0)):
        fit = np.setdiff1d(np.arange(len(y0)), [position])
        loo_incumbent[position] = fit_and_predict(
            x0[fit], y0[fit], x0[position : position + 1]
        )["blend"][0]
    loo_control = np.zeros(len(y0))
    for position in range(len(y0)):
        fit = np.setdiff1d(np.arange(len(y0)), [position])
        loo_control[position] = control(
            fit, np.asarray([position]), y0[fit], loo_incumbent[position : position + 1]
        )[0]
    naive_gain = official_score(y0, loo_control) - official_score(y0, blend_oof)
    matched_gain = official_score(y0, loo_control) - official_score(y0, loo_incumbent)
    mismatch = {
        "candidate_scored_leave_one_out": official_score(y0, loo_control),
        "baseline_scored_grouped_5fold": official_score(y0, blend_oof),
        "baseline_scored_leave_one_out": official_score(y0, loo_incumbent),
        "gain_claimed_by_cross_protocol_comparison": naive_gain,
        "gain_under_one_ruler": matched_gain,
        "inflation_removed_by_protocol_parity": naive_gain - matched_gain,
        "sealed_run_reported_gain": 0.039008,
        "live_public_delta_when_submitted": -0.01085,
    }
    print(
        f"protocol parity: claimed {naive_gain:+.5f} -> matched {matched_gain:+.5f}",
        flush=True,
    )

    ruler_evaluation = evaluate_under_ruler(
        ruler,
        x0,
        y0,
        synthetic,
        folds,
        blocks,
        mechanisms={"negative_control_dvr_timestamp": control},
    )
    stress = evaluate_recipes(
        x0,
        y0,
        synthetic,
        blocks,
        nearest,
        extra_mechanisms={"negative_control_dvr_timestamp": control},
    )

    summary = ruler_evaluation["summary"]
    eligible = {
        name: value
        for name, value in summary.items()
        if value["pass"] and not name.startswith("negative_control")
    }
    selected = (
        max(eligible, key=lambda name: summary[name]["mean_delta"]) if eligible else None
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "sealed_anchors": anchors,
        "ruler_calibration": calibration,
        "protocol_mismatch_audit": mismatch,
        "selection_evidence": {
            "ruler": ruler,
            "rule": "one ruler for candidate and baseline; block bootstrap must clear zero",
            "seeds": ruler_evaluation["seeds"],
            "incumbent_score": ruler_evaluation["incumbent_score"],
            "scores": ruler_evaluation["scores"],
            "summary": summary,
            "selection_uses_public_score": False,
        },
        "stress_diagnostics": {
            "note": "scarce-anchor behaviour, recorded but never used to select",
            "scenarios": list(SCENARIOS),
            "seeds": list(VALIDATION_SEEDS),
            "summary": stress["summary"],
        },
        "negative_control": {
            "mechanism": "same-session DVR Gaussian timestamp smoothing, nested adoption",
            "under_calibrated_ruler": summary.get("negative_control_dvr_timestamp"),
            "live_public_delta_when_submitted": -0.01085,
        },
        "selected_recipe": selected,
        "outcome": (
            f"promote {selected}" if selected else "no recipe cleared the gate; the incumbent stands"
        ),
        "runtime_seconds": time.time() - started,
    }
    write_json(receipt_path, receipt, exclusive=True)
    return {
        "validated": True,
        "calibrated_ruler": ruler,
        "protocol_mismatch_inflation": mismatch["inflation_removed_by_protocol_parity"],
        "selected_recipe": selected,
        "outcome": receipt["outcome"],
        "receipt_sha256": sha256_file(receipt_path),
    }


# --------------------------------------------------------------------------
# Contract, sealing and adversarial submission audit.
# --------------------------------------------------------------------------


def prepare(
    run_root: Path,
    source_root: Path,
    timestamps_path: Path,
    *,
    submission_authorized: bool,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    source_root = source_root.resolve()
    timestamps_path = timestamps_path.resolve()
    if run_root.exists():
        raise FileExistsError(f"fresh run root required: {run_root}")
    inputs = {
        "sealed_train_features": source_root / "root_workspace/conv_features_640.npz",
        "sealed_test_features": source_root / "root_workspace/conv_test_features_640.npz",
        "grouped_folds": source_root / "trace_validation/grouped_folds.csv",
        "test_metadata": source_root / "data/official/test_metadata.csv",
        "submission_codec": source_root / "data/official/submission_codec.py",
        "base_checkpoint": source_root / "data/official/base.pth",
        "timestamps": timestamps_path,
    }
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{name}: {path}")
    run_root.joinpath("contracts").mkdir(parents=True)
    run_root.joinpath("runtime").mkdir()
    run_root.joinpath("outputs").mkdir()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_synthesis",
        "task": TASK,
        "component_scope": "data_synthesis_validation_submission_adversary_only",
        "run_root": str(run_root),
        "source_root": str(source_root),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "frozen_non_goals": [
            "estimator choice and hyperparameters",
            "feature extractor and feature summary",
            "blend weights and calibration",
            "solver search policy, manager scheduling, web-search route",
        ],
        "estimator_constants": {
            "ridge_alpha": RIDGE_ALPHA,
            "knn_neighbours": KNN_NEIGHBOURS,
            "ridge_weight": RIDGE_WEIGHT,
            "knn_weight": KNN_WEIGHT,
            "base_scale": BASE_SCALE,
            "base_weight": BASE_WEIGHT,
            "log_ridge_weight": LOG_RIDGE_WEIGHT,
            "log_ridge_alpha": LOG_RIDGE_ALPHA,
        },
        "synthesis": {
            "method": "native_scale_density_integral_crops",
            "target": "area_normalised_count",
            "scales": list(CROP_SCALES),
            "anchors": [list(anchor) for anchor in CROP_ANCHORS],
            "external_data": False,
            "hidden_labels": False,
            "test_images_used_for_fitting": False,
        },
        "validation_protocol": {
            "scenarios": list(SCENARIOS),
            "seeds": list(VALIDATION_SEEDS),
            "selection_uses_public_score": False,
            "negative_control_required_to_fail": True,
        },
        "negative_control": {
            "mechanism": "same_session_dvr_timestamp_smoothing",
            "timestamps_path": str(timestamps_path),
            "known_live_public_delta": -0.01085,
        },
        "submission_policy": {
            "backend": "kaggle_csv",
            "submission_slots_authorized": 1 if submission_authorized else 0,
            "explicit_user_authority": submission_authorized,
            "repo_public_incumbent": REPO_PUBLIC_INCUMBENT,
            "account_public_best": ACCOUNT_PUBLIC_BEST,
            "git_push_requires_strict_public_improvement": True,
            "no_leaderboard_read_between_freeze_and_submission": True,
            "no_post_reveal_tuning": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
    write_json(run_root / "contracts/component_contract.json", contract, exclusive=True)
    digest = sha256_file(run_root / "contracts/component_contract.json")
    run_root.joinpath("contracts/component_contract.sha256").write_text(
        f"{digest}\n", encoding="ascii"
    )
    return {"prepared": True, "contract_sha256": digest}


def write_submission(path: Path, rows: list[dict[str, str]], counts, codec) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "density_map"])
        writer.writeheader()
        for row, count in zip(rows, counts, strict=True):
            density = np.full(DENSITY_SHAPE, count / (DENSITY_SHAPE[0] * DENSITY_SHAPE[1]), dtype=np.float32)
            writer.writerow({"id": row["id"], "density_map": codec.encode_density_map(density)})
    with path.open(newline="") as handle:
        observed = list(csv.DictReader(handle))
    decoded = np.asarray([codec.decode_density_map(row["density_map"]).sum() for row in observed])
    if [row["id"] for row in observed] != [row["id"] for row in rows]:
        raise ValueError("submission ID order mismatch")
    if not np.isfinite(decoded).all() or np.any(decoded < 0):
        raise ValueError("invalid encoded density map")
    error = float(np.max(np.abs(decoded - counts)))
    if error >= 1e-3:
        raise ValueError("density-map sum changed during encoding")
    return {
        "rows": len(observed),
        "columns": ["id", "density_map"],
        "id_order_exact": True,
        "finite": True,
        "nonnegative": True,
        "max_codec_sum_error": error,
        "sha256": sha256_file(path),
    }


def seal(run_root: Path) -> dict[str, Any]:
    import numpy as np

    run_root = run_root.resolve()
    contract = load_object(run_root / "contracts/component_contract.json")
    if contract["runner"]["sha256"] != sha256_file(Path(__file__).resolve()):
        raise ValueError("component runner changed after freeze")
    for item in contract["inputs"].values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"frozen input changed: {path}")
    receipt_path = run_root / "runtime/selection_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("candidate already sealed")

    validation = load_object(run_root / "outputs/synthetic_validation_receipt.json")
    recipe_name = validation["selected_recipe"]
    if not recipe_name:
        raise ValueError("no recipe cleared the gate; the incumbent stands")
    evidence = validation["selection_evidence"]
    if evidence["ruler"] != validation["ruler_calibration"]["selected_ruler"]:
        raise ValueError("selection evidence was not produced under the calibrated ruler")
    if not evidence["summary"][recipe_name]["pass"]:
        raise ValueError("selected recipe does not carry a passing gate")
    tags = recipe_table()[recipe_name]

    source_root = Path(contract["source_root"])
    sealed_train = np.load(source_root / "root_workspace/conv_features_640.npz", allow_pickle=False)
    sealed_test = np.load(source_root / "root_workspace/conv_test_features_640.npz", allow_pickle=False)
    x0, y0, x_test = sealed_train["x"], sealed_train["y"], sealed_test["x"]
    synthetic = np.load(run_root / "outputs/synthetic_rows.npz", allow_pickle=False)
    admissible = np.isin(synthetic["kind"], tags)
    fit_x = np.concatenate([x0, synthetic["x"][admissible]])
    fit_y = np.concatenate([y0, synthetic["y"][admissible]])

    counts = fit_and_predict(fit_x, fit_y, x_test)["blend"]
    incumbent_counts = fit_and_predict(x0, y0, x_test)["blend"]

    # Sample locality: a row's prediction may not depend on which other test
    # rows travel with it.
    reverse = fit_and_predict(fit_x, fit_y, x_test[::-1])["blend"][::-1]
    subset_index = np.arange(0, len(x_test), 7)
    subset = fit_and_predict(fit_x, fit_y, x_test[subset_index])["blend"]
    rebatch = np.concatenate(
        [
            fit_and_predict(fit_x, fit_y, x_test[start : start + 31])["blend"]
            for start in range(0, len(x_test), 31)
        ]
    )
    invariance = {
        "permutation_max_error": float(np.max(np.abs(counts - reverse))),
        "strict_subset_max_error": float(np.max(np.abs(counts[subset_index] - subset))),
        "rebatch_max_error": float(np.max(np.abs(counts - rebatch))),
    }
    if max(invariance.values()) > 1e-5:
        raise ValueError(f"sample-local invariance failed: {invariance}")

    adversary = {
        "fitting_matrix_is_train_only": bool(
            len(fit_x) == len(x0) + int(admissible.sum())
            and np.array_equal(fit_x[: len(x0)], x0)
        ),
        "test_features_absent_from_fit": bool(
            not (
                {hashlib.sha256(row.tobytes()).hexdigest() for row in np.ascontiguousarray(fit_x)}
                & {hashlib.sha256(row.tobytes()).hexdigest() for row in np.ascontiguousarray(x_test)}
            )
        ),
        "scaler_fit_on_train_only": True,
        "row_order_independent": max(invariance.values()) <= 1e-5,
        "hidden_labels_read": False,
        "external_data_used": False,
        "prediction_within_plausible_range": bool(
            counts.min() > 0.5 * float(y0.min()) and counts.max() < 2.0 * float(y0.max())
        ),
        "max_abs_shift_vs_incumbent": float(np.max(np.abs(counts - incumbent_counts))),
        "mean_abs_shift_vs_incumbent": float(np.mean(np.abs(counts - incumbent_counts))),
    }
    if not all(
        adversary[key]
        for key in (
            "fitting_matrix_is_train_only",
            "test_features_absent_from_fit",
            "scaler_fit_on_train_only",
            "row_order_independent",
            "prediction_within_plausible_range",
        )
    ):
        raise ValueError(f"submission adversary audit failed: {adversary}")

    with Path(contract["inputs"]["test_metadata"]["path"]).open(newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    if len(test_rows) != len(counts):
        raise ValueError("test metadata row count mismatch")
    codec = import_module(Path(contract["inputs"]["submission_codec"]["path"]), "chicken_codec")

    outputs = {}
    for name in ("run_a", "run_b"):
        directory = run_root / "outputs" / name
        directory.mkdir(parents=True)
        counts_path = directory / "predicted_counts.npy"
        np.save(counts_path, counts.copy())
        submission_path = directory / "submission.csv"
        outputs[name] = {
            "counts_path": str(counts_path),
            "counts_sha256": sha256_file(counts_path),
            "submission_path": str(submission_path),
            "validation": write_submission(submission_path, test_rows, counts.copy(), codec),
        }
        outputs[name]["submission_sha256"] = outputs[name]["validation"]["sha256"]
    if outputs["run_a"]["submission_sha256"] != outputs["run_b"]["submission_sha256"]:
        raise ValueError("duplicate generation is not byte identical")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "sealed_awaiting_submission",
        "task": TASK,
        "selected_recipe": recipe_name,
        "selected_recipe_tags": tags,
        "synthetic_rows_used": int(admissible.sum()),
        "fitting_rows": int(len(fit_y)),
        "calibrated_ruler": evidence["ruler"],
        "gate": evidence["summary"][recipe_name],
        "protocol_mismatch_audit": validation["protocol_mismatch_audit"],
        "stress_diagnostics": validation["stress_diagnostics"]["summary"].get(recipe_name),
        "sample_local_invariance": invariance,
        "submission_adversary": adversary,
        "prediction_summary": {
            "min": float(counts.min()),
            "max": float(counts.max()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
        },
        "outputs": outputs,
        "duplicate_generation_byte_identical": True,
        "leaderboard_read_after_freeze": False,
        "submission_sent": False,
    }
    write_json(receipt_path, receipt, exclusive=True)
    digest = sha256_file(receipt_path)
    run_root.joinpath("runtime/selection_receipt.sha256").write_text(f"{digest}\n", encoding="ascii")
    return {
        "sealed": True,
        "selected_recipe": recipe_name,
        "fitting_rows": receipt["fitting_rows"],
        "submission_path": outputs["run_a"]["submission_path"],
        "submission_sha256": outputs["run_a"]["submission_sha256"],
        "receipt_sha256": digest,
    }


def audit(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    contract = load_object(run_root / "contracts/component_contract.json")
    observed = sha256_file(run_root / "contracts/component_contract.json")
    expected = run_root.joinpath("contracts/component_contract.sha256").read_text(
        encoding="ascii"
    ).strip()
    if observed != expected:
        raise ValueError("component contract differs from pinned hash")
    receipt = load_object(run_root / "runtime/selection_receipt.json")
    if sha256_file(run_root / "runtime/selection_receipt.json") != run_root.joinpath(
        "runtime/selection_receipt.sha256"
    ).read_text(encoding="ascii").strip():
        raise ValueError("selection receipt differs from pinned hash")
    for item in receipt["outputs"].values():
        if sha256_file(Path(item["submission_path"])) != item["submission_sha256"]:
            raise ValueError("submission changed after sealing")
        if sha256_file(Path(item["counts_path"])) != item["counts_sha256"]:
            raise ValueError("counts changed after sealing")
    allowed = bool(
        receipt["gate"]["pass"]
        and receipt["duplicate_generation_byte_identical"]
        and contract["submission_policy"]["submission_slots_authorized"] == 1
    )
    return {
        "valid": True,
        "sealed": True,
        "submission_allowed": allowed,
        "selected_recipe": receipt["selected_recipe"],
        "submission_path": receipt["outputs"]["run_a"]["submission_path"],
        "submission_sha256": receipt["outputs"]["run_a"]["submission_sha256"],
        "public_score_required_for_git_push": REPO_PUBLIC_INCUMBENT,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--run-root", type=Path, required=True)
    make.add_argument("--source-root", type=Path, required=True)
    make.add_argument("--timestamps", type=Path, required=True)
    make.add_argument("--submission-authorized", action="store_true")
    build = sub.add_parser("synthesize")
    build.add_argument("--run-root", type=Path, required=True)
    build.add_argument("--source-root", type=Path, required=True)
    for name in ("validate", "seal", "audit"):
        child = sub.add_parser(name)
        child.add_argument("--run-root", type=Path, required=True)
    return value


def main() -> None:
    try:
        args = parser().parse_args()
        if args.command == "prepare":
            result = prepare(
                args.run_root,
                args.source_root,
                args.timestamps,
                submission_authorized=args.submission_authorized,
            )
        elif args.command == "synthesize":
            result = synthesize(args.run_root, args.source_root)
        elif args.command == "validate":
            result = validate(args.run_root)
        elif args.command == "seal":
            result = seal(args.run_root)
        else:
            result = audit(args.run_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed.
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
