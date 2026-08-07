from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, canonical, read_json


def _rank(values: list[float]) -> np.ndarray:
    """Average ranks with deterministic tie handling; scipy is unnecessary here."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    a, b = _rank(left), _rank(right)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _ridge_fit(x: np.ndarray, y: np.ndarray, strength: float) -> dict[str, Any]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), normalized])
    penalty = np.eye(design.shape[1]) * strength
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    normalized = (x - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(x)), normalized])
    return design @ model["coefficients"]


def _features(evaluation: dict, fold_count: int) -> list[float] | None:
    if evaluation.get("status") != "ok":
        return None
    per_fold = evaluation.get("per_fold")
    if not isinstance(per_fold, list) or len(per_fold) != fold_count:
        return None
    values = [
        evaluation.get("mean"), evaluation.get("std"), evaluation.get("pooled"),
        *per_fold,
    ]
    try:
        output = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    return output if all(math.isfinite(value) for value in output) else None


class FeedbackCalibrator:
    """Versioned Public-LB calibration layered on top of the immutable E0 ruler."""

    def __init__(
        self, root: Path, *, min_feedback: int = 4,
        ridge_strength: float = 10.0, min_rank_gain: float = 0.05,
        max_blend_weight: float = 0.50,
    ):
        self.root = Path(root)
        self.latest_path = self.root / "latest.json"
        self.history = self.root / "history"
        self.history.mkdir(parents=True, exist_ok=True)
        self.min_feedback = min_feedback
        self.ridge_strength = ridge_strength
        self.min_rank_gain = min_rank_gain
        self.max_blend_weight = max_blend_weight

    def update(self, records: list[dict], submissions: list[dict]) -> dict:
        by_id = {record["candidate_id"]: record for record in records}
        scored_rows = [
            row for row in submissions
            if row.get("leaderboard_score") is not None
            and row.get("candidate_id") in by_id
        ]
        fold_lengths = {
            len(by_id[row["candidate_id"]].get("evaluation", {}).get("per_fold", []))
            for row in scored_rows
            if by_id[row["candidate_id"]].get("evaluation", {}).get("status") == "ok"
        }
        fold_count = next(iter(fold_lengths)) if len(fold_lengths) == 1 else 0
        points: list[dict] = []
        if fold_count:
            for row in scored_rows:
                record = by_id[row["candidate_id"]]
                feature = _features(record.get("evaluation", {}), fold_count)
                if feature is None:
                    continue
                try:
                    public_score = float(row["leaderboard_score"])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(public_score):
                    continue
                points.append({
                    "candidate_id": row["candidate_id"],
                    "source_lane": row.get("source_lane"),
                    "local_score": float(record["evaluation"]["mean"]),
                    "leaderboard_score": public_score,
                    "features": feature,
                })

        active = False
        baseline_correlation = calibrated_correlation = 0.0
        residual_rmse = None
        blend_weight = 0.0
        model: dict[str, Any] | None = None
        if len(points) >= self.min_feedback:
            x = np.asarray([point["features"] for point in points], dtype=float)
            y = np.asarray([point["leaderboard_score"] for point in points], dtype=float)
            local = [point["local_score"] for point in points]
            loo: list[float] = []
            for index in range(len(points)):
                keep = np.arange(len(points)) != index
                fitted = _ridge_fit(x[keep], y[keep], self.ridge_strength)
                loo.append(float(_predict(fitted, x[index:index + 1])[0]))
            baseline_correlation = _spearman(local, y.tolist())
            calibrated_correlation = _spearman(loo, y.tolist())
            residual_rmse = float(np.sqrt(np.mean((np.asarray(loo) - y) ** 2)))
            active = (
                calibrated_correlation
                >= baseline_correlation + self.min_rank_gain
                and calibrated_correlation > 0.0
            )
            if active:
                gain = calibrated_correlation - baseline_correlation
                blend_weight = min(self.max_blend_weight, max(0.10, gain / 2.0))
            model = _ridge_fit(x, y, self.ridge_strength)

        # A negative baseline correlation is not weak signal, it is a wrong
        # sign: the local ruler ranks candidates backwards (dryrun8, 5 of 6
        # pairs inverted). Ridge-correcting it would still preserve the wrong
        # order, so say so loudly and let the broker stop ranking by it.
        # Fitting a ridge correction needs min_feedback points; noticing that
        # the ruler ranks backwards only needs the sign of a rank correlation,
        # and waiting for the fitting quorum meant dryrun8 submitted four
        # candidates under a ruler already known to be inverted. Three points
        # is the floor where Spearman is not ±1 by construction.
        inversion_min = min(3, self.min_feedback)
        if len(points) >= inversion_min:
            local_all = [p["local_score"] for p in points]
            lb_all = [p["leaderboard_score"] for p in points]
            baseline_correlation = _spearman(local_all, lb_all)
        rank_inverted = (
            len(points) >= inversion_min and baseline_correlation < 0.0
        )

        candidate_predictions: dict[str, dict] = {}
        if model is not None and fold_count:
            observed = [point["leaderboard_score"] for point in points]
            if observed:
                margin = max(float(residual_rmse or 0.0), 1e-6)
                low, high = min(observed) - margin, max(observed) + margin
            else:
                low, high = 0.0, 0.0
            for record in records:
                feature = _features(record.get("evaluation", {}), fold_count)
                if feature is None:
                    continue
                predicted = float(_predict(
                    model, np.asarray([feature], dtype=float)
                )[0])
                predicted = min(high, max(low, predicted)) if (high >= low) else float(predicted)
                local_score = float(record["evaluation"]["mean"])
                candidate_predictions[record["candidate_id"]] = {
                    "predicted_public_score": round(predicted, 8),
                    "uncertainty_rmse": round(float(residual_rmse or 0.0), 8),
                    "adjusted_score": round(
                        (1.0 - blend_weight) * local_score
                        + blend_weight * predicted
                        if active
                        else predicted,
                        8,
                    ),
                }
        core = {
            "schema_version": 1,
            "base_contract": "immutable_E0_metric_and_folds",
            "active": active,
            "rank_inverted": rank_inverted,
            "reason": (
                "leave_one_out_rank_gain_passed" if active else
                "insufficient_or_non_improving_feedback"
            ),
            "feedback_count": len(points),
            "min_feedback": self.min_feedback,
            "feature_names": (
                ["mean", "std", "pooled"]
                + [f"fold_{index}" for index in range(fold_count)]
            ),
            "ridge_strength": self.ridge_strength,
            "baseline_spearman": round(baseline_correlation, 8),
            "calibrated_loo_spearman": round(calibrated_correlation, 8),
            "residual_rmse": (
                round(float(residual_rmse), 8) if residual_rmse is not None else None
            ),
            "blend_weight": round(blend_weight, 8),
            "points": points,
            "candidate_predictions": candidate_predictions,
        }
        version = hashlib.sha256(canonical(core)).hexdigest()
        output = {**core, "version": version, "updated_at": time.time()}
        previous = read_json(self.latest_path, {})
        if previous.get("version") != version:
            atomic_json(self.history / f"{version[:16]}.json", output)
            atomic_json(self.latest_path, output)
        return output
