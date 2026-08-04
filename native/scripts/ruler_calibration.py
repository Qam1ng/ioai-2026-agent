#!/usr/bin/env python3
"""Ask whether the holdout being scored on resembles the live test set.

A validation scheme is a claim about the test set: that held-out rows sit at
about the same distance from their fitting set as the real test rows sit from
the whole labelled set.  That claim is measurable, from the feature matrices
alone, before any label is touched - and on chicken counting it was false by a
wide margin.  Candidates there were ranked on grouped five-fold, whose held-out
frames sit at median distance 62.62 from their fitting set while the live test
frames sit at 52.71, and the run that trusted it paid 0.011 of public score.

This is the task-agnostic half of `chicken_data_synthesis.py`.  It takes two
feature matrices and reports which ruler the data supports.

    python -m native.scripts.ruler_calibration \\
        --train-features x_train.npz --test-features x_test.npz \\
        --groups blocks.npy --in-use grouped_5fold --out calibration.json

It answers one question and refuses to answer others: it says nothing about
whether a candidate is good, only about whether the ruler used to judge it
describes the test set the submission will actually meet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SEEDS = (20260803, 20260817, 20260831, 20260914, 20260928)
DETERMINISTIC = ("leave_one_out", "contiguous_5fold", "grouped_5fold")
# Beyond this the ruler is describing a different problem from the live one.
GAP_WARNING_RATIO = 0.05


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())


def load_matrix(path: Path, key: str = "x"):
    import numpy as np

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"{path}: expected key {key!r}, found {data.files}")
            return np.asarray(data[key], dtype=np.float64)
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)


def load_groups(path: Path, rows: int):
    import numpy as np

    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            column = next(
                (name for name in ("group", "acquisition_block", "fold") if name in reader.fieldnames),
                None,
            )
            if column is None:
                raise ValueError(f"{path}: no group/acquisition_block/fold column")
            handle.seek(0)
            values = [int(row[column]) for row in csv.DictReader(handle)]
    else:
        values = np.load(path, allow_pickle=False).astype(int).tolist()
    if len(values) != rows:
        raise ValueError(f"{path}: {len(values)} group labels for {rows} rows")
    return np.asarray(values)


def nearest_distances(query, reference, *, block: int = 256):
    """Smallest Euclidean distance from each query row to any reference row."""
    import numpy as np

    reference_sq = np.einsum("ij,ij->i", reference, reference)
    output = np.empty(len(query), dtype=float)
    for start in range(0, len(query), block):
        chunk = query[start : start + block]
        cross = chunk @ reference.T
        squared = (
            np.einsum("ij,ij->i", chunk, chunk)[:, None] - 2.0 * cross + reference_sq[None, :]
        )
        output[start : start + len(chunk)] = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    return output


def self_nearest_distances(matrix, keep, query):
    """Distance from each query row to its nearest row inside `keep`."""
    import numpy as np

    return nearest_distances(matrix[query], matrix[keep])


def holdouts(ruler: str, seed: int, rows: int, groups) -> list[Any]:
    import numpy as np

    index = np.arange(rows)
    rng = np.random.default_rng(seed)
    if ruler == "leave_one_out":
        return [np.asarray([position]) for position in index]
    if ruler == "kfold_5":
        order = rng.permutation(index)
        return [order[part::5] for part in range(5)]
    if ruler == "contiguous_5fold":
        # sklearn's KFold(shuffle=False): five contiguous blocks of the rows as
        # they arrive. Harmless when row order is arbitrary, a group holdout in
        # disguise when it is not.
        return list(np.array_split(index, 5))
    if ruler.startswith("interleaved_"):
        parts = {"interleaved_10pct": 10, "interleaved_25pct": 4, "interleaved_50pct": 2}[ruler]
        order = rng.permutation(index)
        return [order[part::parts] for part in range(parts)]
    if ruler == "as_used":
        raise ValueError("as_used is driven by --folds, not reconstructed")
    if groups is None:
        raise ValueError(f"{ruler} needs --groups")
    unique = np.unique(groups)
    if ruler == "grouped_5fold":
        assignment = {group: position % 5 for position, group in enumerate(unique)}
        folds = np.asarray([assignment[group] for group in groups])
        return [np.flatnonzero(folds == part) for part in range(5)]
    if ruler == "group_holdout_3":
        shuffled = rng.permutation(unique)
        return [
            np.flatnonzero(np.isin(groups, shuffled[start : start + 3]))
            for start in range(0, len(shuffled), 3)
        ]
    raise ValueError(f"unknown ruler: {ruler}")


def calibrate(
    train_features: Path,
    test_features: Path,
    groups_path: Path | None,
    in_use: str | None,
    *,
    folds_path: Path | None = None,
    standardize: bool = True,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    x_train = load_matrix(train_features)
    x_test = load_matrix(test_features)
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"feature width mismatch: train {x_train.shape[1]}, test {x_test.shape[1]}"
        )
    groups = load_groups(groups_path, len(x_train)) if groups_path else None
    folds = load_groups(folds_path, len(x_train)) if folds_path else None

    # Measure in whatever metric the model generalises in. A kNN on raw
    # Euclidean distance does not see the standardised space, so scaling the
    # features here would answer a question nobody asked.
    if standardize:
        scaler = StandardScaler().fit(x_train)
        train_scaled = scaler.transform(x_train)
        test_scaled = scaler.transform(x_test)
    else:
        train_scaled = x_train
        test_scaled = x_test

    live = nearest_distances(test_scaled, train_scaled)
    live_median = float(np.median(live))

    names = [
        "leave_one_out",
        "interleaved_10pct",
        "interleaved_25pct",
        "kfold_5",
        "contiguous_5fold",
    ]
    if groups is not None:
        names += ["grouped_5fold", "group_holdout_3"]

    as_used = (
        [np.flatnonzero(folds == part) for part in np.unique(folds)]
        if folds is not None
        else None
    )
    if as_used:
        names = ["as_used", *names]

    measured: dict[str, Any] = {}
    for ruler in names:
        samples: list[float] = []
        for seed in SEEDS:
            splits = as_used if ruler == "as_used" else holdouts(ruler, seed, len(x_train), groups)
            for held in splits:
                if len(held) == 0 or len(held) >= len(x_train):
                    continue
                keep = np.setdiff1d(np.arange(len(x_train)), held)
                samples.extend(self_nearest_distances(train_scaled, keep, held).tolist())
            if ruler in DETERMINISTIC or ruler == "as_used":
                break
        array = np.asarray(samples)
        median = float(np.median(array))
        measured[ruler] = {
            "median_query_to_fit_distance": median,
            "p25": float(np.percentile(array, 25)),
            "p75": float(np.percentile(array, 75)),
            "gap_vs_live": float(abs(median - live_median)),
            "relative_gap_vs_live": float(abs(median - live_median) / live_median),
        }

    closest = min(measured, key=lambda name: measured[name]["gap_vs_live"])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reads_labels": False,
        "feature_space": "standardised" if standardize else "raw",
        "inputs": {
            "train_features": {"path": str(train_features), "sha256": sha256_file(train_features)},
            "test_features": {"path": str(test_features), "sha256": sha256_file(test_features)},
            "groups": (
                {"path": str(groups_path), "sha256": sha256_file(groups_path)}
                if groups_path
                else None
            ),
            "folds_as_used": (
                {"path": str(folds_path), "sha256": sha256_file(folds_path)}
                if folds_path
                else None
            ),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "feature_dimension": int(x_train.shape[1]),
        },
        "live_test_geometry": {
            "median_test_to_train_distance": live_median,
            "p25": float(np.percentile(live, 25)),
            "p75": float(np.percentile(live, 75)),
        },
        "rulers": measured,
        "closest_ruler": closest,
        "selection_rule": "smallest absolute gap between median query-to-fit and median test-to-train distance",
    }
    if in_use:
        if in_use not in measured:
            raise ValueError(f"--in-use {in_use} was not measured; choose from {sorted(measured)}")
        relative = measured[in_use]["relative_gap_vs_live"]
        report["ruler_in_use"] = {
            "name": in_use,
            "relative_gap_vs_live": relative,
            "closest_available": closest,
            "closest_relative_gap": measured[closest]["relative_gap_vs_live"],
            "is_closest": in_use == closest,
            "warn": relative > GAP_WARNING_RATIO,
            "verdict": (
                f"{in_use} describes the live test geometry to within "
                f"{relative:.1%}; no ruler change indicated"
                if relative <= GAP_WARNING_RATIO
                else (
                    f"{in_use} is {relative:.1%} away from the live geometry while "
                    f"{closest} is {measured[closest]['relative_gap_vs_live']:.1%} away; "
                    "rank candidates on the closer one and re-check any banked margin"
                )
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--groups", type=Path)
    parser.add_argument(
        "--folds",
        type=Path,
        help="the fold assignment actually used, measured verbatim as 'as_used'",
    )
    parser.add_argument("--in-use", help="the ruler currently used to rank candidates")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="measure in the unstandardised feature space, for models that do",
    )
    parser.add_argument("--out", type=Path)
    try:
        args = parser.parse_args()
        report = calibrate(
            args.train_features,
            args.test_features,
            args.groups,
            args.in_use,
            folds_path=args.folds,
            standardize=not args.raw,
        )
        if args.out:
            write_json(args.out, report)
            report["written_to"] = str(args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed.
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
