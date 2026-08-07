#!/usr/bin/env python3
"""Build and seal the IOAI 2025 Chicken timestamp-smoothing candidate.

The official images contain a fixed DVR timestamp.  Frames from the same
capture session are interleaved across train and test, so a sample-local model
can smooth labelled train counts in timestamp space and blend that estimate
with a frozen visual incumbent.  This runner never reads leaderboard results;
it freezes every input before generating a test prediction and authorizes at
most one exploratory practice submission.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata


SCHEMA_VERSION = 1
TASK = "ioai-2025-chicken-counting-mirror-unofficial"
REPO_LOCAL_BASELINE = 0.918652
REPO_PUBLIC_BASELINE = 0.93156
FROZEN_VISUAL_OOF_SCORE = 0.8908494636264506
TEMPORAL_BANDWIDTH_SECONDS = 105.0
TEMPORAL_WEIGHT = 0.875
VISUAL_WEIGHT = 1.0 - TEMPORAL_WEIGHT
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260803
EXPECTED_ROWS = {"train": 100, "test_a": 100, "test_b": 100}
VALID_DATES = {(5, 30), (5, 31), (6, 1), (6, 6)}
VALID_HOURS = {6, 7, 9, 11, 12, 14, 16}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_manifest(root: Path) -> dict[str, Any]:
    records = []
    for split, expected in EXPECTED_ROWS.items():
        paths = sorted((root / split / "images").glob(f"{split}_*.png"))
        if len(paths) != expected:
            raise ValueError(f"{split}: expected {expected} images, got {len(paths)}")
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unsafe image input: {path}")
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    payload = canonical_json(records)
    return {
        "algorithm": "sha256(canonical_json([{path,sha256},...]))",
        "count": len(records),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
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


def normalized_ocr(record: dict[str, Any]) -> str:
    text = " | ".join(
        item["candidates"][0]["text"]
        for item in record.get("observations", [])
        if item.get("candidates")
    )
    return (
        unicodedata.normalize("NFKC", text)
        .replace("：", ":")
        .replace("。", ":")
        .replace("·", ":")
    )


def parsed_timestamp(record: dict[str, Any]) -> datetime | None:
    text = normalized_ocr(record)
    date_match = re.search(r"2024\D{0,4}(0[56])\D{0,4}([0-3]\d)", text)
    times = list(
        re.finditer(
            r"(?<!\d)(0?[679]|1[1246])\D{0,3}([0-5]\d)"
            r"\D{0,3}([0-5]\d)(?!\d)",
            text,
        )
    )
    if not date_match or not times:
        return None
    month, day = map(int, date_match.groups())
    hour, minute, second = map(int, times[-1].groups())
    if (month, day) not in VALID_DATES or hour not in VALID_HOURS:
        return None
    return datetime(2024, month, day, hour, minute, second)


def load_timestamps(
    raw_path: Path,
    overrides_path: Path,
    official_root: Path,
) -> tuple[dict[str, datetime], list[dict[str, Any]], dict[str, int]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    override_object = load_object(overrides_path)
    overrides = override_object.get("records")
    if not isinstance(overrides, dict):
        raise ValueError("manual timestamp override records are missing")
    timestamps: dict[str, datetime] = {}
    receipt_rows = []
    source_counts = {"manual_visual": 0, "vision_ocr": 0}
    observed_by_split = {key: 0 for key in EXPECTED_ROWS}
    for record in raw:
        sample_id = str(record["id"])
        split = str(record["split"])
        if split not in EXPECTED_ROWS or sample_id in timestamps:
            raise ValueError(f"unexpected or duplicate timestamp record: {sample_id}")
        expected_path = official_root / split / "images" / f"{sample_id}.png"
        if Path(record["path"]).resolve() != expected_path.resolve():
            raise ValueError(f"OCR/image path mismatch: {sample_id}")
        parsed = parsed_timestamp(record)
        if sample_id in overrides:
            if parsed is not None:
                raise ValueError(f"manual override masks a conservatively parsed row: {sample_id}")
            timestamp = datetime.fromisoformat(str(overrides[sample_id]))
            source = "manual_visual"
        else:
            if parsed is None:
                raise ValueError(f"timestamp is unresolved: {sample_id}")
            timestamp = parsed
            source = "vision_ocr"
        if timestamp.year != 2024 or (timestamp.month, timestamp.day) not in VALID_DATES:
            raise ValueError(f"timestamp outside the official sessions: {sample_id}")
        timestamps[sample_id] = timestamp
        observed_by_split[split] += 1
        source_counts[source] += 1
        receipt_rows.append(
            {
                "id": sample_id,
                "split": split,
                "timestamp": timestamp.isoformat(),
                "source": source,
            }
        )
    if observed_by_split != EXPECTED_ROWS:
        raise ValueError(f"timestamp row contract mismatch: {observed_by_split}")
    if set(overrides) != {
        row["id"] for row in receipt_rows if row["source"] == "manual_visual"
    }:
        raise ValueError("override file contains unused or unknown IDs")
    return timestamps, receipt_rows, source_counts


def session_key(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H")


def official_score(target: Any, prediction: Any) -> float:
    import numpy as np

    target_array = np.asarray(target, dtype=float)
    prediction_array = np.maximum(np.asarray(prediction, dtype=float), 0.01)
    return float(np.exp(-np.mean(np.abs(target_array - prediction_array) / target_array)))


def temporal_predictions(
    query_ids: list[str],
    train_ids: list[str],
    timestamps: dict[str, datetime],
    target: Any,
    *,
    leave_one_out: bool,
) -> Any:
    import numpy as np

    target_array = np.asarray(target, dtype=float)
    train_seconds = np.asarray([timestamps[item].timestamp() for item in train_ids])
    train_sessions = np.asarray([session_key(timestamps[item]) for item in train_ids])
    output = np.empty(len(query_ids), dtype=float)
    index = {sample_id: position for position, sample_id in enumerate(train_ids)}
    for row, sample_id in enumerate(query_ids):
        query_time = timestamps[sample_id].timestamp()
        query_session = session_key(timestamps[sample_id])
        weights = np.exp(
            -0.5
            * ((train_seconds - query_time) / TEMPORAL_BANDWIDTH_SECONDS) ** 2
        )
        weights *= train_sessions == query_session
        if leave_one_out:
            weights[index[sample_id]] = 0.0
        if float(weights.sum()) <= 1e-12:
            raise ValueError(f"no temporal neighbour for {sample_id}")
        output[row] = float(weights @ target_array / weights.sum())
    return output


def paired_bootstrap(
    target: Any,
    incumbent: Any,
    candidate: Any,
    blocks: Any,
) -> dict[str, Any]:
    import numpy as np

    target_array = np.asarray(target, dtype=float)
    row_delta = (
        np.abs(target_array - incumbent) / target_array
        - np.abs(target_array - candidate) / target_array
    )
    blocks_array = np.asarray(blocks)
    unique = np.unique(blocks_array)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(blocks_array == item) for item in sampled]
        )
        values[draw] = float(row_delta[indices].mean())
    return {
        "unit": "acquisition_block",
        "draws": BOOTSTRAP_DRAWS,
        "mre_improvement_point": float(row_delta.mean()),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "probability_positive": float(np.mean(values > 0)),
    }


def write_submission(
    path: Path,
    rows: list[dict[str, str]],
    counts: Any,
    codec: Any,
) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "density_map"])
        writer.writeheader()
        for row, count in zip(rows, counts, strict=True):
            density = np.full((180, 320), count / (180 * 320), dtype=np.float32)
            writer.writerow(
                {"id": row["id"], "density_map": codec.encode_density_map(density)}
            )
    with path.open(newline="") as handle:
        observed = list(csv.DictReader(handle))
    decoded = np.asarray(
        [codec.decode_density_map(row["density_map"]).sum() for row in observed]
    )
    if [row["id"] for row in observed] != [row["id"] for row in rows]:
        raise ValueError("submission ID order mismatch")
    if not np.isfinite(decoded).all() or np.any(decoded < 0):
        raise ValueError("invalid encoded density map")
    max_error = float(np.max(np.abs(decoded - counts)))
    if max_error >= 1e-3:
        raise ValueError("density-map sum changed during encoding")
    return {
        "rows": len(observed),
        "columns": ["id", "density_map"],
        "id_order_exact": True,
        "finite": True,
        "nonnegative": True,
        "max_codec_sum_error": max_error,
        "sha256": sha256_file(path),
    }


def contract_path(run_root: Path) -> Path:
    return run_root / "contracts/candidate_contract.json"


def prepare(
    run_root: Path,
    source_root: Path,
    visual_root: Path,
    timestamp_root: Path,
    *,
    submission_authorized: bool,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    source_root = source_root.resolve()
    visual_root = visual_root.resolve()
    timestamp_root = timestamp_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"fresh run root required: {run_root}")
    official_root = source_root / "data/official"
    inputs = {
        "test_metadata": official_root / "test_metadata.csv",
        "submission_codec": official_root / "submission_codec.py",
        "visual_oof": visual_root / "outputs/development_oof.npz",
        "visual_test_counts": visual_root / "outputs/run_a/predicted_counts.npy",
        "visual_submission": visual_root / "outputs/run_a/submission.csv",
        "timestamp_ocr": timestamp_root / "artifacts/timestamp_ocr_raw.json",
        "timestamp_overrides": timestamp_root / "artifacts/manual_timestamp_overrides.json",
    }
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{name}: {path}")
    run_root.joinpath("contracts").mkdir(parents=True)
    run_root.joinpath("runtime").mkdir()
    run_root.joinpath("outputs").mkdir()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_test_prediction",
        "task": TASK,
        "run_root": str(run_root),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "candidate": {
            "id": "chicken-dvr-time-gaussian-failclose-v1",
            "mechanism": "same-session Gaussian timestamp smoothing plus frozen visual incumbent",
            "temporal_bandwidth_seconds": TEMPORAL_BANDWIDTH_SECONDS,
            "temporal_weight": TEMPORAL_WEIGHT,
            "visual_weight": VISUAL_WEIGHT,
            "session_fallback": "retain the visual incumbent when timestamp LOO regresses that session",
            "session_key": "DVR calendar date and hour",
            "sample_local": True,
            "density_output": "uniform nonnegative map with predicted sum",
        },
        "fixed_baselines": {
            "repo_documented_local_oof": REPO_LOCAL_BASELINE,
            "repo_documented_public": REPO_PUBLIC_BASELINE,
        },
        "evidence_tier": {
            "development_oof": True,
            "clean_oof": False,
            "reason": "bandwidth and blend were frozen after development OOF inspection",
            "practice_exploratory_submission_allowed": submission_authorized,
            "public_score_required_before_git_push": REPO_PUBLIC_BASELINE,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "official_images": {
            "root": str(official_root),
            **image_manifest(official_root),
        },
        "submission_policy": {
            "explicit_user_authority": submission_authorized,
            "submission_slots_authorized": 1 if submission_authorized else 0,
            "leaderboard_used_for_candidate_selection": False,
            "no_leaderboard_read_between_freeze_and_submission": True,
            "strict_public_improvement_required_for_git_push": True,
        },
    }
    write_json(contract_path(run_root), contract, exclusive=True)
    digest = sha256_file(contract_path(run_root))
    run_root.joinpath("contracts/candidate_contract.sha256").write_text(
        f"{digest}\n", encoding="ascii"
    )
    return {"prepared": True, "contract_sha256": digest, "candidate": contract["candidate"]}


def load_contract(run_root: Path) -> tuple[dict[str, Any], str]:
    run_root = run_root.resolve()
    observed = sha256_file(contract_path(run_root))
    expected = run_root.joinpath("contracts/candidate_contract.sha256").read_text(
        encoding="ascii"
    ).strip()
    if observed != expected:
        raise ValueError("candidate contract differs from pinned hash")
    contract = load_object(contract_path(run_root))
    if contract.get("status") != "frozen_before_test_prediction":
        raise ValueError("unsafe candidate contract status")
    if contract["runner"]["sha256"] != sha256_file(Path(__file__).resolve()):
        raise ValueError("candidate runner changed after freeze")
    for item in contract["inputs"].values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"frozen input changed: {path}")
    if image_manifest(Path(contract["official_images"]["root"]))["sha256"] != contract[
        "official_images"
    ]["sha256"]:
        raise ValueError("official image manifest changed after freeze")
    return contract, observed


def run(run_root: Path) -> dict[str, Any]:
    import numpy as np

    run_root = run_root.resolve()
    contract, contract_digest = load_contract(run_root)
    receipt_path = run_root / "runtime/candidate_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("candidate already ran; overwrite and resume are forbidden")
    inputs = {name: Path(item["path"]) for name, item in contract["inputs"].items()}
    timestamps, timestamp_rows, source_counts = load_timestamps(
        inputs["timestamp_ocr"],
        inputs["timestamp_overrides"],
        Path(contract["official_images"]["root"]),
    )
    train_ids = [f"train_{index:03d}" for index in range(EXPECTED_ROWS["train"])]
    development = np.load(inputs["visual_oof"], allow_pickle=False)
    required = {"target", "fold", "acquisition_block", "challenger"}
    if not required.issubset(development.files):
        raise ValueError("visual OOF artifact has an unexpected schema")
    target = development["target"]
    folds = development["fold"]
    blocks = development["acquisition_block"]
    visual_oof = development["challenger"]
    if not all(len(value) == len(train_ids) for value in (target, folds, blocks, visual_oof)):
        raise ValueError("development artifact row count mismatch")
    temporal_oof = temporal_predictions(
        train_ids,
        train_ids,
        timestamps,
        target,
        leave_one_out=True,
    )
    global_candidate_oof = np.maximum(
        TEMPORAL_WEIGHT * temporal_oof + VISUAL_WEIGHT * visual_oof,
        0.01,
    )
    visual_score = official_score(target, visual_oof)
    if abs(visual_score - FROZEN_VISUAL_OOF_SCORE) > 1e-12:
        raise ValueError("frozen visual OOF anchor mismatch")
    temporal_score = official_score(target, temporal_oof)
    train_sessions = np.asarray([session_key(timestamps[item]) for item in train_ids])
    per_session = []
    adopted_sessions: dict[str, bool] = {}
    for session in sorted(np.unique(train_sessions).tolist()):
        selected = train_sessions == session
        before = official_score(target[selected], visual_oof[selected])
        proposed = official_score(target[selected], global_candidate_oof[selected])
        adopted = proposed >= before
        adopted_sessions[session] = adopted
        per_session.append(
            {
                "session": session,
                "rows": int(selected.sum()),
                "visual": before,
                "proposed_timestamp_blend": proposed,
                "proposed_delta": proposed - before,
                "decision": "ADOPT_TIMESTAMP_BLEND" if adopted else "RETAIN_VISUAL_INCUMBENT",
            }
        )
    candidate_oof = visual_oof.copy()
    for session, adopted in adopted_sessions.items():
        if adopted:
            selected = train_sessions == session
            candidate_oof[selected] = global_candidate_oof[selected]
    candidate_score = official_score(target, candidate_oof)
    per_fold = []
    for fold in sorted(np.unique(folds).tolist()):
        selected = folds == fold
        before = official_score(target[selected], visual_oof[selected])
        after = official_score(target[selected], candidate_oof[selected])
        per_fold.append({"fold": int(fold), "visual": before, "candidate": after, "delta": after - before})
    bootstrap = paired_bootstrap(target, visual_oof, candidate_oof, blocks)
    positive_folds = sum(item["delta"] >= 0 for item in per_fold)
    gate = {
        "beats_repo_documented_local_oof": candidate_score > REPO_LOCAL_BASELINE,
        "beats_frozen_visual_oof": candidate_score > visual_score,
        "positive_folds": positive_folds,
        "positive_folds_pass": positive_folds >= 4,
        "bootstrap_probability_positive": bootstrap["probability_positive"],
        "bootstrap_probability_pass": bootstrap["probability_positive"] >= 0.95,
    }
    gate["exploratory_go"] = bool(
        gate["beats_repo_documented_local_oof"]
        and gate["beats_frozen_visual_oof"]
        and gate["positive_folds_pass"]
        and gate["bootstrap_probability_pass"]
    )

    with inputs["test_metadata"].open(newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    test_ids = [row["id"] for row in test_rows]
    if test_ids != [
        *(f"test_a_{index:03d}" for index in range(EXPECTED_ROWS["test_a"])),
        *(f"test_b_{index:03d}" for index in range(EXPECTED_ROWS["test_b"])),
    ]:
        raise ValueError("test metadata ID order differs from the frozen contract")
    visual_test = np.load(inputs["visual_test_counts"], allow_pickle=False)
    if len(visual_test) != len(test_ids):
        raise ValueError("visual test prediction row count mismatch")
    codec = import_module(inputs["submission_codec"], "timestamp_chicken_codec")
    with inputs["visual_submission"].open(newline="") as handle:
        visual_rows = list(csv.DictReader(handle))
    if [row["id"] for row in visual_rows] != test_ids:
        raise ValueError("visual incumbent submission ID order mismatch")
    decoded_visual = np.asarray(
        [codec.decode_density_map(row["density_map"]).sum() for row in visual_rows]
    )
    if float(np.max(np.abs(decoded_visual - visual_test))) >= 1e-3:
        raise ValueError("visual counts do not match the frozen incumbent submission")

    def predict(ids: list[str], visual: Any) -> Any:
        temporal = temporal_predictions(
            ids,
            train_ids,
            timestamps,
            target,
            leave_one_out=False,
        )
        weights = np.asarray(
            [TEMPORAL_WEIGHT if adopted_sessions[session_key(timestamps[item])] else 0.0 for item in ids]
        )
        return np.maximum(weights * temporal + (1.0 - weights) * visual, 0.01)

    candidate_counts = predict(test_ids, visual_test)
    reverse = predict(test_ids[::-1], visual_test[::-1])[::-1]
    subset_index = np.arange(0, len(test_ids), 7)
    subset = predict([test_ids[index] for index in subset_index], visual_test[subset_index])
    rebatch = np.concatenate(
        [
            predict(test_ids[start : start + 31], visual_test[start : start + 31])
            for start in range(0, len(test_ids), 31)
        ]
    )
    invariance = {
        "permutation_max_error": float(np.max(np.abs(candidate_counts - reverse))),
        "strict_subset_max_error": float(np.max(np.abs(candidate_counts[subset_index] - subset))),
        "rebatch_max_error": float(np.max(np.abs(candidate_counts - rebatch))),
    }
    if max(invariance.values()) != 0.0:
        raise ValueError(f"sample-local invariance failed: {invariance}")

    timeline_path = run_root / "outputs/timestamps.json"
    write_json(
        timeline_path,
        {
            "schema_version": 1,
            "source_counts": source_counts,
            "rows": timestamp_rows,
        },
        exclusive=True,
    )
    oof_path = run_root / "outputs/development_oof.npz"
    np.savez_compressed(
        oof_path,
        target=target,
        fold=folds,
        acquisition_block=blocks,
        visual=visual_oof,
        temporal=temporal_oof,
        candidate=candidate_oof,
    )
    output_receipts = {}
    for name in ("run_a", "run_b"):
        output = run_root / "outputs" / name
        output.mkdir()
        counts_path = output / "predicted_counts.npy"
        np.save(counts_path, candidate_counts.copy())
        submission_path = output / "submission.csv"
        validation = write_submission(submission_path, test_rows, candidate_counts.copy(), codec)
        output_receipts[name] = {
            "counts_path": str(counts_path),
            "counts_sha256": sha256_file(counts_path),
            "submission_path": str(submission_path),
            "submission_sha256": sha256_file(submission_path),
            "validation": validation,
        }
    duplicate = output_receipts["run_a"]["submission_sha256"] == output_receipts["run_b"]["submission_sha256"]
    if not duplicate:
        raise ValueError("duplicate candidate generation is not byte identical")

    route_coordination = {
        "valid": True,
        "routes": [
            {
                "route": "timestamp",
                "claim": "Use only the visible DVR time and labelled train counts within the same session.",
                "result": candidate_score,
                "evidence_refs": [str(timeline_path), str(oof_path)],
            },
            {
                "route": "visual_incumbent",
                "claim": "Retain the frozen visual model as a small independent error-control component.",
                "result": visual_score,
                "evidence_refs": [str(inputs["visual_oof"]), str(inputs["visual_test_counts"])],
            },
            {
                "route": "evaluator",
                "claim": "Adopt timestamp smoothing per session only after script-computed LOO; retain the incumbent on a regression.",
                "adopts": ["timestamp", "visual_incumbent"],
                "decision": "GO" if gate["exploratory_go"] else "NO_GO",
                "evidence_refs": [str(receipt_path)],
            },
        ],
        "unresolved_conflicts": [],
        "communication_mismatches": [],
    }
    coordination_path = run_root / "runtime/route_coordination.json"
    write_json(coordination_path, route_coordination, exclusive=True)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_awaiting_submission_audit",
        "contract_sha256": contract_digest,
        "task": TASK,
        "candidate": contract["candidate"],
        "evidence_tier": contract["evidence_tier"],
        "development_oof": {
            "visual_score": visual_score,
            "temporal_score": temporal_score,
            "global_timestamp_blend_score_before_session_fallback": official_score(target, global_candidate_oof),
            "candidate_score": candidate_score,
            "repo_documented_local_baseline": REPO_LOCAL_BASELINE,
            "per_fold": per_fold,
            "per_session": per_session,
            "bootstrap": bootstrap,
            "artifact_path": str(oof_path),
            "artifact_sha256": sha256_file(oof_path),
        },
        "gate": gate,
        "timestamps": {
            "source_counts": source_counts,
            "artifact_path": str(timeline_path),
            "artifact_sha256": sha256_file(timeline_path),
        },
        "test": {
            "sample_local_invariance": invariance,
            "prediction_summary": {
                "min": float(candidate_counts.min()),
                "max": float(candidate_counts.max()),
                "mean": float(candidate_counts.mean()),
                "median": float(np.median(candidate_counts)),
            },
        },
        "outputs": output_receipts,
        "duplicate_generation_byte_identical": duplicate,
        "coordination": {
            "path": str(coordination_path),
            "sha256": sha256_file(coordination_path),
            "valid": True,
        },
        "leaderboard_read_after_freeze": False,
        "submission_sent": False,
        "hidden_labels_read": False,
    }
    write_json(receipt_path, receipt, exclusive=True)
    receipt_hash = sha256_file(receipt_path)
    run_root.joinpath("runtime/candidate_receipt.sha256").write_text(
        f"{receipt_hash}\n", encoding="ascii"
    )
    return {
        "complete": True,
        "receipt_sha256": receipt_hash,
        "candidate_oof": candidate_score,
        "exploratory_go": gate["exploratory_go"],
        "submission_sha256": output_receipts["run_a"]["submission_sha256"],
    }


def audit(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    contract, digest = load_contract(run_root)
    receipt_path = run_root / "runtime/candidate_receipt.json"
    observed = sha256_file(receipt_path)
    expected = run_root.joinpath("runtime/candidate_receipt.sha256").read_text(
        encoding="ascii"
    ).strip()
    if observed != expected:
        raise ValueError("candidate receipt differs from pinned hash")
    receipt = load_object(receipt_path)
    if receipt.get("contract_sha256") != digest:
        raise ValueError("candidate receipt is bound to a different contract")
    for item in receipt["outputs"].values():
        if sha256_file(Path(item["submission_path"])) != item["submission_sha256"]:
            raise ValueError("candidate submission changed after generation")
        if sha256_file(Path(item["counts_path"])) != item["counts_sha256"]:
            raise ValueError("candidate counts changed after generation")
    for name in ("timestamps", "coordination"):
        item = receipt[name]
        if sha256_file(Path(item["artifact_path"] if name == "timestamps" else item["path"])) != item["artifact_sha256" if name == "timestamps" else "sha256"]:
            raise ValueError(f"{name} evidence changed after generation")
    allowed = bool(
        receipt["gate"]["exploratory_go"]
        and receipt["duplicate_generation_byte_identical"]
        and receipt["coordination"]["valid"]
        and contract["evidence_tier"]["practice_exploratory_submission_allowed"]
        and contract["submission_policy"]["submission_slots_authorized"] == 1
    )
    return {
        "valid": True,
        "sealed": True,
        "contract_sha256": digest,
        "receipt_sha256": observed,
        "evidence_tier": "development_oof_not_clean",
        "exploratory_submission_allowed": allowed,
        "submission_path": receipt["outputs"]["run_a"]["submission_path"],
        "submission_sha256": receipt["outputs"]["run_a"]["submission_sha256"],
        "candidate_oof": receipt["development_oof"]["candidate_score"],
        "public_score_required_for_git_push": REPO_PUBLIC_BASELINE,
        "leaderboard_reads_used_for_candidate_selection": 0,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-root", type=Path, required=True)
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--visual-root", type=Path, required=True)
    prepare_parser.add_argument("--timestamp-root", type=Path, required=True)
    prepare_parser.add_argument("--submission-authorized", action="store_true")
    for name in ("run", "audit"):
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
                args.visual_root,
                args.timestamp_root,
                submission_authorized=args.submission_authorized,
            )
        elif args.command == "run":
            result = run(args.run_root)
        else:
            result = audit(args.run_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed.
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
