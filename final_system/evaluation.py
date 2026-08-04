from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from native.scripts.checkfolds import validate_spec

from .io import atomic_json, canonical, read_json, sha256_file


def _load_score(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"ioai_metric_{time.time_ns()}", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import metric: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    score = getattr(module, "score", None)
    if not callable(score):
        raise ValueError("metric.py must expose score(y_true, y_pred)")
    return score


def _as_float(value: Any) -> float:
    return float(value["score"]) if isinstance(value, dict) else float(value)


def freeze_contract(source_workspace: Path, target: Path, *, direction: str) -> dict:
    """Freeze the first valid HearSay metric/folds pair; later edits are ignored."""
    source_workspace = Path(source_workspace)
    target = Path(target)
    existing = read_json(target / "contract.json")
    if existing:
        verify_contract(target)
        return existing

    metric = source_workspace / "metric.py"
    folds_path = source_workspace / "folds.json"
    if not metric.is_file() or not folds_path.is_file():
        raise FileNotFoundError("metric.py/folds.json are not both ready")
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    problems = validate_spec(folds)
    if problems:
        raise ValueError(f"invalid folds.json: {problems}")
    _load_score(metric)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    shutil.copy2(metric, temporary / "metric.py")
    shutil.copy2(folds_path, temporary / "folds.json")
    value = {
        "schema_version": 1,
        "source": "hearsay_evaluator",
        "created_at": time.time(),
        "direction": direction,
        "metric_sha256": sha256_file(temporary / "metric.py"),
        "folds_sha256": sha256_file(temporary / "folds.json"),
        "fold_count": len({int(x) for x in folds["fold"] if int(x) >= 0}),
        "unit_count": len(folds["ids"]),
    }
    value["contract_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    atomic_json(temporary / "contract.json", value)
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            shutil.rmtree(temporary, ignore_errors=True)
            raise ValueError("evaluation target exists but is not empty")
        # controller 会预建共享挂载点；空目录可以安全替换，已有内容绝不覆盖。
        target.rmdir()
    os.replace(temporary, target)
    return value


def verify_contract(root: Path) -> dict:
    root = Path(root)
    value = read_json(root / "contract.json")
    if not isinstance(value, dict):
        raise ValueError("evaluation contract is missing")
    if sha256_file(root / "metric.py") != value.get("metric_sha256"):
        raise ValueError("metric.py changed after freeze")
    if sha256_file(root / "folds.json") != value.get("folds_sha256"):
        raise ValueError("folds.json changed after freeze")
    signed = dict(value)
    expected = signed.pop("contract_sha256", "")
    if hashlib.sha256(canonical(signed)).hexdigest() != expected:
        raise ValueError("evaluation contract digest mismatch")
    return value


def evaluate_candidate(contract_root: Path, candidate_root: Path) -> dict:
    import numpy as np

    contract = verify_contract(contract_root)
    folds = json.loads((Path(contract_root) / "folds.json").read_text())
    predictions_path = Path(candidate_root) / "out" / "oof.npy"
    if not predictions_path.is_file():
        return {"status": "no_oof", "contract_sha256": contract["contract_sha256"]}
    labels = np.asarray(folds["labels"])
    fold = np.asarray(folds["fold"])
    predictions = np.load(predictions_path, allow_pickle=False)
    if len(predictions) != len(labels):
        return {
            "status": "shape_mismatch",
            "prediction_rows": len(predictions),
            "expected_rows": len(labels),
            "contract_sha256": contract["contract_sha256"],
        }
    score = _load_score(Path(contract_root) / "metric.py")
    scored_folds = sorted({int(value) for value in fold.tolist() if int(value) >= 0})
    values: list[float] = []
    try:
        for fold_id in scored_folds:
            mask = fold == fold_id
            values.append(_as_float(score(labels[mask], predictions[mask])))
        keep = fold >= 0
        pooled = _as_float(score(labels[keep], predictions[keep]))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "metric_error",
            "detail": f"{type(exc).__name__}: {exc}",
            "contract_sha256": contract["contract_sha256"],
        }
    array = np.asarray(values, dtype=float)
    return {
        "status": "ok",
        "score_source": "common_harness",
        "mean": round(float(array.mean()), 8),
        "std": round(float(array.std()), 8),
        "per_fold": [round(float(value), 8) for value in values],
        "pooled": round(float(pooled), 8),
        "n": int((fold >= 0).sum()),
        "prediction_sha256": sha256_file(predictions_path),
        "contract_sha256": contract["contract_sha256"],
    }


def _find_sample_submission(assets: Path) -> Path | None:
    candidates = [
        path for path in Path(assets).rglob("*.csv")
        if "sample" in path.name.lower() and "submission" in path.name.lower()
    ]
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0] \
        if candidates else None


def validate_submission_csv(assets: Path, candidate_csv: Path) -> dict:
    """Check the format facts a deterministic script can establish."""
    candidate_csv = Path(candidate_csv)
    if not candidate_csv.is_file() or candidate_csv.stat().st_size == 0:
        return {"valid": False, "errors": ["out/submission.csv is missing or empty"]}
    try:
        with candidate_csv.open(newline="", encoding="utf-8-sig") as handle:
            candidate_rows = list(csv.reader(handle))
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "errors": [f"candidate CSV unreadable: {exc}"]}
    if not candidate_rows or not candidate_rows[0]:
        return {"valid": False, "errors": ["candidate CSV has no header"]}
    errors: list[str] = []
    sample = _find_sample_submission(assets)
    if sample:
        with sample.open(newline="", encoding="utf-8-sig") as handle:
            sample_rows = list(csv.reader(handle))
        if candidate_rows[0] != sample_rows[0]:
            errors.append(
                f"header mismatch: expected {sample_rows[0]}, got {candidate_rows[0]}"
            )
        if len(candidate_rows) != len(sample_rows):
            errors.append(
                f"row count mismatch: expected {len(sample_rows)-1}, "
                f"got {len(candidate_rows)-1}"
            )
        if len(candidate_rows) == len(sample_rows) and candidate_rows[0] == sample_rows[0]:
            if sample_rows and len(sample_rows[0]) > 1:
                expected_ids = [row[0] for row in sample_rows[1:]]
                observed_ids = [row[0] for row in candidate_rows[1:]]
                if observed_ids != expected_ids:
                    errors.append("first-column IDs/order differ from sample_submission.csv")
    width = len(candidate_rows[0])
    if any(len(row) != width for row in candidate_rows[1:]):
        errors.append("CSV rows have inconsistent column counts")
    return {
        "valid": not errors,
        "errors": errors,
        "sample_submission": str(sample) if sample else None,
        "rows": max(0, len(candidate_rows) - 1),
        "columns": candidate_rows[0],
        "submission_sha256": sha256_file(candidate_csv),
    }
