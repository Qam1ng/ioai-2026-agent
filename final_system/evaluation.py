from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
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


def _verify_metric_tests(metric_path: Path, tests_path: Path) -> dict:
    try:
        value = json.loads(tests_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"metric_tests.json is unreadable: {exc}") from exc
    cases = value.get("cases") if isinstance(value, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("metric_tests.json must contain at least one case")
    score = _load_score(metric_path)
    checked = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not all(
            key in case for key in ("y_true", "y_pred", "expected")
        ):
            raise ValueError(f"metric test case {index} is incomplete")
        tolerance = float(case.get("tolerance", 1e-8))
        observed = _as_float(score(case["y_true"], case["y_pred"]))
        expected = float(case["expected"])
        if not (abs(observed - expected) <= tolerance):
            raise ValueError(
                f"metric test case {index} failed: expected {expected}, got {observed}"
            )
        checked += 1
    return {"case_count": checked}


def _verify_contract_evidence(path: Path, direction: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"contract_evidence.json is unreadable: {exc}") from exc
    required = (
        "metric_name", "direction", "metric_source", "id_source",
        "label_source", "split_rationale",
    )
    if not isinstance(value, dict) or any(
        not str(value.get(key, "")).strip() for key in required
    ):
        raise ValueError(f"contract_evidence.json requires non-empty {required}")
    if value["direction"] not in {"maximize", "minimize"}:
        raise ValueError("contract evidence direction must be maximize or minimize")
    if direction != "auto" and value["direction"] != direction:
        raise ValueError(
            f"contract direction {value['direction']!r} differs from {direction!r}"
        )
    return value


def freeze_contract(source_workspace: Path, target: Path, *, direction: str) -> dict:
    """Freeze the first structurally and semantically self-tested evaluator."""
    source_workspace = Path(source_workspace)
    target = Path(target)
    existing = read_json(target / "contract.json")
    if existing:
        verify_contract(target)
        return existing

    metric = source_workspace / "metric.py"
    folds_path = source_workspace / "folds.json"
    tests_path = source_workspace / "metric_tests.json"
    evidence_path = source_workspace / "contract_evidence.json"
    required = (metric, folds_path, tests_path, evidence_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            "metric.py/folds.json/metric_tests.json/contract_evidence.json "
            "are not all ready"
        )
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    problems = validate_spec(folds)
    if problems:
        raise ValueError(f"invalid folds.json: {problems}")
    test_result = _verify_metric_tests(metric, tests_path)
    evidence = _verify_contract_evidence(evidence_path, direction)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    shutil.copy2(metric, temporary / "metric.py")
    shutil.copy2(folds_path, temporary / "folds.json")
    shutil.copy2(tests_path, temporary / "metric_tests.json")
    shutil.copy2(evidence_path, temporary / "contract_evidence.json")
    value = {
        "schema_version": 1,
        "source": "hearsay_evaluator",
        "created_at": time.time(),
        "direction": evidence["direction"],
        "metric_sha256": sha256_file(temporary / "metric.py"),
        "folds_sha256": sha256_file(temporary / "folds.json"),
        "metric_tests_sha256": sha256_file(temporary / "metric_tests.json"),
        "contract_evidence_sha256": sha256_file(
            temporary / "contract_evidence.json"
        ),
        "metric_test_count": test_result["case_count"],
        "metric_name": evidence["metric_name"],
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
    if sha256_file(root / "metric_tests.json") != value.get("metric_tests_sha256"):
        raise ValueError("metric_tests.json changed after freeze")
    if sha256_file(root / "contract_evidence.json") != value.get(
        "contract_evidence_sha256"
    ):
        raise ValueError("contract_evidence.json changed after freeze")
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
    candidates: list[tuple[int, Path]] = []
    for path in Path(assets).rglob("*.csv"):
        name = path.name.lower().replace("-", "_")
        if "sample" in name and "submission" in name:
            priority = 0
        elif name == "submission.csv":
            priority = 1
        else:
            continue
        candidates.append((priority, path))
    return min(
        candidates, key=lambda item: (item[0], len(item[1].parts), str(item[1]))
    )[1] if candidates else None


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
    if sample is None:
        errors.append("official submission template was not found in approved assets")
    else:
        with sample.open(newline="", encoding="utf-8-sig") as handle:
            sample_rows = list(csv.reader(handle))
        if not sample_rows or not sample_rows[0]:
            errors.append("official submission template has no header")
            sample_rows = []
        if sample_rows and any(
            len(row) != len(sample_rows[0]) for row in sample_rows[1:]
        ):
            errors.append("official submission template has inconsistent row widths")
        if not sample_rows:
            sample_rows = [[]]
        if candidate_rows[0] != sample_rows[0]:
            errors.append(
                f"header mismatch: expected {sample_rows[0]}, got {candidate_rows[0]}"
            )
        if len(candidate_rows) != len(sample_rows):
            errors.append(
                f"row count mismatch: expected {len(sample_rows)-1}, "
                f"got {len(candidate_rows)-1}"
            )
        sample_widths_valid = sample_rows and all(
            len(row) == len(sample_rows[0]) for row in sample_rows[1:]
        )
        candidate_widths_valid = all(
            len(row) == len(candidate_rows[0]) for row in candidate_rows[1:]
        )
        if (
            len(candidate_rows) == len(sample_rows)
            and candidate_rows[0] == sample_rows[0]
            and sample_widths_valid
            and candidate_widths_valid
        ):
            if sample_rows and len(sample_rows[0]) > 1:
                expected_ids = [row[0] for row in sample_rows[1:]]
                observed_ids = [row[0] for row in candidate_rows[1:]]
                if observed_ids != expected_ids:
                    errors.append("first-column IDs/order differ from sample_submission.csv")
            for column in range(1, len(sample_rows[0])):
                try:
                    reference = [float(row[column]) for row in sample_rows[1:]]
                    numeric = bool(reference) and all(
                        math.isfinite(value) for value in reference
                    )
                except (TypeError, ValueError):
                    numeric = False
                if not numeric:
                    continue
                try:
                    observed = [float(row[column]) for row in candidate_rows[1:]]
                    valid_numbers = all(math.isfinite(value) for value in observed)
                except (TypeError, ValueError):
                    valid_numbers = False
                if not valid_numbers:
                    errors.append(
                        f"numeric submission column {sample_rows[0][column]!r} "
                        "contains a non-numeric or non-finite value"
                    )
    width = len(candidate_rows[0])
    if not all(str(value).strip() for value in candidate_rows[0]):
        errors.append("CSV header contains an empty column name")
    if len(set(candidate_rows[0])) != width:
        errors.append("CSV header contains duplicate column names")
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


#: IOAI grades the submitted code with a short Report at the top of it. The
#: instruction reaches the agents through the Starter prompt, but a missing
#: Report is only discoverable after the competition (the recovery is a Report
#: Generation prompt plus a Late Submission inside 30 minutes), so it is worth
#: refusing to spend a submission on a kernel that has none.
_REPORT_MARKERS = ("report", "报告")
_REPORT_MIN_CHARS = 120


def _report_header_problem(code_path: Path) -> str:
    """Empty string when the script opens with a Report block."""
    try:
        text = code_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"kernel code_file is unreadable: {exc}"
    header: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!") or stripped.startswith("# -*-"):
            continue
        if stripped.startswith("#") or stripped.startswith(('"""', "'''")):
            header.append(line)
            continue
        if not stripped and header:
            header.append(line)
            continue
        if not stripped:
            continue
        break
    # A docstring opener means the whole leading string literal is the header.
    if text.lstrip().startswith(('"""', "'''")):
        quote = text.lstrip()[:3]
        rest = text.lstrip()[3:]
        end = rest.find(quote)
        header = [rest[: end if end >= 0 else len(rest)]]
    block = "\n".join(header).strip()
    if len(block) < _REPORT_MIN_CHARS or not any(
        marker in block.lower() for marker in _REPORT_MARKERS
    ):
        return (
            "kernel code_file must open with a short Report describing the "
            "submitted code (a leading docstring or comment block naming it "
            "'Report', at least "
            f"{_REPORT_MIN_CHARS} characters); IOAI grades submissions on it"
        )
    return ""


def validate_kernel_package(candidate_root: Path) -> dict:
    """Require the standalone script package that Kaggle actually uploads."""
    candidate_root = Path(candidate_root)
    source = candidate_root / "out" / "kernel"
    if not source.is_dir():
        source = candidate_root / "kernel"
    errors: list[str] = []
    metadata_path = source / "kernel-metadata.json"
    metadata: dict[str, Any] = {}
    if not metadata_path.is_file():
        errors.append("kernel-metadata.json is missing")
    else:
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"kernel-metadata.json is unreadable: {exc}")
    code_name = str(metadata.get("code_file", ""))
    code_path = source / code_name if code_name else source / "__missing__"
    if not code_name:
        errors.append("kernel metadata has no code_file")
    elif Path(code_name).is_absolute() or ".." in Path(code_name).parts:
        errors.append("kernel code_file must stay inside its package")
    elif not code_path.is_file():
        errors.append(f"kernel code_file is missing: {code_name}")
    elif code_path.suffix.lower() != ".py":
        errors.append("formal IOAI kernel code_file must be a plain .py script")
    else:
        problem = _report_header_problem(code_path)
        if problem:
            errors.append(problem)
    if metadata and metadata.get("kernel_type", "script") != "script":
        errors.append("kernel_type must be script")
    if metadata and metadata.get("language", "python") != "python":
        errors.append("kernel language must be python")
    allowed = {"kernel-metadata.json", code_name}
    extras = [
        path.relative_to(source).as_posix()
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.relative_to(source).as_posix() not in allowed
    ] if source.is_dir() else []
    if extras:
        errors.append(
            "Kaggle uploads only code_file; bundle helpers into the standalone "
            f"script instead of adding files: {extras}"
        )
    for field in ("dataset_sources", "kernel_sources", "model_sources"):
        values = metadata.get(field, []) if metadata else []
        if not isinstance(values, list) or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            errors.append(f"kernel metadata {field} must be a list of strings")
    return {
        "valid": not errors,
        "errors": errors,
        "source": str(source),
        "code_file": code_name,
        "metadata": metadata,
    }
