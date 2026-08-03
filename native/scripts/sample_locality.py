#!/usr/bin/env python3
"""Verify that each test prediction depends only on its own sample.

Candidates provide ``out/sample_locality_probes.npz`` with predictions from
the same frozen model under four executions:

``canonical_ids``, ``canonical_predictions``
``permuted_ids``, ``permuted_predictions``
``subset_ids``, ``subset_predictions``
``rebatched_ids``, ``rebatched_predictions``

The script aligns by ID and checks permutation, strict-subset, and rebatch
invariance.  The latter two catch dataset-global preprocessing (test means,
quantiles, clustering, sequence decoders) that a row-order-only heuristic does
not.  The receipt is hash-bound to the probe bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from native import evidence


PROBES = "out/sample_locality_probes.npz"
RECEIPT = "out/sample_locality_receipt.json"
REQUIRED = (
    "canonical_ids", "canonical_predictions",
    "permuted_ids", "permuted_predictions",
    "subset_ids", "subset_predictions",
    "rebatched_ids", "rebatched_predictions",
)


def _unique(ids, label: str) -> list[str]:
    values = [str(item) for item in ids.tolist()]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} IDs contain duplicates")
    return values


def _compare(base_ids: list[str], base_pred, ids, pred, label: str,
             *, atol: float, rtol: float, strict_subset: bool = False) -> dict:
    import numpy as np

    ids = _unique(ids, label)
    if len(pred) != len(ids):
        raise ValueError(f"{label} has {len(ids)} IDs but {len(pred)} predictions")
    base_index = {value: index for index, value in enumerate(base_ids)}
    unknown = sorted(set(ids) - set(base_ids))
    if unknown:
        raise ValueError(f"{label} contains unknown IDs: {unknown[:3]}")
    if strict_subset:
        if not 0 < len(ids) < len(base_ids):
            raise ValueError("subset must be non-empty and strictly smaller than canonical")
    elif set(ids) != set(base_ids):
        raise ValueError(f"{label} must contain exactly the canonical IDs")
    aligned = np.asarray([base_pred[base_index[value]] for value in ids])
    observed = np.asarray(pred)
    if aligned.shape != observed.shape:
        raise ValueError(f"{label} prediction shape {observed.shape} != {aligned.shape}")
    if not (np.issubdtype(aligned.dtype, np.number)
            and np.issubdtype(observed.dtype, np.number)):
        equal = bool(np.array_equal(aligned, observed))
        return {"pass": equal, "max_abs_error": None, "rows": len(ids)}
    delta = np.abs(aligned.astype(float) - observed.astype(float))
    max_error = float(delta.max()) if delta.size else 0.0
    equal = bool(np.allclose(aligned, observed, atol=atol, rtol=rtol,
                             equal_nan=False))
    return {"pass": equal, "max_abs_error": max_error, "rows": len(ids)}


def record(ws: Path, candidate: str, *, atol: float = 1e-6,
           rtol: float = 1e-6) -> dict:
    import numpy as np

    root = Path(ws) / candidate
    probe = root / PROBES
    if not probe.exists():
        raise FileNotFoundError(probe)
    with np.load(probe, allow_pickle=False) as data:
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            raise ValueError(f"probe bundle missing {missing}")
        base_ids = _unique(data["canonical_ids"], "canonical")
        base_pred = data["canonical_predictions"]
        if len(base_pred) != len(base_ids):
            raise ValueError("canonical IDs/predictions length mismatch")
        checks = {
            "permutation": _compare(
                base_ids, base_pred, data["permuted_ids"],
                data["permuted_predictions"], "permutation",
                atol=atol, rtol=rtol,
            ),
            "strict_subset": _compare(
                base_ids, base_pred, data["subset_ids"],
                data["subset_predictions"], "strict_subset",
                atol=atol, rtol=rtol, strict_subset=True,
            ),
            "rebatch": _compare(
                base_ids, base_pred, data["rebatched_ids"],
                data["rebatched_predictions"], "rebatch",
                atol=atol, rtol=rtol,
            ),
        }
    receipt = {
        "schema_version": 1,
        "candidate": candidate,
        "probe_path": PROBES,
        "probe_sha256": evidence.sha256_file(probe),
        "atol": atol,
        "rtol": rtol,
        "checks": checks,
        "valid": all(item["pass"] for item in checks.values()),
    }
    evidence.write_json(root / RECEIPT, receipt)
    return receipt


def check(ws: Path, candidate: str) -> dict:
    root = Path(ws) / candidate
    receipt_path = root / RECEIPT
    probe_path = root / PROBES
    if not receipt_path.exists() or not probe_path.exists():
        return {"candidate": candidate, "valid": False,
                "errors": ["sample-locality probes/receipt missing"]}
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"candidate": candidate, "valid": False,
                "errors": [f"unreadable receipt: {type(exc).__name__}: {exc}"]}
    errors: list[str] = []
    if value.get("candidate") != candidate:
        errors.append("receipt candidate mismatch")
    if value.get("probe_sha256") != evidence.sha256_file(probe_path):
        errors.append("probe bundle changed after receipt")
    for name in ("permutation", "strict_subset", "rebatch"):
        if value.get("checks", {}).get(name, {}).get("pass") is not True:
            errors.append(f"{name} invariance failed")
    if value.get("valid") is not True:
        errors.append("receipt is not valid")
    return {**value, "valid": not errors, "errors": errors,
            "receipt_sha256": evidence.sha256_file(receipt_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    try:
        result = (record(Path(args.workspace), args.candidate,
                         atol=args.atol, rtol=args.rtol)
                  if args.record else check(Path(args.workspace), args.candidate))
    except Exception as exc:  # noqa: BLE001
        result = {"candidate": args.candidate, "valid": False,
                  "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
