#!/usr/bin/env python3
"""Validate an equal-budget, seal-before-unblind Agent-System A/B contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from native import evidence


CONTROLLED = (
    "model", "effort", "deadline_min", "max_cost_usd", "max_submissions",
    "solver_count", "evaluator", "hardware",
)


def validate(path: Path) -> dict:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    arms = contract.get("arms", [])
    if len(arms) != 2:
        return {"valid": False, "errors": ["exactly two treatment arms required"]}
    left, right = arms
    for key in CONTROLLED:
        if left.get(key) != right.get(key):
            errors.append(f"controlled field differs: {key}")
    order_a = left.get("task_order", [])
    order_b = right.get("task_order", [])
    if order_b != list(reversed(order_a)):
        errors.append("task order must be reversed across the pair")
    roots_a = {str(item) for item in left.get("run_roots", [])}
    roots_b = {str(item) for item in right.get("run_roots", [])}
    if roots_a & roots_b:
        errors.append("treatment arms share run roots")
    for arm in arms:
        if arm.get("leaderboard_reads_during_search") not in (0, False):
            errors.append(f"{arm.get('name')} read leaderboard during search")
        sealed = arm.get("sealed_at_unix")
        unblind = arm.get("unblind_at_unix")
        if not isinstance(sealed, (int, float)) or not isinstance(unblind, (int, float)):
            errors.append(f"{arm.get('name')} lacks seal/unblind timestamps")
        elif sealed >= unblind:
            errors.append(f"{arm.get('name')} unblinded before sealing")
        manifest = Path(str(arm.get("artifact_manifest", "")))
        expected = arm.get("artifact_manifest_sha256")
        if not manifest.is_file():
            errors.append(f"{arm.get('name')} artifact manifest missing")
        elif evidence.sha256_file(manifest) != expected:
            errors.append(f"{arm.get('name')} artifact manifest hash mismatch")
    return {
        "schema_version": 1,
        "benchmark_id": contract.get("benchmark_id"),
        "valid": not errors,
        "controlled_fields": list(CONTROLLED),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--receipt")
    args = parser.parse_args()
    try:
        result = validate(Path(args.contract))
    except Exception as exc:  # noqa: BLE001
        result = {"valid": False,
                  "errors": [f"{type(exc).__name__}: {exc}"]}
    if args.receipt:
        evidence.write_json(Path(args.receipt), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
