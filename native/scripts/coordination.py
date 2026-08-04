#!/usr/bin/env python3
"""Audit what the routes actually shared and what the final run used.

The facts board is communication, not proof of cooperation.  This script turns
its typed references into an explicit route-coordination receipt: claims,
adoptions, conflicts, evaluator decisions, result provenance, and collaboration
edges.  Missing routes or invented references fail closed in clean-benchmark
mode; zero collaboration edges is valid but reported as an independent race.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from native import evidence, facts


def _read(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            out.append({"_invalid_line": number, "_error": str(exc)})
    return out


def audit(ws: Path, routes: list[str]) -> dict:
    ws = Path(ws)
    board_audit = facts.audit_jsonl(ws / "facts.jsonl", "task")
    records = board_audit["records"]
    mode = evidence.run_mode(ws)
    errors: list[str] = [f"facts board: {item}"
                         for item in board_audit["errors"]]
    warnings: list[str] = []

    ids = {str(item.get("id")): item for item in records if item.get("id")}
    claims = [item for item in records if item.get("kind") == "claim"]
    for route in routes:
        if not any(item.get("src") == route for item in claims):
            errors.append(f"{route} has no declared claim")

    edges: list[dict] = []
    adoptions = [item for item in records if item.get("kind") == "adoption"]
    conflicts = [item for item in records if item.get("kind") == "conflict"]
    decisions = [item for item in records if item.get("kind") == "decision"]
    results = [item for item in records if item.get("kind") == "result"]
    for item in adoptions:
        for related in item.get("related_ids", []):
            source = ids.get(str(related))
            if source is None:
                errors.append(f"adoption {item.get('id')} references missing {related}")
                continue
            if source.get("src") == item.get("src"):
                errors.append(
                    f"adoption {item.get('id')} is a self-adoption of {related}"
                )
                continue
            edges.append({
                "from": source.get("src"),
                "to": item.get("src"),
                "fact_id": related,
                "adoption_id": item.get("id"),
            })

    resolved_conflicts: set[str] = set()
    for item in decisions:
        for related in item.get("related_ids", []):
            if related not in ids:
                errors.append(f"decision {item.get('id')} references missing {related}")
            if ids.get(related, {}).get("kind") == "conflict":
                resolved_conflicts.add(related)
    unresolved = [item.get("id") for item in conflicts
                  if item.get("id") not in resolved_conflicts]
    if unresolved:
        message = f"unresolved conflicts: {unresolved}"
        (errors if mode == "clean-benchmark" else warnings).append(message)

    for item in results + decisions:
        if item.get("src") not in facts.TRUSTED:
            errors.append(f"untrusted {item.get('kind')} source {item.get('src')}")
        refs = item.get("evidence_refs", [])
        if not refs:
            errors.append(f"{item.get('id')} has no evidence refs")
        for ref in refs:
            if not (ws / ref).is_file():
                errors.append(f"{item.get('id')} evidence does not exist: {ref}")

    tainted = [item.get("id") for item in records if item.get("tainted")]
    if tainted and mode == "clean-benchmark":
        errors.append(f"tainted facts entered clean board: {tainted}")
    if not edges:
        warnings.append("no cross-route adoption was recorded; this was an independent race")

    firewall = evidence.audit(ws)
    if not firewall["valid"]:
        errors.extend(firewall["errors"])
    if mode == "clean-benchmark" and firewall["tainted_processes"]:
        errors.append("clean benchmark contains tainted processes: "
                      + ", ".join(firewall["tainted_processes"]))

    receipt = {
        "schema_version": 1,
        "mode": mode,
        "routes": routes,
        "claims": claims,
        "adoptions": adoptions,
        "collaboration_edges": edges,
        "conflicts": conflicts,
        "decisions": decisions,
        "results": results,
        "board_audit": {
            "valid": board_audit["valid"],
            "record_count": board_audit["record_count"],
        },
        "firewall": firewall,
        "interpretation": (
            "cooperative" if edges else "independent_e2e_race"
        ),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    evidence.write_json(ws / "route_coordination.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--routes", required=True,
                        help="comma-separated solver identities")
    args = parser.parse_args()
    result = audit(Path(args.workspace), [item.strip() for item in
                   args.routes.split(",") if item.strip()])
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
