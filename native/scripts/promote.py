#!/usr/bin/env python3
"""No-regression gate + Last-Known-Good.

This is also where "machine-checkable failure criteria" lives. We deliberately
do NOT make solvers declare a structured hypothesis and a predicate before each
experiment — that is a format cage around the agent kernel, and preserving the
kernel is the whole bet. Instead the check happens one level down, at the
*candidate* level, where it is cheap and unambiguous: a candidate replaces LKG
only if this script's independently computed score is higher. How the solver got
there is its own business.

LKG has three evidence channels, because they fail differently:
  * ``development`` — best script-computed OOF used during search.
  * ``clean``       — independently reviewed, hash-bound one-shot OOF evidence.
  * ``public``      — best Kaggle-scored submission; never copied into clean.

``local`` and ``confirmed`` remain read-only aliases for old callers.

Compatibility aliases expose ``public`` as ``confirmed`` and ``development``
as ``local`` to older callers. In a 2-hour window with three problems sharing
one GPU quota, the catastrophic outcome is not a mediocre score — it is zero.

Usage:
  python -m native.scripts.promote --candidate solver_a --workspace .
  python -m native.scripts.promote --confirm solver_a --lb 0.913 --workspace .
  python -m native.scripts.promote --show --workspace .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from native import evidence
from native.scripts.integrity import check_provenance, evidence_path
from native.scripts.sample_locality import check as check_sample_locality
from native.scripts.evaluate import evaluate

STATE = "LKG/state.json"


def load(ws: Path) -> dict:
    p = ws / STATE
    if not p.exists():
        return {"development": None, "clean": None, "public": None,
                "local": None, "confirmed": None}
    value = json.loads(p.read_text())
    value.setdefault("development", value.get("local"))
    value.setdefault("public", value.get("confirmed"))
    value.setdefault("clean", None)
    value["local"] = value["development"]
    value["confirmed"] = value["public"]
    return value


def save(ws: Path, st: dict) -> None:
    p = ws / STATE
    p.parent.mkdir(parents=True, exist_ok=True)
    st["local"] = st.get("development")
    st["confirmed"] = st.get("public")
    evidence.write_json(p, st)


def verify_clean_receipt(ws: Path, cand: str, path: Path,
                         proposer: str, reviewer: str) -> list[str]:
    errors: list[str] = []
    if not proposer or not reviewer or proposer == reviewer:
        errors.append("clean promotion requires reviewer != proposer")
    if proposer != cand:
        errors.append("clean promotion proposer must be the candidate route")
    if reviewer != "evaluator":
        errors.append("clean promotion reviewer must be evaluator")
    if evidence.is_tainted(ws, cand) or evidence.is_tainted(ws, proposer):
        errors.append("candidate/proposer is tainted")
    manifest_path = evidence_path(ws, cand)
    if not manifest_path.exists():
        return errors + ["candidate evidence manifest missing"]
    if not path.exists():
        return errors + ["clean review receipt missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return errors + [f"unreadable clean evidence: {type(exc).__name__}: {exc}"]
    receipt_digest = receipt.pop("receipt_sha256", "")
    if receipt_digest != hashlib.sha256(evidence.canonical(receipt)).hexdigest():
        errors.append("clean review receipt hash mismatch")
    required = {
        "candidate": cand,
        "proposer": proposer,
        "reviewer": reviewer,
        "evidence_tier": "clean_oof",
        "decision": "promote",
        "prediction_sha256": manifest.get("oof"),
        "split_sha256": manifest.get("folds"),
        "metric_sha256": manifest.get("metric"),
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            errors.append(f"clean receipt {key} does not bind candidate evidence")
    expected_manifest_ref = str(manifest_path.relative_to(ws))
    if receipt.get("candidate_manifest") != expected_manifest_ref:
        errors.append("clean receipt candidate manifest path mismatch")
    if receipt.get("candidate_manifest_sha256") != evidence.sha256_file(manifest_path):
        errors.append("clean receipt candidate manifest hash mismatch")

    review_path = ws / "review" / f"{cand}.json"
    expected_review_ref = str(review_path.relative_to(ws))
    if receipt.get("evaluator_review") != expected_review_ref:
        errors.append("clean receipt evaluator review path mismatch")
    if not review_path.is_file():
        errors.append("clean promotion evaluator review is missing")
    else:
        if receipt.get("evaluator_review_sha256") != evidence.sha256_file(review_path):
            errors.append("clean receipt evaluator review hash mismatch")
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if str(review.get("verdict", "")).upper() != "APPROVE":
                errors.append("clean promotion requires evaluator APPROVE")
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"clean evaluator review unreadable: {exc}")

    contract = ws / evidence.RUN_CONTRACT
    observed_contract = evidence.sha256_file(contract) if contract.is_file() else None
    if receipt.get("run_contract_sha256") != observed_contract:
        errors.append("clean receipt run contract hash mismatch")
    locality_receipt = ws / cand / "out" / "sample_locality_receipt.json"
    observed_locality = (
        evidence.sha256_file(locality_receipt) if locality_receipt.is_file() else None
    )
    if receipt.get("sample_locality_receipt_sha256") != observed_locality:
        errors.append("clean receipt sample-locality hash mismatch")
    locality = check_sample_locality(ws, cand)
    if not locality.get("valid"):
        errors.append("clean promotion requires sample-locality receipt")
    errors.extend(check_provenance(ws, cand))
    return errors


def verify_snapshot(path: Path) -> list[str]:
    path = Path(path)
    receipt_path = path / "snapshot_manifest.json"
    if not receipt_path.is_file():
        return ["snapshot manifest missing"]
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [f"snapshot manifest unreadable: {exc}"]
    expected_receipt = receipt.pop("receipt_sha256", "")
    observed_receipt = hashlib.sha256(evidence.canonical(receipt)).hexdigest()
    errors = []
    if expected_receipt != observed_receipt:
        errors.append("snapshot receipt hash mismatch")
    expected_files = receipt.get("files", {})
    observed_files: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item == receipt_path:
            continue
        if item.is_symlink():
            errors.append(f"snapshot contains a symlink: {item.relative_to(path)}")
        elif item.is_file():
            observed_files[str(item.relative_to(path))] = evidence.sha256_file(item)
    if observed_files != expected_files:
        errors.append("snapshot file manifest mismatch")
    digest = hashlib.sha256(evidence.canonical(expected_files)).hexdigest()
    if receipt.get("artifact_sha256") != digest:
        errors.append("snapshot artifact digest mismatch")
    return errors


def snapshot(ws: Path, cand: str, tier: str,
             tier_receipt: Path | None = None) -> str:
    """Content-addressed copy; prior LKG artifacts are never overwritten."""
    src = ws / cand
    manifest: dict[str, str] = {}
    for rel in ("out", "kernel"):
        root = src / rel
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"snapshot input is a symlink: {path}")
            if path.is_file():
                manifest[str(path.relative_to(src))] = evidence.sha256_file(path)
    candidate_evidence = evidence_path(ws, cand)
    if candidate_evidence.is_file():
        manifest["evidence_manifest.json"] = evidence.sha256_file(candidate_evidence)
    if tier_receipt is not None:
        tier_receipt = Path(tier_receipt)
        if not tier_receipt.is_file():
            raise ValueError(f"tier receipt is missing: {tier_receipt}")
        manifest["clean_review_receipt.json"] = evidence.sha256_file(tier_receipt)
    digest = hashlib.sha256(evidence.canonical(manifest)).hexdigest()
    safe_cand = "".join(c if c.isalnum() or c in "-_" else "-" for c in cand)
    dst = ws / "LKG" / "snapshots" / tier / f"{safe_cand}-{digest[:16]}"
    if dst.exists():
        problems = verify_snapshot(dst)
        if problems:
            raise ValueError(f"invalid existing LKG snapshot {dst}: {problems}")
        return str(dst)
    dst.mkdir(parents=True)
    for rel in ("out", "kernel"):
        if (src / rel).exists():
            shutil.copytree(src / rel, dst / rel)
    if candidate_evidence.is_file():
        shutil.copy2(candidate_evidence, dst / "evidence_manifest.json")
    if tier_receipt is not None:
        shutil.copy2(tier_receipt, dst / "clean_review_receipt.json")
    snapshot_receipt = {
        "schema_version": 1,
        "candidate": cand,
        "tier_at_snapshot": tier,
        "artifact_sha256": digest,
        "files": manifest,
    }
    snapshot_receipt["receipt_sha256"] = hashlib.sha256(
        evidence.canonical(snapshot_receipt)
    ).hexdigest()
    evidence.write_json(dst / "snapshot_manifest.json", snapshot_receipt)
    return str(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--candidate", default=None, help="promote by local OOF score")
    ap.add_argument("--tier", choices=["development", "clean"],
                    default="development")
    ap.add_argument("--clean-receipt", default=None)
    ap.add_argument("--proposer", default="")
    ap.add_argument("--reviewer", default="")
    ap.add_argument("--confirm", default=None, help="candidate whose LB score is known")
    ap.add_argument("--lb", type=float, default=None, help="Kaggle-reported score")
    ap.add_argument("--submitted-snapshot", default=None,
                    help="immutable LKG snapshot that was actually submitted")
    ap.add_argument("--submission-sha256", default=None,
                    help="hash of the exact submitted CSV")
    ap.add_argument("--higher-is-better", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    ws = Path(a.workspace)
    st = load(ws)
    better = (lambda new, old: new > old) if a.higher_is_better else (lambda n, o: n < o)

    if a.show:
        print(json.dumps(st, indent=2))
        return 0

    if a.confirm is not None:
        if a.lb is None:
            raise SystemExit("--confirm requires --lb")
        cur = st.get("public")
        if cur is None or better(a.lb, cur["lb"]):
            if a.submitted_snapshot:
                submitted = Path(a.submitted_snapshot).resolve()
                snapshot_root = (ws / "LKG" / "snapshots").resolve()
                if not submitted.is_relative_to(snapshot_root):
                    raise SystemExit("--submitted-snapshot is outside LKG snapshots")
                problems = verify_snapshot(submitted)
                if problems:
                    raise SystemExit("submitted snapshot failed audit: "
                                     + "; ".join(problems))
                submission = submitted / "out" / "submission.csv"
                if not submission.is_file():
                    raise SystemExit("submitted snapshot has no out/submission.csv")
                observed_hash = evidence.sha256_file(submission)
                if not a.submission_sha256 or observed_hash != a.submission_sha256:
                    raise SystemExit("submitted CSV hash does not match its receipt")
                path = str(submitted)
            else:
                path = snapshot(ws, a.confirm, "public")
                submission = Path(path) / "out" / "submission.csv"
                observed_hash = (evidence.sha256_file(submission)
                                 if submission.is_file() else None)
            st["public"] = {"candidate": a.confirm, "lb": a.lb,
                            "ts": int(time.time()), "path": path,
                            "submission_sha256": observed_hash}
            save(ws, st)
            print(json.dumps({"decision": "PROMOTED", "tier": "confirmed",
                              "candidate": a.confirm, "lb": a.lb,
                              "previous": cur["lb"] if cur else None}))
        else:
            print(json.dumps({"decision": "REJECTED", "tier": "confirmed",
                              "candidate": a.confirm, "lb": a.lb,
                              "incumbent": cur["lb"],
                              "reason": "leaderboard score did not improve"}))
        return 0

    if a.candidate is None:
        raise SystemExit("pass --candidate, --confirm, or --show")

    # Score it here, from out/oof.npy. Deliberately NOT read from score.json:
    # that file lives inside the solver's own directory, so trusting it would
    # make "self-reported scores are never believed" a comment rather than a
    # property. (The self-test catches this: a hand-written score.json claiming
    # 0.99 must not promote.)
    sc = evaluate(ws, a.candidate)
    if sc.get("status") != "ok":
        print(json.dumps({"decision": "REJECTED", "candidate": a.candidate,
                          "reason": f"evaluate status={sc.get('status')}: "
                                    f"{sc.get('detail', '')}"}))
        return 1

    if a.tier == "clean":
        if not a.clean_receipt:
            print(json.dumps({"decision": "REJECTED", "candidate": a.candidate,
                              "reason": "clean tier requires --clean-receipt"}))
            return 1
        problems = verify_clean_receipt(
            ws, a.candidate, Path(a.clean_receipt), a.proposer, a.reviewer)
        if problems:
            print(json.dumps({"decision": "REJECTED", "candidate": a.candidate,
                              "tier": "clean", "reason": "; ".join(problems)}))
            return 1

    key = a.tier
    cur = st.get(key)
    if cur is None or better(sc["mean"], cur["mean"]):
        path = snapshot(
            ws, a.candidate, key,
            Path(a.clean_receipt) if a.tier == "clean" else None,
        )
        st[key] = {"candidate": a.candidate, "mean": sc["mean"],
                   "std": sc["std"], "ts": int(time.time()), "path": path,
                   "reviewer": a.reviewer or None,
                   "proposer": a.proposer or a.candidate}
        save(ws, st)
        print(json.dumps({"decision": "PROMOTED", "tier": "local",
                          "candidate": a.candidate, "mean": sc["mean"],
                          "std": sc["std"],
                          "evidence_tier": key,
                          "previous": cur["mean"] if cur else None}))
    else:
        print(json.dumps({"decision": "REJECTED", "tier": "local",
                          "candidate": a.candidate, "mean": sc["mean"],
                          "evidence_tier": key,
                          "incumbent": cur["mean"],
                          "reason": "did not beat LKG on the shared folds"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
