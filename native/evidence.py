"""Persistent Evidence Firewall state for HearSay.

The agent kernel stays permissive; benchmark claims do not.  This module keeps
the run mode, tainted processes, immutable reveal records, and candidate
evidence in files outside solver-owned directories so a solver cannot turn a
leaderboard read or an unbound score into clean evidence by editing a report.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

try:  # POSIX in every supported IOAI runtime (macOS/Linux).
    import fcntl
except ImportError:  # pragma: no cover - unsupported runtime fails elsewhere.
    fcntl = None


SCHEMA_VERSION = 1
RUN_CONTRACT = "run_contract.json"
FIREWALL_STATE = "evidence/firewall.json"
MODES = ("competition", "clean-benchmark")


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


@contextmanager
def locked(path: Path):
    lock = Path(path).with_suffix(Path(path).suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def init_contract(
    ws: Path,
    *,
    mode: str,
    slug: str,
    model: str,
    effort: str,
    solvers: list[str],
    max_cost_usd: float,
    max_submissions: int,
    deadline_min: float,
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    path = Path(ws) / RUN_CONTRACT
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        immutable = {
            "mode": mode,
            "slug": slug,
            "model": model,
            "effort": effort,
            "solvers": solvers,
            "max_cost_usd": max_cost_usd,
            "max_submissions": max_submissions,
            "deadline_min": deadline_min,
        }
        if any(old.get(key) != value for key, value in immutable.items()):
            raise ValueError("existing run contract differs; use a fresh workspace")
        return old
    value = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_pre_run",
        "created_at_unix": int(time.time()),
        "mode": mode,
        "slug": slug,
        "model": model,
        "effort": effort,
        "solvers": solvers,
        "max_cost_usd": max_cost_usd,
        "max_submissions": max_submissions,
        "deadline_min": deadline_min,
        "leaderboard_feedback_to_solvers": mode == "competition",
        "forbidden_audit_reads_taint_lineage": mode == "clean-benchmark",
        "candidate_scores_must_be_harness_recomputed": True,
        "submission_authority": "deterministic_harness",
    }
    value["contract_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    write_json(path, value)
    return value


def run_mode(ws: Path) -> str:
    path = Path(ws) / RUN_CONTRACT
    if not path.exists():
        return "competition"
    value = json.loads(path.read_text(encoding="utf-8"))
    return str(value.get("mode", "competition"))


def _state_path(ws: Path) -> Path:
    return Path(ws) / FIREWALL_STATE


def load_firewall(ws: Path) -> dict:
    path = _state_path(ws)
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "tainted_processes": {},
            "events": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def taint_process(
    ws: Path,
    process: str,
    reason: str,
    *,
    attempted_action: str = "",
) -> dict:
    path = _state_path(ws)
    with locked(path):
        state = load_firewall(ws)
        item = {
            "process": process,
            "reason": " ".join(str(reason).split())[:500],
            "attempted_action": " ".join(str(attempted_action).split())[:500],
            "ts": int(time.time()),
        }
        state.setdefault("events", []).append({"kind": "taint", **item})
        state.setdefault("tainted_processes", {}).setdefault(process, []).append(item)
        write_json(path, state)
    return item


def is_tainted(ws: Path, process: str) -> bool:
    return process in load_firewall(ws).get("tainted_processes", {})


def record_reveal(ws: Path, value: dict) -> Path:
    """Write an append-only leaderboard reveal receipt outside solver dirs."""
    root = Path(ws) / "evidence" / "reveals"
    root.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": SCHEMA_VERSION, "ts": int(time.time()), **value}
    digest = hashlib.sha256(canonical(body)).hexdigest()
    body["receipt_sha256"] = digest
    path = root / f"{len(list(root.glob('*.json'))):04d}-{digest[:12]}.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, body)
    return path


def audit(ws: Path) -> dict:
    state = load_firewall(ws)
    errors: list[str] = []
    contract_path = Path(ws) / RUN_CONTRACT
    if not contract_path.is_file():
        errors.append("run contract is missing")
    else:
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            expected = contract.pop("contract_sha256", "")
            observed = hashlib.sha256(canonical(contract)).hexdigest()
            if expected != observed:
                errors.append("run contract hash mismatch")
            if contract.get("schema_version") != SCHEMA_VERSION:
                errors.append("unsupported run contract schema")
            if contract.get("mode") not in MODES:
                errors.append("invalid run contract mode")
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            errors.append(f"run contract is unreadable: {exc}")
    if state.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported firewall schema")
    for process, records in state.get("tainted_processes", {}).items():
        if not process or not isinstance(records, list) or not records:
            errors.append(f"invalid taint entry for {process!r}")
    for path in sorted((Path(ws) / "evidence" / "reveals").glob("*.json")):
        try:
            reveal = json.loads(path.read_text(encoding="utf-8"))
            expected = reveal.pop("receipt_sha256", "")
            observed = hashlib.sha256(canonical(reveal)).hexdigest()
            if expected != observed:
                errors.append(f"reveal receipt hash mismatch: {path.name}")
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            errors.append(f"reveal receipt is unreadable: {path.name}: {exc}")
    return {
        "valid": not errors,
        "mode": run_mode(ws),
        "tainted_processes": sorted(state.get("tainted_processes", {})),
        "errors": errors,
    }
