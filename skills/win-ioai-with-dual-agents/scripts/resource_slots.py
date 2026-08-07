#!/usr/bin/env python3
"""用账号级 append-only 账本协调 IOAI CPU/GPU batch Kernel 槽位。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ACTIVE_SLOT_STATUSES = {
    "slot_reserved",
    "push_started",
    "push_ambiguous",
    "remote_active",
    "remote_absent_once",
}
TERMINAL_STATES = {"COMPLETE", "ERROR", "FAILED", "CANCEL_ACKNOWLEDGED"}
ACTIVE_STATES = {"QUEUED", "RUNNING", "CANCEL_REQUESTED"}
ABSENCE_CONFIRM_SECONDS = 60
STATUS_PATTERNS = (
    re.compile(r"KernelWorkerStatus\.([A-Z_]+)", re.IGNORECASE),
    re.compile(r"\bhas\s+status\s+[\"']?(?:KernelWorkerStatus\.)?([A-Z_]+)", re.IGNORECASE),
    re.compile(r"[\"']?status[\"']?\s*[:=]\s*[\"']?(?:KernelWorkerStatus\.)?([A-Z_]+)", re.IGNORECASE),
)


@dataclass(frozen=True)
class SlotDecision:
    acquired: bool
    reason: str
    retry_after_seconds: int = 0


def policy_signature(limits: dict[str, int]) -> str:
    capacity_policy = {
        key: limits[key]
        for key in ("max_concurrent_cpu", "max_concurrent_gpu", "per_actor_inflight")
    }
    encoded = json.dumps(capacity_policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def event_age_seconds(event: dict[str, Any], now: datetime) -> float:
    try:
        return max(0.0, (now - parse_utc(str(event["time_utc"]))).total_seconds())
    except (KeyError, TypeError, ValueError):
        return 0.0


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"槽位账本 {path} 第 {line_number} 行损坏：{exc}") from exc
        if not isinstance(event, dict) or not (event.get("lease_id") or event.get("waiter_id")):
            raise RuntimeError(f"槽位账本 {path} 第 {line_number} 行缺少 lease_id/waiter_id")
        events.append(event)
    return events


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_events(events: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        value = event.get(key)
        if value:
            latest[str(value)] = event
    return latest


def active_slot_refs(path: Path) -> set[str]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
        try:
            latest = latest_events(read_events(path), "lease_id")
            return {
                str(event["kernel_ref"])
                for event in latest.values()
                if event.get("status") in ACTIVE_SLOT_STATUSES and event.get("kernel_ref")
            }
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def classify_status(text: str, returncode: int | None = None) -> str:
    upper = text.upper()
    states = {
        match.group(1).upper()
        for pattern in STATUS_PATTERNS
        for match in pattern.finditer(text)
        if match.group(1).upper() in TERMINAL_STATES | ACTIVE_STATES
    }
    stripped = upper.strip().strip('"\'')
    if stripped in TERMINAL_STATES | ACTIVE_STATES:
        states.add(stripped)
    if len(states) != 1:
        if states:
            return "unknown"
    else:
        state = next(iter(states))
        if state in TERMINAL_STATES:
            return "terminal"
        if state in ACTIVE_STATES:
            return "active"
    if returncode not in {None, 0} and ("NOT FOUND" in upper or re.search(r"\b404\b", upper)):
        return "absent"
    return "unknown"


def fetch_kernel_statuses(kaggle_bin: str, refs: set[str]) -> dict[str, str]:
    def fetch_one(ref: str) -> tuple[str, str]:
        try:
            result = subprocess.run(
                [kaggle_bin, "kernels", "status", ref],
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"槽位对账超时：{ref}") from exc
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        state = classify_status(text, result.returncode)
        if result.returncode != 0 and state != "absent":
            raise RuntimeError(f"槽位对账失败：{ref}（exit={result.returncode}，未记录原始响应）")
        return ref, state

    statuses: dict[str, str] = {}
    if not refs:
        return statuses
    with ThreadPoolExecutor(max_workers=min(8, len(refs))) as executor:
        futures = {executor.submit(fetch_one, ref): ref for ref in sorted(refs)}
        for future in as_completed(futures):
            ref, state = future.result()
            statuses[ref] = state
    return statuses


def _transition(path: Path, event: dict[str, Any], now: datetime, status: str, **extra: Any) -> None:
    updated = dict(event)
    updated.update(extra)
    updated.update({"time_utc": iso(now), "status": status})
    append_event(path, updated)


def _reconcile_locked(
    path: Path,
    now: datetime,
    statuses: dict[str, str],
    orphan_seconds: int,
    waiter_ttl_seconds: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    leases = latest_events(read_events(path), "lease_id")
    for event in list(leases.values()):
        if event.get("status") not in ACTIVE_SLOT_STATUSES:
            continue
        ref = str(event.get("kernel_ref") or "")
        remote = statuses.get(ref, "unknown")
        age = event_age_seconds(event, now)
        try:
            event_orphan_seconds = int(event.get("orphan_reservation_seconds", max(orphan_seconds, 300)))
        except (TypeError, ValueError):
            event_orphan_seconds = max(orphan_seconds, 300)
        if remote == "terminal":
            _transition(path, event, now, "released_terminal")
        elif remote == "active" and event.get("status") != "remote_active":
            _transition(path, event, now, "remote_active")
        elif remote == "absent" and event.get("status") == "slot_reserved" and age >= event_orphan_seconds:
            _transition(path, event, now, "released_orphan_before_push")
        elif remote == "absent" and event.get("status") == "push_started" and age >= event_orphan_seconds:
            _transition(path, event, now, "remote_absent_once")
        elif remote == "absent" and event.get("status") == "push_ambiguous" and age >= ABSENCE_CONFIRM_SECONDS:
            _transition(path, event, now, "remote_absent_once")
        elif remote == "absent" and event.get("status") == "remote_absent_once" and age >= ABSENCE_CONFIRM_SECONDS:
            _transition(path, event, now, "released_absent")

    waiters = latest_events(read_events(path), "waiter_id")
    for event in list(waiters.values()):
        try:
            event_waiter_ttl = int(event.get("final_waiter_ttl_seconds", max(waiter_ttl_seconds, 45)))
        except (TypeError, ValueError):
            event_waiter_ttl = max(waiter_ttl_seconds, 45)
        if event.get("status") == "final_waiting" and event_age_seconds(event, now) >= event_waiter_ttl:
            _transition(path, event, now, "final_wait_expired")
    return (
        latest_events(read_events(path), "lease_id"),
        latest_events(read_events(path), "waiter_id"),
    )


def reconcile_slots(
    path: Path,
    *,
    limits: dict[str, int],
    statuses: dict[str, str],
    now: datetime | None = None,
) -> None:
    """只做远端状态对账；未知或查询失败的状态不会释放槽位。"""
    now = now or datetime.now(timezone.utc)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            _reconcile_locked(
                path,
                now,
                statuses,
                limits["orphan_reservation_seconds"],
                limits["final_waiter_ttl_seconds"],
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def try_acquire_slot(
    *,
    path: Path,
    lease_id: str,
    actor: str,
    candidate: str,
    competition: str,
    kernel_ref: str,
    resource_class: str,
    priority: str,
    contract_sha256: str,
    limits: dict[str, int],
    deadline: datetime,
    now: datetime,
    statuses: dict[str, str],
) -> SlotDecision:
    if resource_class not in {"cpu", "gpu"} or priority not in {"normal", "final"}:
        raise RuntimeError("resource_class/priority 非法")
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            leases, waiters = _reconcile_locked(
                path,
                now,
                statuses,
                limits["orphan_reservation_seconds"],
                limits["final_waiter_ttl_seconds"],
            )
            active = [event for event in leases.values() if event.get("status") in ACTIVE_SLOT_STATUSES]
            if any(event.get("lease_id") == lease_id for event in active):
                return SlotDecision(True, "existing_lease")

            current_policy = policy_signature(limits)
            active_policies = {str(event.get("policy_signature") or "UNKNOWN") for event in active}
            if active_policies and active_policies != {current_policy}:
                return SlotDecision(False, "账号中存在不同容量策略的 active Kernel，等待其终态", 15)

            same_actor = [event for event in active if event.get("actor") == actor]
            if len(same_actor) >= limits["per_actor_inflight"]:
                return SlotDecision(False, f"actor {actor} 已有 in-flight Kernel", 15)

            in_final_window = now >= deadline - timedelta(seconds=limits["final_priority_window_seconds"])
            active_waiters = [
                event
                for event in waiters.values()
                if event.get("status") == "final_waiting"
                and event.get("resource_class") == resource_class
                and event_age_seconds(event, now)
                < int(event.get("final_waiter_ttl_seconds", max(limits["final_waiter_ttl_seconds"], 45)))
            ]
            active_waiters.sort(key=lambda event: str(event.get("first_wait_utc") or event.get("time_utc") or ""))
            waiter_id = f"{competition}|{resource_class}|{actor}|{candidate}"
            own_waiter = next((event for event in active_waiters if event.get("waiter_id") == waiter_id), None)

            effective_final = in_final_window and priority == "final"
            if active_waiters and not effective_final:
                return SlotDecision(False, "存在 final waiter，新请求不得抢占释放槽", 15)
            if effective_final and active_waiters and active_waiters[0].get("waiter_id") != waiter_id:
                first_wait = own_waiter.get("first_wait_utc") if own_waiter else iso(now)
                append_event(path, {
                    "waiter_id": waiter_id,
                    "time_utc": iso(now),
                    "first_wait_utc": first_wait,
                    "status": "final_waiting",
                    "actor": actor,
                    "candidate": candidate,
                    "competition": competition,
                    "resource_class": resource_class,
                    "contract_sha256": contract_sha256,
                    "final_waiter_ttl_seconds": limits["final_waiter_ttl_seconds"],
                    "policy_signature": current_policy,
                })
                return SlotDecision(False, "前面已有更早的 final waiter", 10)

            capacity = limits[f"max_concurrent_{resource_class}"]
            occupied = sum(event.get("resource_class") == resource_class for event in active)
            if occupied >= capacity:
                if in_final_window and priority == "final":
                    first_wait = own_waiter.get("first_wait_utc") if own_waiter else iso(now)
                    append_event(path, {
                        "waiter_id": waiter_id,
                        "time_utc": iso(now),
                        "first_wait_utc": first_wait,
                        "status": "final_waiting",
                        "actor": actor,
                        "candidate": candidate,
                        "competition": competition,
                        "resource_class": resource_class,
                        "contract_sha256": contract_sha256,
                        "final_waiter_ttl_seconds": limits["final_waiter_ttl_seconds"],
                        "policy_signature": current_policy,
                    })
                    return SlotDecision(False, f"{resource_class} 槽满，已登记 final waiter", 10)
                return SlotDecision(False, f"{resource_class} 槽满 {occupied}/{capacity}", 15)

            if own_waiter:
                _transition(path, own_waiter, now, "final_wait_admitted")
            append_event(path, {
                "lease_id": lease_id,
                "time_utc": iso(now),
                "status": "slot_reserved",
                "actor": actor,
                "candidate": candidate,
                "competition": competition,
                "kernel_ref": kernel_ref,
                "resource_class": resource_class,
                "priority": priority,
                "contract_sha256": contract_sha256,
                "orphan_reservation_seconds": limits["orphan_reservation_seconds"],
                "policy_signature": current_policy,
            })
            return SlotDecision(True, "slot_acquired")
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def transition_slot(path: Path, lease_id: str, status: str, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            event = latest_events(read_events(path), "lease_id").get(lease_id)
            if event and event.get("status") in ACTIVE_SLOT_STATUSES:
                _transition(path, event, now, status)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
