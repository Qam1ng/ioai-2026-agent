from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from .io import atomic_json, locked, read_json


class DayResourceGate:
    """Cross-process Kaggle compute gate shared by one account for the event."""

    def __init__(
        self,
        root: Path,
        *,
        pool_id: str,
        floor_group_id: str,
        expected_slugs: tuple[str, ...],
        gpu_limit_hours: float,
        gpu_concurrency: int,
        cpu_concurrency: int,
        poll_seconds: float = 2.0,
        acquire_wait_seconds: float = 5.0,
        floor_grace_seconds: float = 900.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "resource_state.json"
        self.pool_id = pool_id
        self.floor_group_id = floor_group_id
        self.expected_slugs = tuple(sorted(set(expected_slugs)))
        self.gpu_limit_seconds = float(gpu_limit_hours) * 3600.0
        self.gpu_concurrency = int(gpu_concurrency)
        self.cpu_concurrency = int(cpu_concurrency)
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.acquire_wait_seconds = float(acquire_wait_seconds)
        self.floor_grace_seconds = float(floor_grace_seconds)
        if not self.expected_slugs:
            raise ValueError("expected_slugs must not be empty")
        if not self.floor_group_id:
            raise ValueError("floor_group_id must not be empty")
        if self.gpu_limit_seconds <= 0:
            raise ValueError("gpu_limit_hours must be positive")
        if self.gpu_concurrency <= 0 or self.cpu_concurrency <= 0:
            raise ValueError("resource concurrency must be positive")
        if self.acquire_wait_seconds <= 0:
            raise ValueError("acquire_wait_seconds must be positive")
        if self.floor_grace_seconds < 0:
            raise ValueError("floor_grace_seconds must be non-negative")
        with locked(self.state_path):
            state = read_json(self.state_path)
            if state is None:
                atomic_json(self.state_path, self._new_state())
            else:
                self._verify_contract(state)
                groups = state.setdefault("floor_groups", {})
                observed = groups.get(self.floor_group_id)
                if observed is None:
                    groups[self.floor_group_id] = self._new_floor_group(time.time())
                    state["updated_at"] = time.time()
                    atomic_json(self.state_path, state)
                elif tuple(observed.get("expected_slugs") or ()) != self.expected_slugs:
                    raise ValueError(
                        "floor-group slugs differ from the existing shared contract"
                    )
                else:
                    changed = False
                    # 兼容 schema v2 的已有资源池：迁移时从现在起给完整宽限期。
                    if "created_at" not in observed:
                        observed["created_at"] = time.time()
                        changed = True
                    prior_grace = observed.get("floor_grace_seconds")
                    if prior_grace is None:
                        observed["floor_grace_seconds"] = self.floor_grace_seconds
                        changed = True
                    elif float(prior_grace) != self.floor_grace_seconds:
                        raise ValueError(
                            "floor-group grace differs from the existing shared contract"
                        )
                    if changed:
                        state["updated_at"] = time.time()
                        atomic_json(self.state_path, state)

    def _new_floor_group(self, created_at: float) -> dict[str, Any]:
        return {
            "expected_slugs": list(self.expected_slugs),
            "floors": {slug: False for slug in self.expected_slugs},
            "created_at": float(created_at),
            "floor_grace_seconds": self.floor_grace_seconds,
        }

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "pool_id": self.pool_id,
            "gpu_limit_seconds": self.gpu_limit_seconds,
            "gpu_concurrency": self.gpu_concurrency,
            "cpu_concurrency": self.cpu_concurrency,
            "gpu_seconds_used": 0.0,
            "floor_groups": {
                self.floor_group_id: self._new_floor_group(time.time()),
            },
            "waiters": {},
            "leases": {},
            "updated_at": time.time(),
        }

    def _verify_contract(self, state: dict[str, Any]) -> None:
        observed = (
            int(state.get("schema_version", -1)),
            state.get("pool_id"),
            float(state.get("gpu_limit_seconds", -1)),
            int(state.get("gpu_concurrency", -1)),
            int(state.get("cpu_concurrency", -1)),
        )
        expected = (
            2,
            self.pool_id,
            self.gpu_limit_seconds,
            self.gpu_concurrency,
            self.cpu_concurrency,
        )
        if observed != expected:
            raise ValueError(
                "shared Kaggle resource-pool contract differs from this process: "
                f"observed={observed!r}, expected={expected!r}"
            )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _cleanup(self, state: dict[str, Any], now: float) -> None:
        waiters = state.setdefault("waiters", {})
        for key, value in list(waiters.items()):
            if float(value.get("deadline_epoch", 0)) <= now or not self._pid_alive(
                int(value.get("pid", 0))
            ):
                waiters.pop(key, None)
        leases = state.setdefault("leases", {})
        for key, value in list(leases.items()):
            # 只有尚未 push 的本地预留可以在 owner 崩溃后自动释放。
            # 已经有 external_ref 的 Kernel 可能仍在 Kaggle 运行，必须显式对账。
            if (
                value.get("phase") == "reserved"
                and not value.get("external_ref")
                and not self._pid_alive(int(value.get("pid", 0)))
            ):
                leases.pop(key, None)

    @staticmethod
    def _kind(accelerator: str) -> str:
        return "cpu" if accelerator.lower() == "cpu" else "gpu"

    def _floor_gate(self, state: dict[str, Any], now: float) -> dict[str, Any]:
        group = state.get("floor_groups", {}).get(self.floor_group_id, {})
        floors = group.get("floors", {})
        missing = [slug for slug in self.expected_slugs if not floors.get(slug)]
        grace_deadline = float(group.get("created_at", now)) + float(
            group.get("floor_grace_seconds", self.floor_grace_seconds)
        )
        complete = not missing
        grace_expired = now >= grace_deadline
        return {
            "open": complete or grace_expired,
            "complete": complete,
            "grace_expired": grace_expired,
            "grace_deadline_epoch": grace_deadline,
            "missing_slugs": missing,
        }

    def acquire(
        self,
        *,
        submission_id: str,
        slug: str,
        submission_class: str,
        accelerator: str,
        estimated_seconds: float,
        deadline_epoch: float,
        local_score: float | None = None,
    ) -> tuple[dict[str, Any] | None, str, str]:
        if slug not in self.expected_slugs:
            return (
                None,
                "rejected",
                f"slug {slug!r} is not registered in this resource pool",
            )
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        wait_deadline = min(
            float(deadline_epoch), time.time() + self.acquire_wait_seconds
        )
        request = {
            "lease_id": lease_id,
            "submission_id": submission_id,
            "slug": slug,
            "floor_group_id": self.floor_group_id,
            "submission_class": submission_class,
            "accelerator": accelerator.lower(),
            "kind": self._kind(accelerator),
            "estimated_seconds": max(0.0, float(estimated_seconds)),
            # For queue ordering only: higher is better, missing sorts last.
            "local_score": (float(local_score)
                            if local_score is not None else None),
            "created_at": time.time(),
            "deadline_epoch": wait_deadline,
            "latest_start_epoch": float(deadline_epoch),
            "pid": os.getpid(),
        }
        terminal_reason = ""
        while time.time() < wait_deadline:
            now = time.time()
            terminal_reason = ""
            with locked(self.state_path):
                state = read_json(self.state_path, self._new_state())
                self._verify_contract(state)
                self._cleanup(state, now)
                state["waiters"].setdefault(lease_id, request)

                floor_gate = self._floor_gate(state, now)
                ordered = sorted(
                    (
                        item for item in state["waiters"].values()
                        if item.get("kind") == request["kind"]
                    ),
                    key=lambda item: (
                        0 if item.get("submission_class") == "floor" else 1,
                        # The best candidate takes the next slot, not whoever
                        # queued first. On the live run the top OOF (0.934) sat
                        # behind a 0.906 for the whole window because arrival
                        # order was the only tie-break. None sorts last.
                        -(item.get("local_score")
                          if item.get("local_score") is not None else -1e9),
                        float(item.get("created_at", 0)),
                        str(item.get("lease_id", "")),
                    ),
                )
                first = ordered[0] if ordered else None
                if (
                    submission_class != "floor"
                    and not floor_gate["open"]
                ):
                    missing = ", ".join(floor_gate["missing_slugs"])
                    remaining = max(
                        0.0, floor_gate["grace_deadline_epoch"] - now
                    )
                    terminal_reason = (
                        "waiting for floor submissions from "
                        f"[{missing}]; soft gate opens in {remaining:.1f}s"
                    )
                elif not first or first.get("lease_id") != lease_id:
                    terminal_reason = "waiting behind an earlier resource request"
                else:
                    active = list(state.get("leases", {}).values())
                    kind = request["kind"]
                    cap = (
                        self.cpu_concurrency if kind == "cpu"
                        else self.gpu_concurrency
                    )
                    in_use = sum(item.get("kind") == kind for item in active)
                    if in_use >= cap:
                        terminal_reason = f"{kind} concurrency is full"
                    elif kind == "gpu":
                        used = float(state.get("gpu_seconds_used", 0.0))
                        committed = used + sum(
                            float(item.get("estimated_seconds", 0.0))
                            for item in active
                            if item.get("kind") == "gpu"
                        )
                        if used + request["estimated_seconds"] > self.gpu_limit_seconds:
                            terminal_reason = "shared GPU-hour quota is exhausted"
                            state["waiters"].pop(lease_id, None)
                            state["updated_at"] = now
                            atomic_json(self.state_path, state)
                            return None, "resource_exhausted", terminal_reason
                        if committed + request["estimated_seconds"] > self.gpu_limit_seconds:
                            terminal_reason = (
                                "shared GPU-hour quota is temporarily reserved "
                                "by active kernels"
                            )
                    if not terminal_reason:
                        lease = {
                            **request,
                            "phase": "reserved",
                            "acquired_at": now,
                            "external_ref": "",
                        }
                        state["waiters"].pop(lease_id, None)
                        state["leases"][lease_id] = lease
                        state["updated_at"] = now
                        atomic_json(self.state_path, state)
                        return lease, "acquired", "acquired"
                state["updated_at"] = now
                atomic_json(self.state_path, state)
            time.sleep(min(self.poll_seconds, max(0.0, wait_deadline - time.time())))

        with locked(self.state_path):
            state = read_json(self.state_path, self._new_state())
            state.setdefault("waiters", {}).pop(lease_id, None)
            state["updated_at"] = time.time()
            atomic_json(self.state_path, state)
        status = (
            "deadline_blocked"
            if time.time() >= float(deadline_epoch) else "resource_deferred"
        )
        return None, status, terminal_reason or "resource wait budget exhausted"

    def mark_running(self, lease_id: str, external_ref: str) -> None:
        with locked(self.state_path):
            state = read_json(self.state_path, self._new_state())
            lease = state.get("leases", {}).get(lease_id)
            if lease is None:
                raise ValueError(f"unknown resource lease: {lease_id}")
            lease["phase"] = "running"
            lease["external_ref"] = external_ref
            lease["running_at"] = time.time()
            state["updated_at"] = time.time()
            atomic_json(self.state_path, state)

    def mark_external_unknown(self, lease_id: str) -> None:
        with locked(self.state_path):
            state = read_json(self.state_path, self._new_state())
            lease = state.get("leases", {}).get(lease_id)
            if lease is not None:
                lease["phase"] = "external_unknown"
                state["updated_at"] = time.time()
                atomic_json(self.state_path, state)

    def release(self, lease_id: str) -> None:
        with locked(self.state_path):
            state = read_json(self.state_path, self._new_state())
            state.setdefault("leases", {}).pop(lease_id, None)
            state["updated_at"] = time.time()
            atomic_json(self.state_path, state)

    def settle(self, lease_id: str, actual_seconds: float) -> None:
        with locked(self.state_path):
            state = read_json(self.state_path, self._new_state())
            lease = state.setdefault("leases", {}).pop(lease_id, None)
            if lease and lease.get("kind") == "gpu":
                state["gpu_seconds_used"] = float(
                    state.get("gpu_seconds_used", 0.0)
                ) + max(0.0, float(actual_seconds))
            state["updated_at"] = time.time()
            atomic_json(self.state_path, state)

    def mark_floor(self, slug: str) -> None:
        with locked(self.state_path):
            state = read_json(self.state_path, self._new_state())
            group = state.get("floor_groups", {}).get(self.floor_group_id, {})
            if slug not in group.get("floors", {}):
                raise ValueError(f"unknown floor slug: {slug}")
            group["floors"][slug] = True
            state["updated_at"] = time.time()
            atomic_json(self.state_path, state)

    def external_unknown_leases(self) -> list[dict[str, Any]]:
        """Return timed-out jobs and remote jobs whose controller has died."""
        with locked(self.state_path):
            state = read_json(self.state_path, self._new_state())
            self._verify_contract(state)
            self._cleanup(state, time.time())
            return [
                dict(lease) for lease in state.get("leases", {}).values()
                if lease.get("phase") == "external_unknown" or (
                    bool(lease.get("external_ref"))
                    and not self._pid_alive(int(lease.get("pid", 0)))
                )
            ]

    def status(self) -> dict[str, Any]:
        with locked(self.state_path):
            state = read_json(self.state_path, self._new_state())
            now = time.time()
            self._cleanup(state, now)
            active = list(state.get("leases", {}).values())
            result = {
                "pool_id": self.pool_id,
                "floor_group_id": self.floor_group_id,
                "expected_slugs": list(self.expected_slugs),
                "floors": dict(
                    state.get("floor_groups", {})
                    .get(self.floor_group_id, {}).get("floors", {})
                ),
                "floor_groups": dict(state.get("floor_groups", {})),
                "floor_gate": self._floor_gate(state, now),
                "gpu_hours_used": round(
                    float(state.get("gpu_seconds_used", 0.0)) / 3600.0, 4
                ),
                "gpu_hours_reserved": round(
                    sum(
                        float(item.get("estimated_seconds", 0.0))
                        for item in active if item.get("kind") == "gpu"
                    ) / 3600.0,
                    4,
                ),
                "gpu_hours_limit": round(self.gpu_limit_seconds / 3600.0, 4),
                "gpu_inflight": sum(item.get("kind") == "gpu" for item in active),
                "cpu_inflight": sum(item.get("kind") == "cpu" for item in active),
                "waiters": len(state.get("waiters", {})),
                "leases": active,
            }
            state["updated_at"] = time.time()
            atomic_json(self.state_path, state)
            return result
