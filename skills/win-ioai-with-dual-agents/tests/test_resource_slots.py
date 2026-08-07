from __future__ import annotations

import multiprocessing
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from resource_slots import (  # noqa: E402
    classify_status,
    latest_events,
    read_events,
    reconcile_slots,
    transition_slot,
    try_acquire_slot,
)


NOW = datetime(2026, 8, 7, 0, 0, tzinfo=timezone.utc)
DEADLINE = NOW + timedelta(hours=2)


def limits(cpu: int = 5, gpu: int = 2, per_actor: int = 1) -> dict[str, int]:
    return {
        "max_concurrent_cpu": cpu,
        "max_concurrent_gpu": gpu,
        "per_actor_inflight": per_actor,
        "final_priority_window_seconds": 1500,
        "final_waiter_ttl_seconds": 30,
        "orphan_reservation_seconds": 300,
    }


def acquire(
    ledger: Path,
    lease: str,
    actor: str,
    resource: str,
    *,
    priority: str = "normal",
    now: datetime = NOW,
    deadline: datetime = DEADLINE,
    statuses: dict[str, str] | None = None,
    slot_limits: dict[str, int] | None = None,
):
    return try_acquire_slot(
        path=ledger,
        lease_id=lease,
        actor=actor,
        candidate=lease,
        competition="task",
        kernel_ref=f"owner/{lease}",
        resource_class=resource,
        priority=priority,
        contract_sha256="a" * 64,
        limits=slot_limits or limits(),
        deadline=deadline,
        now=now,
        statuses=statuses or {},
    )


def process_acquire(ledger: str, start: multiprocessing.synchronize.Event, index: int, queue) -> None:
    start.wait()
    result = acquire(Path(ledger), f"p{index}", f"actor{index}", "gpu")
    queue.put(result.acquired)


class ResourceSlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.temp.name) / "kernel-slot-ledger.jsonl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_gpu_two_and_cpu_five_caps(self) -> None:
        self.assertTrue(acquire(self.ledger, "g1", "A", "gpu").acquired)
        self.assertTrue(acquire(self.ledger, "g2", "B", "gpu").acquired)
        self.assertFalse(acquire(self.ledger, "g3", "C", "gpu").acquired)

        for index in range(5):
            self.assertTrue(acquire(self.ledger, f"c{index}", f"C{index}", "cpu").acquired)
        self.assertFalse(acquire(self.ledger, "c5", "C5", "cpu").acquired)

    def test_per_actor_limit(self) -> None:
        self.assertTrue(acquire(self.ledger, "one", "CX", "gpu").acquired)
        self.assertFalse(acquire(self.ledger, "two", "CX", "cpu").acquired)

    def test_terminal_releases_capacity(self) -> None:
        self.assertTrue(acquire(self.ledger, "g1", "A", "gpu").acquired)
        self.assertTrue(acquire(self.ledger, "g2", "B", "gpu").acquired)
        decision = acquire(
            self.ledger,
            "g3",
            "C",
            "gpu",
            now=NOW + timedelta(seconds=1),
            statuses={"owner/g1": "terminal", "owner/g2": "active"},
        )
        self.assertTrue(decision.acquired)

    def test_final_waiter_gets_next_free_slot(self) -> None:
        one_gpu = limits(gpu=1)
        final_now = DEADLINE - timedelta(minutes=10)
        self.assertTrue(acquire(self.ledger, "normal1", "A", "gpu", slot_limits=one_gpu).acquired)
        self.assertFalse(
            acquire(
                self.ledger,
                "final1",
                "B",
                "gpu",
                priority="final",
                now=final_now,
                slot_limits=one_gpu,
            ).acquired
        )
        normal = acquire(
            self.ledger,
            "normal2",
            "C",
            "gpu",
            now=final_now + timedelta(seconds=5),
            deadline=DEADLINE + timedelta(hours=2),
            statuses={"owner/normal1": "terminal"},
            slot_limits=one_gpu,
        )
        self.assertFalse(normal.acquired)
        final = acquire(
            self.ledger,
            "final1",
            "B",
            "gpu",
            priority="final",
            now=final_now + timedelta(seconds=6),
            slot_limits=one_gpu,
        )
        self.assertTrue(final.acquired)

    def test_expired_final_waiter_does_not_deadlock_normal(self) -> None:
        one_gpu = limits(gpu=1)
        final_now = DEADLINE - timedelta(minutes=10)
        self.assertTrue(acquire(self.ledger, "normal1", "A", "gpu", slot_limits=one_gpu).acquired)
        self.assertFalse(
            acquire(
                self.ledger,
                "final1",
                "B",
                "gpu",
                priority="final",
                now=final_now,
                slot_limits=one_gpu,
            ).acquired
        )
        normal = acquire(
            self.ledger,
            "normal2",
            "C",
            "gpu",
            now=final_now + timedelta(seconds=31),
            statuses={"owner/normal1": "terminal"},
            slot_limits=one_gpu,
        )
        self.assertTrue(normal.acquired)

    def test_each_event_keeps_its_own_waiter_and_orphan_ttl(self) -> None:
        one_gpu = limits(gpu=1)
        final_now = DEADLINE - timedelta(minutes=10)
        self.assertTrue(acquire(self.ledger, "normal1", "A", "gpu", slot_limits=one_gpu).acquired)
        self.assertFalse(
            acquire(
                self.ledger,
                "final1",
                "B",
                "gpu",
                priority="final",
                now=final_now,
                slot_limits=one_gpu,
            ).acquired
        )
        short_ttl = {**one_gpu, "final_waiter_ttl_seconds": 15, "orphan_reservation_seconds": 60}
        normal = acquire(
            self.ledger,
            "normal2",
            "C",
            "gpu",
            now=final_now + timedelta(seconds=20),
            deadline=DEADLINE + timedelta(hours=2),
            statuses={"owner/normal1": "terminal"},
            slot_limits=short_ttl,
        )
        self.assertFalse(normal.acquired)

        second = Path(self.temp.name) / "orphan-ledger.jsonl"
        self.assertTrue(acquire(second, "lease", "A", "gpu", slot_limits=one_gpu).acquired)
        reconcile_slots(
            second,
            limits=short_ttl,
            statuses={"owner/lease": "absent"},
            now=NOW + timedelta(seconds=61),
        )
        self.assertEqual(latest_events(read_events(second), "lease_id")["lease"]["status"], "slot_reserved")

    def test_active_capacity_policy_mismatch_fails_closed(self) -> None:
        self.assertTrue(acquire(self.ledger, "g1", "A", "gpu").acquired)
        different = {**limits(), "max_concurrent_gpu": 3}
        decision = acquire(self.ledger, "g2", "B", "gpu", slot_limits=different)
        self.assertFalse(decision.acquired)
        self.assertIn("不同容量策略", decision.reason)

    def test_ambiguous_push_needs_two_absence_confirmations(self) -> None:
        self.assertTrue(acquire(self.ledger, "g1", "A", "gpu").acquired)
        transition_slot(self.ledger, "g1", "push_started", NOW)
        reconcile_slots(
            self.ledger,
            limits=limits(),
            statuses={"owner/g1": "absent"},
            now=NOW + timedelta(seconds=61),
        )
        latest = latest_events(read_events(self.ledger), "lease_id")
        self.assertEqual(latest["g1"]["status"], "push_started")
        reconcile_slots(
            self.ledger,
            limits=limits(),
            statuses={"owner/g1": "absent"},
            now=NOW + timedelta(seconds=122),
        )
        latest = latest_events(read_events(self.ledger), "lease_id")
        self.assertEqual(latest["g1"]["status"], "push_started")
        reconcile_slots(
            self.ledger,
            limits=limits(),
            statuses={"owner/g1": "absent"},
            now=NOW + timedelta(seconds=301),
        )
        latest = latest_events(read_events(self.ledger), "lease_id")
        self.assertEqual(latest["g1"]["status"], "remote_absent_once")
        reconcile_slots(
            self.ledger,
            limits=limits(),
            statuses={"owner/g1": "absent"},
            now=NOW + timedelta(seconds=362),
        )
        latest = latest_events(read_events(self.ledger), "lease_id")
        self.assertEqual(latest["g1"]["status"], "released_absent")

    def test_returned_ambiguous_push_uses_two_short_confirmations(self) -> None:
        self.assertTrue(acquire(self.ledger, "g1", "A", "gpu").acquired)
        transition_slot(self.ledger, "g1", "push_ambiguous", NOW)
        reconcile_slots(
            self.ledger,
            limits=limits(),
            statuses={"owner/g1": "absent"},
            now=NOW + timedelta(seconds=61),
        )
        latest = latest_events(read_events(self.ledger), "lease_id")
        self.assertEqual(latest["g1"]["status"], "remote_absent_once")
        reconcile_slots(
            self.ledger,
            limits=limits(),
            statuses={"owner/g1": "absent"},
            now=NOW + timedelta(seconds=122),
        )
        latest = latest_events(read_events(self.ledger), "lease_id")
        self.assertEqual(latest["g1"]["status"], "released_absent")

    def test_status_parser_requires_one_explicit_state(self) -> None:
        self.assertEqual(
            classify_status('{"status":"KernelWorkerStatus.RUNNING","failureMessage":"ERROR text"}'),
            "active",
        )
        self.assertEqual(
            classify_status("KernelWorkerStatus.RUNNING then KernelWorkerStatus.COMPLETE"),
            "unknown",
        )
        self.assertEqual(classify_status("run is not complete yet"), "unknown")
        self.assertEqual(
            classify_status("KernelWorkerStatus.RUNNING\nwarning: log not found", 0),
            "active",
        )
        self.assertEqual(classify_status("kernel not found", 1), "absent")
        self.assertEqual(classify_status("kernel not found", 0), "unknown")

    def test_process_race_never_exceeds_gpu_capacity(self) -> None:
        context = multiprocessing.get_context("fork")
        start = context.Event()
        queue = context.Queue()
        processes = [
            context.Process(target=process_acquire, args=(str(self.ledger), start, index, queue))
            for index in range(6)
        ]
        for process in processes:
            process.start()
        start.set()
        results = [queue.get(timeout=5) for _ in processes]
        for process in processes:
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(sum(results), 2)


if __name__ == "__main__":
    unittest.main()
