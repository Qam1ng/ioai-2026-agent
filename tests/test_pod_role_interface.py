"""Pin the pod <-> role parsing contract.

Two rehearsals ran on an empty task card because the pod probed for a generic
``role.parse`` that no role implements — every Profiler and Designer report was
silently discarded while the run *looked* like it was thinking. These tests
fail loudly if the interface drifts again, in either direction.
"""

from __future__ import annotations

from swarm.budget import PodBudget, QuotaPool
from swarm.bus import Blackboard
from swarm.config import SwarmConfig
from swarm.pod import TaskPod
from swarm.roles import ROLE_CLASSES
from swarm.roles.base import RoleResult


PROFILER_REPORT = {
    "title": "Radar semantic segmentation",
    "task_type": "supervised",
    "metric_name": "RadarMetric",
    "metric_description": "per-pixel multi-class score",
    "submission_format": "filename,pixel_0..pixel_9049",
    "data_summary": "1000 train (7,50,181), 500 test (6,50,181)",
    "constraints": [{"kind": "must", "quote": "predictions shifted back by -1"}],
    "grouping_variable": "filename",
    "routing_confidence": 0.9,
}

DESIGNER_REPORT = {
    "plans": [
        {
            "title": "constant background floor",
            "family": "floor",
            "architecture": "predict class 0 everywhere",
            "accelerator": "cpu",
        },
        {
            "title": "small unet",
            "family": "cnn_segmentation",
            "architecture": "3-level unet",
            "accelerator": "t4",
        },
    ]
}


def make_pod(tmp_path, roles: dict) -> TaskPod:
    cfg = SwarmConfig()
    cfg.slug = "iface-test"
    cfg.dry_run = True
    bb = Blackboard(tmp_path / "ws")
    quota = QuotaPool(tmp_path / "q.json", limit_hours=1.0)
    budget = PodBudget(tmp_path / "b.json", deadline_s=3600, quota=quota)

    class NullBroker:
        pass

    return TaskPod(cfg, bb, budget, quota, NullBroker(), roles=roles)


class FakeRole:
    """Stands in for any role: returns a canned report from run()."""

    def __init__(self, report: dict, parser):
        self._report = report
        self._parser = parser

    def run(self, instructions, context=None):
        return RoleResult(role="fake", ok=True, report=self._report)

    def __getattr__(self, item):
        # Delegate parse_card / parse_plans / parse_action to the real class,
        # so the test exercises the genuine parsing code the pod will call.
        return getattr(self._parser, item)


def test_profiler_report_actually_lands_in_the_task_card(tmp_path):
    pod = make_pod(
        tmp_path, {"profiler": FakeRole(PROFILER_REPORT, ROLE_CLASSES["profiler"])}
    )
    card = pod.ensure_task_card()
    assert card.task_type == "supervised"
    assert card.metric_name == "RadarMetric"
    assert card.constraints and card.constraints[0].quote.startswith("predictions")
    # And it must be persisted, not just returned.
    assert pod.bb.get_task_card().metric_name == "RadarMetric"


def test_designer_report_actually_lands_in_plans(tmp_path):
    pod = make_pod(
        tmp_path, {"designer": FakeRole(DESIGNER_REPORT, ROLE_CLASSES["designer"])}
    )
    plans = pod.ensure_plans(2)
    assert len(plans) == 2
    assert {p.family for p in plans} == {"floor", "cnn_segmentation"}


def test_every_role_class_has_the_method_the_pod_calls():
    """The pod calls exactly these; a rename on either side must fail CI."""
    assert hasattr(ROLE_CLASSES["profiler"], "parse_card")
    assert hasattr(ROLE_CLASSES["designer"], "parse_plans")
    assert hasattr(ROLE_CLASSES["manager"], "parse_action")


def test_garbage_reports_degrade_without_crashing(tmp_path):
    pod = make_pod(tmp_path, {"profiler": FakeRole({}, ROLE_CLASSES["profiler"])})
    card = pod.ensure_task_card()
    assert card.task_type == "unknown"
    assert pod._errors, "an unusable card must be recorded as an error, not ignored"
