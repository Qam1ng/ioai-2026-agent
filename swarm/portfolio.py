"""Cross-problem allocation for a competition day.

Three problems share one six-hour window, one Kaggle account, and therefore
one 30-hour GPU quota. Pods run concurrently; this layer decides who gets the
scarce resources.

The scoring rule drives the policy. Scores are normalised within the AI track
per task and ranked by percentile, with all three tasks weighted equally, so:

* **Never abandon a problem.** A zero is the most expensive outcome available,
  and it cannot be bought back by excelling elsewhere.
* **After every problem has a floor, feed the margin.** Because scores are
  normalised against the best AI entry on that task, a problem where everyone
  struggles rewards a small absolute gain far more than a problem where the
  field is already near the ceiling.

Reallocation is deliberately deterministic. An LLM allocator was considered and
rejected: the decision is a two-line rule over numbers we already track, and a
mis-firing allocator can starve a problem, which is unrecoverable.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import PodBudget, QuotaPool
from .bus import Blackboard
from .config import SwarmConfig
from .pod import PodResult, TaskPod


@dataclass
class DayPlan:
    """One competition day: several slugs sharing a window and an account."""

    slugs: list[str]
    window_min: float = 360.0
    gpu_quota_hours: float = 30.0
    workspace_root: Path = Path("workspace")

    def per_problem_gpu_hours(self) -> float:
        return self.gpu_quota_hours / max(1, len(self.slugs))


@dataclass
class PortfolioResult:
    results: dict[str, PodResult] = field(default_factory=dict)
    reallocations: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def all_have_submission(self) -> bool:
        return all(r.n_submissions > 0 for r in self.results.values())


class Portfolio:
    """Runs the pods for one day and rebalances between them."""

    def __init__(
        self,
        plan: DayPlan,
        cfg_for: Any,
        broker_for: Any,
        review_interval_s: float = 900.0,
    ):
        """``cfg_for(slug) -> SwarmConfig`` and ``broker_for(slug, bb, budget, cfg, quota)``
        are factories so the caller controls construction (and tests can inject fakes)."""
        self.plan = plan
        self.cfg_for = cfg_for
        self.broker_for = broker_for
        self.review_interval_s = review_interval_s
        self.quota = QuotaPool(
            plan.workspace_root / "gpu_quota.json", limit_hours=plan.gpu_quota_hours
        )
        self.pods: dict[str, TaskPod] = {}
        self.result = PortfolioResult()
        self._lock = threading.Lock()

    # ------------------------------------------------------------- build
    def _build_pod(self, slug: str) -> TaskPod:
        cfg: SwarmConfig = self.cfg_for(slug)
        ws = self.plan.workspace_root / slug
        bb = Blackboard(ws)
        budget = PodBudget(
            ws / "budget.json",
            deadline_s=self.plan.window_min * 60,
            max_submissions=cfg.submit.max_submissions,
            max_cost_usd=cfg.max_cost_usd,
            quota=self.quota,
        )
        broker = self.broker_for(slug, bb, budget, cfg, self.quota)
        return TaskPod(cfg, bb, budget, self.quota, broker)

    # -------------------------------------------------------- allocation
    def review(self) -> dict | None:
        """Rebalance candidate slots between pods. Returns the decision, if any."""
        with self._lock:
            states = []
            for slug, pod in self.pods.items():
                subs = pod.bb.get_submissions()
                best = pod.bb.best_candidate()
                states.append(
                    {
                        "slug": slug,
                        "has_floor": any(s.lane == "floor" for s in subs),
                        "n_submissions": len(subs),
                        "best_local": best.local_score if best else None,
                        "slots": pod.cfg.parallel.max_candidates,
                        "running": pod._running(),
                    }
                )

            unfloored = [s for s in states if not s["has_floor"]]
            if unfloored:
                # Insurance first. Nothing else is worth a GPU-hour until every
                # problem is guaranteed a non-zero score.
                decision = {
                    "t": time.time(),
                    "policy": "floor_first",
                    "starved": [s["slug"] for s in unfloored],
                }
                for s in unfloored:
                    pod = self.pods[s["slug"]]
                    pod.cfg.parallel.max_candidates = min(
                        pod.cfg.parallel.max_candidates_hard,
                        pod.cfg.parallel.max_candidates + 1,
                    )
                self.result.reallocations.append(decision)
                return decision

            # Everyone has insurance: bias slots toward the problem with the
            # most headroom, approximated by the lowest best-local score.
            scored = [s for s in states if s["best_local"] is not None]
            if len(scored) < 2:
                return None
            weakest = min(scored, key=lambda s: s["best_local"])
            strongest = max(scored, key=lambda s: s["best_local"])
            if weakest["slug"] == strongest["slug"]:
                return None

            wpod = self.pods[weakest["slug"]]
            spod = self.pods[strongest["slug"]]
            if spod.cfg.parallel.max_candidates <= 2:
                return None
            spod.cfg.parallel.max_candidates -= 1
            wpod.cfg.parallel.max_candidates = min(
                wpod.cfg.parallel.max_candidates_hard, wpod.cfg.parallel.max_candidates + 1
            )
            decision = {
                "t": time.time(),
                "policy": "feed_the_margin",
                "from": strongest["slug"],
                "to": weakest["slug"],
                "gpu_hours_remaining": self.quota.status()["gpu_hours_remaining"],
            }
            self.result.reallocations.append(decision)
            return decision

    # --------------------------------------------------------------- run
    def run(self) -> PortfolioResult:
        threads: list[threading.Thread] = []
        for slug in self.plan.slugs:
            try:
                self.pods[slug] = self._build_pod(slug)
            except Exception as exc:
                self.result.errors.append(f"{slug}: pod construction failed: {exc}")

        def _run_pod(slug: str) -> None:
            try:
                self.result.results[slug] = self.pods[slug].run()
            except Exception as exc:
                self.result.errors.append(f"{slug}: {type(exc).__name__}: {exc}")

        for slug in self.pods:
            t = threading.Thread(target=_run_pod, args=(slug,), name=f"pod-{slug}", daemon=True)
            t.start()
            threads.append(t)

        deadline = time.time() + self.plan.window_min * 60
        while any(t.is_alive() for t in threads) and time.time() < deadline:
            time.sleep(min(self.review_interval_s, max(1.0, deadline - time.time())))
            try:
                self.review()
            except Exception as exc:
                self.result.errors.append(f"review failed: {exc}")

        for t in threads:
            t.join(timeout=60)
        return self.result
