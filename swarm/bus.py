"""The blackboard: the only channel roles use to talk to each other.

Roles never call each other directly. Everything goes through files under
``workspace/<slug>/`` so that (a) parallel candidates cannot corrupt shared
state, (b) a crash loses nothing that was already written, and (c) the full
decision trail is on disk for the technical report and the Jury audit.

Concurrency: writes are atomic (temp file + ``os.replace``) and serialised by
an ``fcntl`` lock on a sidecar ``.lock`` file, so several candidate threads —
and even several processes — can append safely.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .schemas import (
    Calibration,
    CandidateState,
    Experiment,
    PlanCard,
    ProbeResult,
    SubmissionRecord,
    TaskCard,
)

_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCAL_LOCKS[key] = lock
        return lock


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialise access to ``path`` across threads and processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _thread_lock(path):
        with open(lock_path, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


class Blackboard:
    """File-backed shared state for one task pod."""

    def __init__(self, workspace: Path):
        self.ws = Path(workspace)
        self.ws.mkdir(parents=True, exist_ok=True)
        (self.ws / "candidates").mkdir(exist_ok=True)

    # ---------------------------------------------------------------- paths
    @property
    def task_card_path(self) -> Path:
        return self.ws / "task_card.json"

    @property
    def plans_path(self) -> Path:
        return self.ws / "plans.jsonl"

    @property
    def ledger_path(self) -> Path:
        return self.ws / "ledger.jsonl"

    @property
    def submissions_path(self) -> Path:
        return self.ws / "submissions.jsonl"

    @property
    def probes_path(self) -> Path:
        return self.ws / "probes.jsonl"

    @property
    def calibration_path(self) -> Path:
        return self.ws / "calibration.json"

    @property
    def events_path(self) -> Path:
        return self.ws / "events.jsonl"

    def candidate_dir(self, candidate_id: str) -> Path:
        d = self.ws / "candidates" / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -------------------------------------------------------------- generic
    def _append(self, path: Path, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with _locked(path):
            with open(path, "a") as fh:
                fh.write(line + "\n")

    def _read_all(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        out: list[dict] = []
        with _locked(path):
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final line can only be the last one; skip it
                    # rather than losing the whole ledger.
                    continue
        return out

    # ------------------------------------------------------------- events
    def event(self, kind: str, **fields: Any) -> None:
        """Append one audit event. Never raises — auditing must not break a run."""
        try:
            self._append(self.events_path, {"t": time.time(), "kind": kind, **fields})
        except Exception:  # pragma: no cover - defensive
            pass

    # ---------------------------------------------------------- task card
    def put_task_card(self, card: TaskCard) -> None:
        with _locked(self.task_card_path):
            _atomic_write(
                self.task_card_path, json.dumps(card.to_dict(), indent=2, ensure_ascii=False)
            )
        self.event("task_card", task_type=card.task_type, metric=card.metric_name)

    def get_task_card(self) -> TaskCard | None:
        if not self.task_card_path.exists():
            return None
        return TaskCard.from_dict(json.loads(self.task_card_path.read_text()))

    # -------------------------------------------------------------- plans
    def add_plan(self, plan: PlanCard) -> None:
        self._append(self.plans_path, plan.to_dict())
        self.event("plan", plan_id=plan.plan_id, family=plan.family, title=plan.title)

    def get_plans(self) -> list[PlanCard]:
        return [PlanCard.from_dict(d) for d in self._read_all(self.plans_path)]

    # --------------------------------------------------------- candidates
    def put_candidate(self, cand: CandidateState) -> None:
        cand.updated_at = time.time()
        path = self.candidate_dir(cand.candidate_id) / "state.json"
        with _locked(path):
            _atomic_write(path, json.dumps(cand.to_dict(), indent=2, ensure_ascii=False))

    def get_candidate(self, candidate_id: str) -> CandidateState | None:
        path = self.ws / "candidates" / candidate_id / "state.json"
        if not path.exists():
            return None
        return CandidateState.from_dict(json.loads(path.read_text()))

    def get_candidates(self) -> list[CandidateState]:
        out: list[CandidateState] = []
        root = self.ws / "candidates"
        if not root.exists():
            return out
        for d in sorted(root.iterdir()):
            p = d / "state.json"
            if p.exists():
                try:
                    out.append(CandidateState.from_dict(json.loads(p.read_text())))
                except (json.JSONDecodeError, TypeError):
                    continue
        return out

    def live_candidates(self) -> list[CandidateState]:
        return [c for c in self.get_candidates() if c.status not in ("retired", "failed")]

    def best_candidate(self) -> CandidateState | None:
        scored = [c for c in self.get_candidates() if c.local_score is not None]
        if not scored:
            return None
        return max(scored, key=lambda c: c.local_score or float("-inf"))

    # ------------------------------------------------------------- ledger
    def add_experiment(self, exp: Experiment) -> None:
        self._append(self.ledger_path, exp.to_dict())

    def get_experiments(self) -> list[Experiment]:
        return [Experiment.from_dict(d) for d in self._read_all(self.ledger_path)]

    # -------------------------------------------------------- submissions
    def add_submission(self, rec: SubmissionRecord) -> None:
        self._append(self.submissions_path, rec.to_dict())
        self.event(
            "submission",
            sub_id=rec.sub_id,
            lane=rec.lane,
            candidate_id=rec.candidate_id,
            local_score=rec.local_score,
        )

    def update_submission(self, sub_id: str, **fields: Any) -> None:
        """Rewrite the log with one record updated (scores arrive asynchronously)."""
        with _locked(self.submissions_path):
            if not self.submissions_path.exists():
                return
            rows = []
            for line in self.submissions_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("sub_id") == sub_id:
                    d.update(fields)
                rows.append(d)
            _atomic_write(
                self.submissions_path,
                "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
            )

    def get_submissions(self) -> list[SubmissionRecord]:
        return [SubmissionRecord.from_dict(d) for d in self._read_all(self.submissions_path)]

    def scored_submissions(self) -> list[SubmissionRecord]:
        return [s for s in self.get_submissions() if s.lb_score is not None]

    # ------------------------------------------------------------- probes
    def add_probe(self, probe: ProbeResult) -> None:
        self._append(self.probes_path, probe.to_dict())
        self.event("probe", probe_id=probe.probe_id, probe_kind=probe.probe_kind)

    def get_probes(self) -> list[ProbeResult]:
        return [ProbeResult.from_dict(d) for d in self._read_all(self.probes_path)]

    # -------------------------------------------------------- calibration
    def put_calibration(self, cal: Calibration) -> None:
        with _locked(self.calibration_path):
            _atomic_write(
                self.calibration_path, json.dumps(cal.to_dict(), indent=2, ensure_ascii=False)
            )

    def get_calibration(self) -> Calibration:
        if not self.calibration_path.exists():
            return Calibration()
        return Calibration.from_dict(json.loads(self.calibration_path.read_text()))

    # ------------------------------------------------------------ summary
    def snapshot(self) -> dict:
        """Compact state summary — this is what the Manager sees each step."""
        card = self.get_task_card()
        cands = self.get_candidates()
        subs = self.get_submissions()
        cal = self.get_calibration()
        return {
            "task_type": card.task_type if card else "unknown",
            "metric": card.metric_name if card else "",
            "n_plans": len(self.get_plans()),
            "candidates": [
                {
                    "id": c.candidate_id,
                    "family": c.family,
                    "status": c.status,
                    "local_score": c.local_score,
                    "local_std": c.local_std,
                    "last_error": c.last_error[:200],
                }
                for c in cands
            ],
            "n_submissions": len(subs),
            "submissions_by_lane": {
                lane: sum(1 for s in subs if s.lane == lane)
                for lane in ("floor", "probe", "milestone", "final")
            },
            "best_local": (self.best_candidate().local_score if self.best_candidate() else None),
            "best_lb": max(
                (s.lb_score for s in subs if s.lb_score is not None), default=None
            ),
            "calibration": {
                "n_points": cal.n_points,
                "noise_band": cal.noise_band,
                "gap": cal.gap,
                "warning": cal.warning,
            },
            "probe_findings": [p.inference for p in self.get_probes() if p.inference][-8:],
        }
