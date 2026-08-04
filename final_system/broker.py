from __future__ import annotations

import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Protocol

from .io import append_jsonl, atomic_json, locked, read_json
from .registry import CandidateRegistry


class SubmissionAdapter(Protocol):
    def submit(self, record: dict, submission_id: str, submission_class: str): ...
    def scores(self) -> dict[str, float]: ...


class SubmissionBroker:
    """Persistent 50-slot policy and the only caller of a Kaggle adapter."""

    def __init__(
        self,
        root: Path,
        *,
        registry: CandidateRegistry,
        adapter: SubmissionAdapter,
        max_submissions: int,
        final_reserve: int,
        initial_calibrations: int,
        final_start_fraction: float,
        anti_monopoly_fraction: float,
        min_local_gain: float,
        direction: str,
    ):
        self.root = Path(root)
        self.registry = registry
        self.adapter = adapter
        self.state_path = self.root / "broker_state.json"
        self.status_path = self.root / "BROKER_STATUS.json"
        self.events_path = self.root / "broker_events.jsonl"
        self.feedback_root = self.root / "feedback"
        self.feedback_root.mkdir(parents=True, exist_ok=True)
        self.max_submissions = max_submissions
        self.final_reserve = final_reserve
        self.initial_calibrations = initial_calibrations
        self.final_start_fraction = final_start_fraction
        self.anti_monopoly_fraction = anti_monopoly_fraction
        self.min_local_gain = min_local_gain
        self.direction = direction
        if not self.state_path.exists():
            atomic_json(
                self.state_path,
                {
                    "schema_version": 1,
                    "created_at": time.time(),
                    "max_submissions": max_submissions,
                    "final_reserve": final_reserve,
                    "initial_calibrations": initial_calibrations,
                    "submissions": [],
                },
            )
        else:
            state = self._load()
            immutable = (
                state.get("max_submissions"),
                state.get("final_reserve"),
                state.get("initial_calibrations"),
            )
            expected = (max_submissions, final_reserve, initial_calibrations)
            if immutable != expected:
                raise ValueError("existing Broker quota contract differs from config")
        self.write_status()

    def _load(self) -> dict:
        return read_json(self.state_path, {"submissions": []})

    @staticmethod
    def _consumed(row: dict) -> bool:
        return bool(row.get("result", {}).get("consumed"))

    def submissions(self) -> list[dict]:
        return list(self._load().get("submissions", []))

    def consumed(self) -> list[dict]:
        return [row for row in self.submissions() if self._consumed(row)]

    @staticmethod
    def _reserved(row: dict) -> bool:
        return row.get("result", {}).get("status") in {"reserved", "ambiguous"}

    def committed(self) -> list[dict]:
        """Slots already consumed or held by an in-flight/crash-ambiguous call."""
        return [
            row for row in self.submissions()
            if self._consumed(row) or self._reserved(row)
        ]

    def reservable(self) -> int:
        return max(0, self.max_submissions - len(self.committed()))

    def remaining(self) -> int:
        return max(0, self.max_submissions - len(self.consumed()))

    def write_status(self, *, fraction_elapsed: float | None = None) -> dict:
        consumed = self.consumed()
        by_class = Counter(row["submission_class"] for row in consumed)
        by_source = Counter(row["source_lane"] for row in consumed)
        status = {
            "schema_version": 1,
            "updated_at": time.time(),
            "max_submissions": self.max_submissions,
            "consumed": len(consumed),
            "remaining": self.remaining(),
            "inflight_or_ambiguous": sum(
                self._reserved(row) for row in self.submissions()
            ),
            "reservable": self.reservable(),
            "final_reserve": self.final_reserve,
            "final_reserve_remaining": max(
                0, self.final_reserve - by_class.get("final", 0)
            ),
            "by_submission_class": dict(by_class),
            "by_source_lane": dict(by_source),
            "fraction_elapsed": fraction_elapsed,
            "submission_authority": "final_system.SubmissionBroker",
        }
        atomic_json(self.status_path, status)
        return status

    def _score(self, record: dict) -> float | None:
        value = record.get("evaluation", {}).get("mean")
        return float(value) if value is not None else None

    def _sort(self, records: list[dict]) -> list[dict]:
        sign = 1 if self.direction == "maximize" else -1
        def key(record: dict) -> tuple[bool, float, float]:
            score = self._score(record)
            return (
                score is not None,
                sign * score if score is not None else float("-inf"),
                -float(record.get("created_at", 0)),
            )
        return sorted(
            records,
            key=key,
            reverse=True,
        )

    def _eligible(self) -> list[dict]:
        attempts = {row["candidate_id"] for row in self.submissions()}
        return [
            record for record in self.registry.records()
            if record.get("status") == "eligible"
            and record["candidate_id"] not in attempts
            and record.get("format", {}).get("valid")
        ]

    def _best_submitted_local(self) -> float | None:
        values = [
            row.get("local_score") for row in self.consumed()
            if row.get("local_score") is not None
        ]
        if not values:
            return None
        return (max(values) if self.direction == "maximize" else min(values))

    def _improves(self, score: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        return (
            score > incumbent + self.min_local_gain
            if self.direction == "maximize"
            else score < incumbent - self.min_local_gain
        )

    def _anti_monopoly_choice(self, ranked: list[dict]) -> dict | None:
        if not ranked:
            return None
        nonfloor = [
            row for row in self.committed()
            if row["submission_class"] != "floor"
        ]
        counts = Counter(row["source_lane"] for row in nonfloor)
        for record in ranked:
            lane = record["source_lane"]
            total_after = len(nonfloor) + 1
            share = (counts[lane] + 1) / total_after
            other_ready = any(
                item["source_lane"] != lane
                and counts[item["source_lane"]] < counts[lane]
                for item in ranked
            )
            if not other_ready or share <= self.anti_monopoly_fraction:
                return record
        return None

    def select(self, *, fraction_elapsed: float, force_final: bool = False) -> tuple[dict, str] | None:
        if self.reservable() <= 0:
            return None
        eligible = self._sort(self._eligible())
        if not eligible:
            return None
        allocated = self.committed()
        if not allocated:
            return eligible[0], "floor"

        final_phase = force_final or fraction_elapsed >= self.final_start_fraction
        if final_phase:
            finals = sum(row["submission_class"] == "final" for row in allocated)
            if finals >= self.final_reserve:
                return None
            scored = [record for record in eligible if self._score(record) is not None]
            return ((scored or eligible)[0], "final")

        # 结束前保留 final_reserve；探索阶段不会借用它。
        if self.reservable() <= self.final_reserve:
            return None
        calibrated = {
            row["source_lane"] for row in allocated
            if row["submission_class"] == "calibration"
        }
        calibration_count = sum(
            row["submission_class"] == "calibration" for row in allocated
        )
        if calibration_count < self.initial_calibrations:
            choices = [
                record for record in eligible
                if record["source_lane"] not in calibrated
                and self._score(record) is not None
            ]
            if choices:
                return choices[0], "calibration"

        incumbent = self._best_submitted_local()
        milestones = [
            record for record in eligible
            if self._score(record) is not None
            and self._improves(self._score(record), incumbent)  # type: ignore[arg-type]
        ]
        choice = (
            self._anti_monopoly_choice(milestones)
            if fraction_elapsed < 0.5 else
            (milestones[0] if milestones else None)
        )
        return (choice, "milestone") if choice else None

    def tick(self, *, fraction_elapsed: float, force_final: bool = False) -> dict | None:
        # 选择和预留必须共用一把锁；两个后台 kernel 否则可能选择同一候选，
        # 或在只剩一个额度时同时越过检查。
        with locked(self.state_path):
            selected = self.select(
                fraction_elapsed=fraction_elapsed, force_final=force_final
            )
            if selected is None:
                attempt = None
            else:
                record, submission_class = selected
                submission_id = f"sub-{uuid.uuid4().hex[:10]}"
                attempt = {
                    "submission_id": submission_id,
                    "candidate_id": record["candidate_id"],
                    "source_lane": record["source_lane"],
                    "submission_class": submission_class,
                    "local_score": self._score(record),
                    "reserved_at": time.time(),
                    "result": {"consumed": False, "status": "reserved"},
                }
            state = self._load()
            if attempt is not None:
                state["submissions"].append(attempt)
                atomic_json(self.state_path, state)
        if attempt is None:
            self.write_status(fraction_elapsed=fraction_elapsed)
            return None
        append_jsonl(self.events_path, {"event": "reserved", **attempt})

        try:
            result = self.adapter.submit(record, submission_id, submission_class)
            result_value = result.to_dict()
        except Exception as exc:  # noqa: BLE001
            result_value = {
                "consumed": False,
                "status": "ambiguous",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        with locked(self.state_path):
            state = self._load()
            for row in state["submissions"]:
                if row["submission_id"] == submission_id:
                    row["result"] = result_value
                    row["completed_at"] = time.time()
                    break
            atomic_json(self.state_path, state)
        if result_value.get("consumed"):
            self.registry.mark_submitted(record["candidate_id"])
        append_jsonl(
            self.events_path,
            {"event": "submission_result", "submission_id": submission_id, **result_value},
        )
        self.write_status(fraction_elapsed=fraction_elapsed)
        return next(
            row for row in self.submissions() if row["submission_id"] == submission_id
        )

    def refresh_scores(self) -> int:
        scores = self.adapter.scores()
        if not scores:
            return 0
        updates: list[dict] = []
        with locked(self.state_path):
            state = self._load()
            for row in state["submissions"]:
                score = scores.get(row["submission_id"])
                if score is None or row.get("leaderboard_score") == score:
                    continue
                row["leaderboard_score"] = score
                row["scored_at"] = time.time()
                if row.get("result", {}).get("status") == "ambiguous":
                    row["result"]["consumed"] = True
                    row["result"]["status"] = "submitted_reconciled"
                updates.append(row.copy())
            if updates:
                atomic_json(self.state_path, state)
        for row in updates:
            feedback = {
                "timestamp": row["scored_at"],
                "submission_id": row["submission_id"],
                "candidate_id": row["candidate_id"],
                "submission_class": row["submission_class"],
                "local_score": row.get("local_score"),
                "leaderboard_score": row["leaderboard_score"],
            }
            append_jsonl(
                self.feedback_root / f"{row['source_lane']}.jsonl", feedback
            )
            append_jsonl(self.events_path, {"event": "leaderboard_score", **feedback})
        self.write_status()
        return len(updates)
