from __future__ import annotations

import hashlib
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, Protocol

from search_system.ioai_agent_system.redaction import redact_value

from .io import append_jsonl, atomic_json, canonical, locked, read_json
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
        manager_enabled: bool = False,
        recommendation_path: Path | None = None,
        calibration_path: Path | None = None,
        manager_fallback_seconds: float = 0.0,
        max_retryable_attempts: int = 3,
        retry_backoff_seconds: float = 60.0,
        floor_settle_seconds: float = 0.0,
        floor_settle_candidates: int = 0,
        floor_done: Callable[[], bool] | None = None,
    ):
        self.root = Path(root)
        self.registry = registry
        self.adapter = adapter
        # Cross-run view of the day's floor obligation; see DayResourceGate.
        self.floor_done = floor_done
        self.state_path = self.root / "broker_state.json"
        self.status_path = self.root / "BROKER_STATUS.json"
        self.events_path = self.root / "broker_events.jsonl"
        self.feedback_root = self.root / "feedback"
        self.feedback_root.mkdir(parents=True, exist_ok=True)
        # Set by the controller: HearSay's shared blackboard. A leaderboard
        # score delivered only through feedback/hearsay.jsonl reaches each
        # solver once, in private, and is gone after a session restart (the
        # per-solver cursor has already advanced). On the board it is a fact:
        # hash-chained, replayed to a revived solver, citable by claim and
        # adoption, and visible to the evaluator that has to calibrate the
        # local ruler against it.
        self.hearsay_board_path: Path | None = None
        self.max_submissions = max_submissions
        self.final_reserve = final_reserve
        self.initial_calibrations = initial_calibrations
        self.final_start_fraction = final_start_fraction
        self.anti_monopoly_fraction = anti_monopoly_fraction
        self.min_local_gain = min_local_gain
        self.direction = direction
        self.manager_enabled = manager_enabled
        self.recommendation_path = Path(recommendation_path) \
            if recommendation_path else None
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.manager_fallback_seconds = manager_fallback_seconds
        self.max_retryable_attempts = max(1, int(max_retryable_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        # Floor is the first submission and does not compare scores — it just
        # gets *something* legal onto the board. On the live run it fired at
        # minute 11 on the best-so-far 0.906 and occupied a GPU slot; the 0.934
        # arrived at minute 15 and had to wait. Give the ruler a moment to
        # settle so the floor is chosen from a real field, but never past a
        # fraction of the window (a short task must still floor early), and
        # never wait for candidates that will not come.
        #
        # Default 0: the Broker is a policy object, and a delay that only the
        # process environment knows about is invisible to config and to every
        # test of unrelated behaviour. The controller injects the real window
        # from [run] like every other knob.
        self._t_start = time.monotonic()
        self.floor_settle_seconds = max(0.0, float(floor_settle_seconds))
        self.floor_settle_candidates = max(0, int(floor_settle_candidates))
        self.manager_context_state_path = (
            self.root / "selection_manager" / "context_state.json"
        )
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

    def set_direction(self, direction: str) -> None:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("resolved metric direction must be maximize or minimize")
        self.direction = direction
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
        return row.get("result", {}).get("status") in {
            "reserved", "ambiguous", "external_running",
            "kernel_complete_pending_output",
        }

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
            "resource_deferred_events": sum(
                row.get("result", {}).get("status") == "resource_deferred"
                for row in self.submissions()
            ),
            "kernel_tail_pending": sum(
                row.get("result", {}).get("status")
                == "kernel_complete_pending_output"
                for row in self.submissions()
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
            "metric_direction": self.direction,
        }
        atomic_json(self.status_path, status)
        return status

    def _score(self, record: dict) -> float | None:
        value = record.get("evaluation", {}).get("mean")
        return float(value) if value is not None else None

    def _selection_score(
        self, record: dict, calibration: dict | None = None,
    ) -> float | None:
        local = self._score(record)
        if local is None:
            return None
        calibration = calibration or {}
        if calibration.get("rank_inverted"):
            # The ruler ranks backwards against the only ground truth we have.
            # Ordering by it would keep handing the broker the worst candidate
            # first (dryrun8 submitted the lowest public score of the run
            # precisely because it had the highest local one). Flatten the
            # local component so the remaining tie-breakers — recency and
            # method diversity — decide, and let the manager see the flag.
            return 0.0
        prediction = calibration.get("candidate_predictions", {}).get(
            record["candidate_id"], {}
        )
        if not calibration.get("active"):
            value = prediction.get("predicted_public_score")
            try:
                return float(value) if value is not None else local
            except (TypeError, ValueError):
                return local
        value = prediction.get("adjusted_score")
        try:
            return float(value) if value is not None else local
        except (TypeError, ValueError):
            return local

    def _feedback_packet(self, row: dict, event: str) -> dict:
        record = self.registry.get(row["candidate_id"]) or {}
        evaluation = record.get("evaluation") or {}
        local = row.get("local_score")
        leaderboard = row.get("leaderboard_score")
        residual = None
        if local is not None and leaderboard is not None:
            residual = float(leaderboard) - float(local)
        calibration = read_json(self.calibration_path, {}) \
            if self.calibration_path else {}
        return redact_value({
            "schema_version": 1,
            "event": event,
            "timestamp": time.time(),
            "submission_id": row.get("submission_id"),
            "candidate_id": row.get("candidate_id"),
            "source_lane": row.get("source_lane"),
            "submission_class": row.get("submission_class"),
            "selection_reason": row.get("selection_reason"),
            "candidate": {
                "purpose": record.get("purpose"),
                "parent_id": record.get("parent_id"),
                "accelerator": record.get("accelerator"),
                "submission_mode": record.get("submission_mode"),
            },
            "local_evaluation": {
                "mean": evaluation.get("mean"),
                "std": evaluation.get("std"),
                "pooled": evaluation.get("pooled"),
                "per_fold": evaluation.get("per_fold"),
                "contract_sha256": evaluation.get("contract_sha256"),
            },
            "leaderboard": {
                "public_score": leaderboard,
                "local_public_residual": residual,
            },
            "execution": row.get("result"),
            "calibration": {
                "version": calibration.get("version"),
                "active": calibration.get("active", False),
                "candidate_prediction": (
                    calibration.get("candidate_predictions", {}).get(
                        row.get("candidate_id")
                    ) if calibration.get("active") else None
                ),
            },
            "quota": {
                "remaining": self.remaining(),
                "reservable": self.reservable(),
                "final_reserve": self.final_reserve,
            },
        })

    #: Lanes whose feedback files exist for the whole run. global.jsonl is
    #: written but no lane reads it — each one only symlinks its own file.
    FEEDBACK_LANES = ("hearsay", "codex", "claude")

    def _emit_feedback(self, row: dict, event: str) -> None:
        packet = self._feedback_packet(row, event)
        append_jsonl(self.feedback_root / "global.jsonl", packet)
        append_jsonl(
            self.feedback_root / f"{row['source_lane']}.jsonl", packet
        )
        append_jsonl(self.events_path, {"event": event, **packet})
        if event == "leaderboard_score":
            self._post_score_to_board(row)
        self._broadcast_kernel_failure(row, packet)

    def _kernel_failure_detail(self, row: dict) -> str:
        result = row.get("result") or {}
        if result.get("consumed"):
            return ""
        detail = str(result.get("detail") or "")
        return detail if "kernel error" in detail.lower() else ""

    def _broadcast_kernel_failure(self, row: dict, packet: dict) -> None:
        """Tell every lane when a kernel package fails, not just its author.

        A packaging mistake is infrastructure, not strategy: the same
        "wheels not found under /kaggle/input" cost codex a push in two
        separate runs, and hearsay and claude would each have had to hit the
        identical wall on their own to learn it. The score of a candidate is
        private to the lane that built it; the reason a kernel would not start
        is common ground.
        """
        detail = self._kernel_failure_detail(row)
        if not detail:
            return
        author = row.get("source_lane")
        note = dict(packet)
        note["event"] = "peer_kernel_failure"
        hint = ""
        if "not found" in detail.lower() and "wheel" in detail.lower():
            hint = ("  诊断：wheels not found 几乎总是 dataset_sources 的 "
                    "owner/slug 写错——Kaggle 对不存在的数据源【静默忽略】，"
                    "挂载点为空，错误要到运行时才暴露。官方 wheel 数据集是 "
                    "`kamalkhan/ioai-2026-wheel-dataset`。")
        note["note"] = (
            f"另一条路线（{author}）的 kernel 启动失败。同样的打包错误会让你的"
            f"推送也失败——先对照检查你的 kernel-metadata.json 与脚本路径。{hint}"
        )
        for lane in self.FEEDBACK_LANES:
            if lane == author:
                continue
            append_jsonl(self.feedback_root / f"{lane}.jsonl", note)
        self._post_failure_to_board(author, detail)

    def _post_failure_to_board(self, author: str, detail: str) -> None:
        board_path = self.hearsay_board_path
        if not board_path or not Path(board_path).parent.is_dir():
            return
        head = " ".join(detail.split())[:180]
        text = (f"{author} 路线的 kernel 启动失败（额度未消耗）：{head}。"
                f"打包错误是共通的，推送前对照检查自己的 kernel 包。")
        try:
            import native.facts as facts
            board = facts.Board(Path(board_path),
                                Path(board_path).with_name("day_facts.jsonl"))
            board.post("failure", text, src="harness", layer="task")
        except Exception:  # noqa: BLE001 - the board is advisory, never fatal
            pass

    def _post_score_to_board(self, row: dict) -> None:
        """Publish a scored submission as a fact on HearSay's board."""
        board_path = self.hearsay_board_path
        if not board_path or not Path(board_path).parent.is_dir():
            return
        leaderboard = row.get("leaderboard_score")
        if leaderboard is None:
            return
        local = row.get("local_score")
        residual = (f"{float(leaderboard) - float(local):+.4f}"
                    if local is not None else "n/a")
        text = (
            f"提交 {row.get('candidate_id')}（{row.get('source_lane')} 路线，"
            f"{row.get('submission_class')}）榜分 {leaderboard}；"
            f"本地 {local if local is not None else 'n/a'}，残差 {residual}。"
        )
        try:
            import native.facts as facts  # local import: native is optional
            board = facts.Board(Path(board_path),
                                Path(board_path).with_name("day_facts.jsonl"))
            board.post(
                "result", text, src="harness", layer="task",
                evidence_refs=[f"control/submissions/{row.get('submission_id')}"],
            )
        except Exception:  # noqa: BLE001 - the board is advisory, never fatal
            pass

    def _sort(
        self, records: list[dict], calibration: dict | None = None,
    ) -> list[dict]:
        sign = -1 if self.direction == "minimize" else 1
        def key(record: dict) -> tuple[bool, float, float]:
            score = self._selection_score(record, calibration)
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
        now = time.time()
        attempts: dict[str, list[dict]] = {}
        for row in self.submissions():
            attempts.setdefault(row["candidate_id"], []).append(row)

        def blocked(candidate_id: str) -> bool:
            rows = attempts.get(candidate_id, [])
            retryable = 0
            for row in rows:
                result = row.get("result", {})
                status = result.get("status")
                if self._consumed(row) or status in {
                    "reserved", "ambiguous", "external_running", "rejected",
                    "kernel_complete_pending_output", "execution_error",
                    "resource_exhausted", "deadline_blocked",
                }:
                    return True
                if status == "resource_deferred":
                    if float(result.get("retry_after", 0)) > now:
                        return True
                    continue
                if status == "retryable":
                    retryable += 1
                    if float(result.get("retry_after", 0)) > now:
                        return True
            return retryable >= self.max_retryable_attempts

        return [
            record for record in self.registry.records()
            if record.get("status") == "eligible"
            and not blocked(record["candidate_id"])
            and record.get("format", {}).get("valid")
            and record.get("kernel_format", {}).get("valid", True)
        ]

    def _best_submitted_score(self, calibration: dict) -> float | None:
        values: list[float] = []
        for row in self.consumed():
            record = self.registry.get(row["candidate_id"])
            value = (
                self._selection_score(record, calibration)
                if record is not None else row.get("local_score")
            )
            if value is not None:
                values.append(float(value))
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

    def _anti_monopoly_options(self, ranked: list[dict]) -> list[dict]:
        if not ranked:
            return []
        nonfloor = [
            row for row in self.committed()
            if row["submission_class"] != "floor"
        ]
        counts = Counter(row["source_lane"] for row in nonfloor)
        output: list[dict] = []
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
                output.append(record)
        return output

    def _policy_options(
        self, *, fraction_elapsed: float, force_final: bool = False,
    ) -> tuple[list[dict], str]:
        if self.reservable() <= 0:
            return [], ""
        calibration = read_json(self.calibration_path, {}) \
            if self.calibration_path else {}
        eligible = self._sort(self._eligible(), calibration)
        if not eligible:
            return [], ""
        allocated = self.committed()
        already_floored = bool(self.floor_done and self.floor_done())
        if not allocated and not already_floored:
            # Hold the floor briefly so it is chosen from a settled field rather
            # than from whatever finished first — unless the field is already
            # deep enough, or the settle window has passed, or the run is short
            # enough that the window fraction lands first (fraction_elapsed is
            # the run's own clock, so this scales with --duration-minutes).
            settled = (
                self.floor_settle_seconds <= 0
                or len(eligible) >= self.floor_settle_candidates
                or (time.monotonic() - self._t_start) >= self.floor_settle_seconds
                or fraction_elapsed >= 0.15
            )
            if not settled:
                return [], ""
            return eligible, "floor"

        final_phase = force_final or fraction_elapsed >= self.final_start_fraction
        if final_phase:
            finals = sum(row["submission_class"] == "final" for row in allocated)
            if finals < self.final_reserve:
                scored = [
                    record for record in eligible if self._score(record) is not None
                ]
                return (scored or eligible), "final"
            # The reserve is spent. A slot still held at the deadline is worth
            # exactly zero, so keep submitting the best candidate we have not
            # sent yet instead of falling silent — task 5 sat on 7 unused slots
            # for 49 minutes because this branch used to `return [], ""`.
            sent = {row["candidate_id"] for row in allocated}
            fresh = [
                record for record in eligible if record["candidate_id"] not in sent
            ]
            if fresh:
                return fresh, "milestone"
            return [], ""

        if self.reservable() <= self.final_reserve:
            return [], ""
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
                return choices, "calibration"

        incumbent = self._best_submitted_score(calibration)
        milestones = []
        for record in eligible:
            value = self._selection_score(record, calibration)
            if value is not None and self._improves(value, incumbent):
                milestones.append(record)
        if fraction_elapsed < 0.5:
            milestones = self._anti_monopoly_options(milestones)
        return milestones, "milestone" if milestones else ""

    def _candidate_view(self, record: dict, calibration: dict) -> dict:
        notes_path = Path(record["snapshot_path"]) / "evidence" / "NOTES.md"
        try:
            notes = notes_path.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            notes = ""
        return {
            "candidate_id": record["candidate_id"],
            "source_lane": record["source_lane"],
            "purpose": record.get("purpose"),
            "parent_id": record.get("parent_id"),
            "submission_mode": record.get("submission_mode"),
            "accelerator": record.get("accelerator"),
            "evaluation": record.get("evaluation"),
            "calibration": (
                calibration.get("candidate_predictions", {}).get(
                    record["candidate_id"]
                ) if calibration.get("active") else None
            ),
            "evidence_notes": notes,
        }

    def manager_context(
        self, *, fraction_elapsed: float, force_final: bool = False,
    ) -> dict | None:
        options, submission_class = self._policy_options(
            fraction_elapsed=fraction_elapsed, force_final=force_final
        )
        if not options:
            return None
        calibration = read_json(self.calibration_path, {}) \
            if self.calibration_path else {}
        history = []
        for row in self.submissions():
            history.append({
                "submission_id": row.get("submission_id"),
                "candidate_id": row.get("candidate_id"),
                "source_lane": row.get("source_lane"),
                "submission_class": row.get("submission_class"),
                "selection_reason": row.get("selection_reason"),
                "local_score": row.get("local_score"),
                "leaderboard_score": row.get("leaderboard_score"),
                "result": row.get("result"),
            })
        core = {
            "schema_version": 1,
            "submission_class": submission_class,
            "metric_direction": self.direction,
            "quota": {
                "max": self.max_submissions,
                "consumed": len(self.consumed()),
                "reservable": self.reservable(),
                "final_reserve": self.final_reserve,
            },
            "calibration": calibration,
            "submission_history": history,
            "candidates": [
                self._candidate_view(record, calibration) for record in options
            ],
        }
        safe_core = redact_value(core)
        if not isinstance(safe_core, dict):  # pragma: no cover - structural guard
            raise TypeError("redacted manager context must remain an object")
        return {
            **safe_core,
            "context_sha256": hashlib.sha256(canonical(safe_core)).hexdigest(),
        }

    def _manager_decision(
        self, context: dict, options: list[dict]
    ) -> tuple[str, dict | None, str]:
        digest = context["context_sha256"]
        state = read_json(self.manager_context_state_path, {})
        if state.get("context_sha256") != digest:
            state = {
                "context_sha256": digest,
                # 候选持续到达时不能无限重置等待窗口；一次真正预留后才重置。
                "first_seen_at": float(state.get("first_seen_at", time.time())),
            }
            atomic_json(self.manager_context_state_path, state)
        recommendation = read_json(self.recommendation_path, {}) \
            if self.recommendation_path else {}
        valid = (
            recommendation.get("context_sha256") == digest
            and float(recommendation.get("expires_at", 0)) > time.time()
        )
        if valid and recommendation.get("decision") == "wait":
            return "wait", None, "selection_manager_wait"
        if valid and recommendation.get("decision") == "submit":
            candidate_id = recommendation.get("candidate_id")
            selected = next(
                (record for record in options if record["candidate_id"] == candidate_id),
                None,
            )
            if selected is not None:
                return "submit", selected, "selection_manager"
        age = time.time() - float(state.get("first_seen_at", time.time()))
        if age < self.manager_fallback_seconds:
            return "wait", None, "selection_manager_pending"
        return "submit", options[0], "deterministic_manager_fallback"

    def select(
        self, *, fraction_elapsed: float, force_final: bool = False,
    ) -> tuple[dict, str, str] | None:
        options, submission_class = self._policy_options(
            fraction_elapsed=fraction_elapsed, force_final=force_final
        )
        if not options:
            return None
        if not self.manager_enabled:
            return options[0], submission_class, "deterministic_policy"
        context = self.manager_context(
            fraction_elapsed=fraction_elapsed, force_final=force_final
        )
        if context is None:
            return None
        decision, record, reason = self._manager_decision(context, options)
        return (
            (record, submission_class, reason)
            if decision == "submit" and record is not None else None
        )

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
                record, submission_class, selection_reason = selected
                submission_id = f"sub-{uuid.uuid4().hex[:10]}"
                attempt = {
                    "submission_id": submission_id,
                    "candidate_id": record["candidate_id"],
                    "source_lane": record["source_lane"],
                    "submission_class": submission_class,
                    "selection_reason": selection_reason,
                    "local_score": self._score(record),
                    "reserved_at": time.time(),
                    "attempt_number": 1 + sum(
                        row.get("candidate_id") == record["candidate_id"]
                        for row in self.submissions()
                    ),
                    "result": {"consumed": False, "status": "reserved"},
                }
            state = self._load()
            if attempt is not None:
                state["submissions"].append(attempt)
                atomic_json(self.state_path, state)
                if self.manager_enabled:
                    atomic_json(self.manager_context_state_path, {})
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
        # 兼容旧 Adapter 的非消耗型 error；这类已知失败不能永久拉黑候选，
        # 但必须受退避和最大尝试次数约束，避免紧循环耗尽时间。
        if (
            result_value.get("status") == "error"
            and not result_value.get("consumed")
        ):
            result_value["status"] = "retryable"
        if result_value.get("status") in {
            "retryable", "resource_deferred", "kernel_complete_pending_output",
        }:
            result_value.setdefault(
                "retry_after", time.time() + self.retry_backoff_seconds
            )
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
        self.write_status(fraction_elapsed=fraction_elapsed)
        completed = next(
            row for row in self.submissions() if row["submission_id"] == submission_id
        )
        self._emit_feedback(completed, "submission_result")
        return completed

    def refresh_scores(self) -> int:
        updates: list[dict] = []
        reconcile_method = getattr(self.adapter, "reconcile_external", None)
        if callable(reconcile_method):
            pending = [
                row for row in self.submissions()
                if row.get("result", {}).get("status") in {
                    "external_running", "kernel_complete_pending_output",
                }
                and float(row.get("result", {}).get("retry_after", 0))
                <= time.time()
            ]
            for stale in pending:
                record = self.registry.get(stale["candidate_id"])
                if record is None:
                    continue
                try:
                    reconciled = reconcile_method(record, stale)
                except Exception:  # noqa: BLE001
                    # 远端状态查询失败不证明 Kernel 已终止；保留 lease，下轮再查。
                    continue
                if reconciled is None:
                    continue
                result_value = reconciled.to_dict()
                if result_value.get("status") in {
                    "retryable", "kernel_complete_pending_output",
                }:
                    result_value.setdefault(
                        "retry_after", time.time() + self.retry_backoff_seconds
                    )
                changed_row = None
                with locked(self.state_path):
                    state = self._load()
                    for row in state["submissions"]:
                        if (
                            row["submission_id"] == stale["submission_id"]
                            and row.get("result", {}).get("status")
                            == stale.get("result", {}).get("status")
                        ):
                            row["result"] = result_value
                            row["completed_at"] = time.time()
                            changed_row = row.copy()
                            break
                    if changed_row is not None:
                        atomic_json(self.state_path, state)
                if changed_row is not None:
                    if result_value.get("consumed"):
                        self.registry.mark_submitted(record["candidate_id"])
                    updates.append(changed_row)
                    self._emit_feedback(changed_row, "external_kernel_reconciled")

        states_method = getattr(self.adapter, "submission_states", None)
        states = states_method() if callable(states_method) else {
            key: {"public_score": value, "exists": True}
            for key, value in self.adapter.scores().items()
        }
        if not states:
            self.write_status()
            return len(updates)
        score_updates: list[tuple[dict, str]] = []
        with locked(self.state_path):
            state = self._load()
            for row in state["submissions"]:
                external = states.get(row["submission_id"])
                if not external:
                    continue
                changed = False
                score_changed = False
                if row.get("result", {}).get("status") == "ambiguous" \
                        and external.get("exists"):
                    row["result"]["consumed"] = True
                    row["result"]["status"] = "submitted_reconciled"
                    changed = True
                score = external.get("public_score")
                if score is not None and row.get("leaderboard_score") != score:
                    row["leaderboard_score"] = score
                    row["scored_at"] = time.time()
                    changed = True
                    score_changed = True
                if changed:
                    score_updates.append((
                        row.copy(),
                        "leaderboard_score" if score_changed
                        else "submission_reconciled",
                    ))
            if score_updates:
                atomic_json(self.state_path, state)
        for row, event in score_updates:
            if row.get("result", {}).get("status") == "submitted_reconciled":
                self.registry.mark_submitted(row["candidate_id"])
                hook = getattr(self.adapter, "on_reconciled_submission", None)
                if callable(hook):
                    hook(row)
            self._emit_feedback(row, event)
        updates.extend(row for row, _event in score_updates)
        self.write_status()
        return len(updates)
