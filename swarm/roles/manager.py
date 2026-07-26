"""Manager: one blackboard snapshot in, exactly one action out.

The single-action loop is the point. A Manager that emits a multi-step plan is
committing to decisions whose premises will be stale by the time the second step
runs, and it makes the trajectory unauditable — the Jury cannot tell which piece
of evidence caused which action.

Parsing degrades rather than raises: an unusable Manager report must not kill a
pod that still has candidates running and a deadline to hit. ``wait`` is the
safe default because it costs one loop iteration and changes nothing.
"""

from __future__ import annotations

from typing import Any

from ..schemas import VALID_ACTIONS, ManagerAction
from .base import Role, RoleResult

#: Actions that open a new direction. Forbidden once the freeze gate has passed.
NEW_DIRECTION_ACTIONS = frozenset({"design", "code"})


class ManagerRole(Role):
    """Routes the pod, one action per step."""

    role_name = "manager"
    prompt_name = "manager"
    # Read-only and deliberately thin: the snapshot is the Manager's input, not
    # the filesystem. Every tool call it makes is wall-clock the pod is short of.
    tools = ("read_file", "list_dir", "memory_recall")
    max_steps = 6

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_action(report: dict) -> ManagerAction:
        """Turn a raw report into a valid :class:`ManagerAction`.

        Never raises. An invalid, missing or non-dict action degrades to
        ``wait`` with the reason recorded, so the pod keeps looping instead of
        crashing on a malformed model turn.
        """
        if not isinstance(report, dict) or not report:
            return ManagerAction(action="wait", reason="manager produced no JSON report")

        raw_action = report.get("action")
        if not isinstance(raw_action, str) or not raw_action.strip():
            return ManagerAction(
                action="wait",
                reason=f"manager report had no action field (keys: {sorted(report)[:8]})",
                instructions=str(report.get("instructions", "") or ""),
            )

        action = raw_action.strip().lower()
        if action not in VALID_ACTIONS:
            return ManagerAction(
                action="wait",
                target=str(report.get("target", "") or ""),
                reason=f"manager emitted invalid action {raw_action!r}; waiting instead",
                instructions=str(report.get("instructions", "") or ""),
            )

        return ManagerAction(
            action=action,
            target=str(report.get("target", "") or ""),
            reason=str(report.get("reason", "") or ""),
            instructions=str(report.get("instructions", "") or ""),
        )

    @staticmethod
    def enforce_gates(
        action: ManagerAction, *, has_floor_submission: bool, frozen: bool
    ) -> ManagerAction:
        """Apply the two gates the Manager is not allowed to talk its way past.

        The prompt states both rules, but a prompt is a request and a gate is a
        guarantee; these are the two whose violation is unrecoverable inside a
        six-hour window.
        """
        if frozen and action.action in NEW_DIRECTION_ACTIONS:
            return ManagerAction(
                action="wait",
                target=action.target,
                reason=(
                    f"freeze gate: {action.action!r} opens a new direction after freeze "
                    f"(manager said: {action.reason[:160]})"
                ),
            )
        if not has_floor_submission and action.action in ("probe", "aggregate", "stop"):
            return ManagerAction(
                action="wait",
                target=action.target,
                reason=(
                    f"floor gate: {action.action!r} is not allowed before a floor "
                    f"submission exists (manager said: {action.reason[:160]})"
                ),
            )
        return action

    # ---------------------------------------------------------------- run
    def decide(
        self,
        snapshot: dict | None = None,
        instructions: str = "",
        *,
        has_floor_submission: bool | None = None,
        frozen: bool = False,
    ) -> ManagerAction:
        """Run one Manager step against ``snapshot`` and return one action.

        ``snapshot`` defaults to the live blackboard snapshot. When
        ``has_floor_submission`` is not supplied it is read off the snapshot,
        so a caller cannot forget to pass it and quietly disable the floor gate.
        """
        snap = snapshot if snapshot is not None else self.bb.snapshot()
        if has_floor_submission is None:
            by_lane = (snap.get("submissions_by_lane") or {}) if isinstance(snap, dict) else {}
            has_floor_submission = bool(by_lane.get("floor", 0))

        instructions = instructions or (
            "Read the snapshot and emit exactly one action with reason and instructions."
        )
        res: RoleResult = self.run(instructions, snap)
        action = self.parse_action(res.report)
        if not res.ok and not res.report:
            action = ManagerAction(
                action="wait", reason=f"manager step failed: {res.error or 'no report'}"
            )
        action = self.enforce_gates(
            action, has_floor_submission=bool(has_floor_submission), frozen=frozen
        )
        self._emit("manager_action", action=action.action, target=action.target,
                   reason=action.reason[:300])
        return action
