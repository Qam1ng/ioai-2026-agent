"""The shared board for one task.

Both agents write into the same workspace and read the same board, which is the
whole point of this design: a candidate one of them produced is prior art the
other can extend, and a leaderboard score either of them earned is evidence for
both. The board itself is plain files under a lock, with no model in it.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

READY = "READY"
_TERMINAL_FAILURES = {"rejected", "invalid", "execution_error"}


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
                   + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


@dataclass
class Attempt:
    candidate_id: str
    author: str
    idea: str
    kind: str = "milestone"          # floor | milestone
    submission_id: str = field(default_factory=lambda: f"bk-{uuid.uuid4().hex[:10]}")
    status: str = "reserved"          # reserved|submitted|deferred|rejected|...
    detail: str = ""
    kernel_ref: str = ""
    kernel_version: int | None = None
    lb_score: float | None = None
    attempts: int = 1
    retry_after: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def consumed(self) -> bool:
        return self.status in {"submitted", "scored"}


class Board:
    def __init__(self, root: Path, *, slug: str, max_submissions: int):
        self.root = Path(root)
        self.slug = slug
        self.max_submissions = max_submissions
        self.work = self.root / "work"
        self.candidates = self.root / "candidates"
        self.state_path = self.root / "state.json"
        self.feedback_path = self.root / "feedback.jsonl"
        self.events_path = self.root / "events.jsonl"
        for path in (self.work, self.candidates):
            path.mkdir(parents=True, exist_ok=True)
        self.feedback_path.touch(exist_ok=True)
        if not self.state_path.exists():
            _atomic(self.state_path, {"slug": slug, "attempts": [],
                                      "created_at": time.time()})

    # -------------------------------------------------------------- events
    def event(self, event_name: str, **payload: Any) -> None:
        # The parameter is deliberately NOT called `kind`: callers pass a
        # `kind=` payload field, and a positional named `kind` collides with it
        # at every call site. That exact bug cost an earlier system two runs.
        with _locked(self.events_path):
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"ts": time.time(), "event": event_name, **payload},
                    ensure_ascii=False, sort_keys=True) + "\n")

    def feedback(self, payload: dict) -> None:
        """Written where both agents can read it; this is their only score channel."""
        with _locked(self.feedback_path):
            with self.feedback_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"ts": time.time(), **payload},
                                        ensure_ascii=False, sort_keys=True) + "\n")

    # ---------------------------------------------------------- candidates
    def ready_candidates(self) -> list[dict]:
        """Every finished candidate on disk, newest first.

        A candidate is finished only once READY exists, so a half-written kernel
        is never picked up.
        """
        found: list[dict] = []
        for directory in sorted(self.candidates.iterdir()):
            if not directory.is_dir() or not (directory / READY).is_file():
                continue
            meta = _read(directory / "meta.json", {})
            kernel = directory / "kernel.py"
            if not isinstance(meta, dict) or not kernel.is_file():
                continue
            found.append({
                "candidate_id": directory.name,
                "path": str(directory),
                "author": str(meta.get("author", "unknown")),
                "idea": str(meta.get("idea", ""))[:400],
                "accelerator": str(meta.get("accelerator", "cpu")).lower(),
                "parent": str(meta.get("parent", "")),
                "self_reported": meta.get("self_reported_score"),
                "created_at": (directory / READY).stat().st_mtime,
            })
        return sorted(found, key=lambda item: item["created_at"], reverse=True)

    # -------------------------------------------------------------- state
    def attempts(self) -> list[Attempt]:
        rows = _read(self.state_path, {}).get("attempts", [])
        return [Attempt(**row) for row in rows]

    def _save(self, attempts: list[Attempt]) -> None:
        state = _read(self.state_path, {"slug": self.slug})
        state["attempts"] = [asdict(a) for a in attempts]
        state["updated_at"] = time.time()
        _atomic(self.state_path, state)

    def consumed(self) -> int:
        return sum(1 for a in self.attempts() if a.consumed())

    def inflight(self) -> int:
        return sum(1 for a in self.attempts() if a.status == "reserved")

    def remaining(self) -> int:
        return max(0, self.max_submissions - self.consumed() - self.inflight())

    def best(self) -> Attempt | None:
        scored = [a for a in self.attempts() if a.lb_score is not None]
        return max(scored, key=lambda a: a.lb_score) if scored else None

    def submittable(self, *, retry_attempts: int) -> list[dict]:
        """Candidates that may still be spent on, newest first.

        A candidate is out of the running once it has been submitted, is in
        flight, failed in a way that will not fix itself, or has burned its
        retries. A transient failure is not one of those: the ONLY reason the
        earlier system lost its best solution twice was treating every failure
        as permanent.
        """
        by_id: dict[str, list[Attempt]] = {}
        for attempt in self.attempts():
            by_id.setdefault(attempt.candidate_id, []).append(attempt)
        now = time.time()
        out = []
        for candidate in self.ready_candidates():
            history = by_id.get(candidate["candidate_id"], [])
            if any(a.consumed() or a.status == "reserved" for a in history):
                continue
            if any(a.status in _TERMINAL_FAILURES for a in history):
                continue
            deferred = [a for a in history if a.status == "deferred"]
            if deferred and max(a.retry_after for a in deferred) > now:
                continue
            if len([a for a in history if a.status == "retryable"]) >= retry_attempts:
                continue
            candidate["prior_attempts"] = len(history)
            out.append(candidate)
        return out

    def reserve(self, candidate: dict, kind: str) -> Attempt | None:
        """Take a submission slot. Returns None when the budget is spent."""
        with _locked(self.state_path):
            attempts = self.attempts()
            consumed = sum(1 for a in attempts if a.consumed())
            inflight = sum(1 for a in attempts if a.status == "reserved")
            if consumed + inflight >= self.max_submissions:
                return None
            if any(
                a.candidate_id == candidate["candidate_id"]
                and (a.consumed() or a.status == "reserved")
                for a in attempts
            ):
                return None
            attempt = Attempt(
                candidate_id=candidate["candidate_id"],
                author=candidate.get("author", "unknown"),
                idea=candidate.get("idea", ""),
                kind=kind,
                attempts=1 + len([
                    a for a in attempts
                    if a.candidate_id == candidate["candidate_id"]
                ]),
            )
            attempts.append(attempt)
            self._save(attempts)
        self.event("reserved", submission_id=attempt.submission_id,
                   candidate_id=attempt.candidate_id, kind=kind)
        return attempt

    def update(self, submission_id: str, **changes: Any) -> Attempt | None:
        with _locked(self.state_path):
            attempts = self.attempts()
            target = None
            for attempt in attempts:
                if attempt.submission_id == submission_id:
                    for key, value in changes.items():
                        setattr(attempt, key, value)
                    attempt.updated_at = time.time()
                    target = attempt
                    break
            if target is not None:
                self._save(attempts)
        return target

    # ------------------------------------------------------------ digests
    def digest(self, *, limit: int = 25) -> str:
        """The board as the agents see it. Only leaderboard numbers are facts."""
        attempts = self.attempts()
        lines = [
            f"Submissions: {self.consumed()} consumed / {self.max_submissions} "
            f"allowed ({self.remaining()} still available, "
            f"{self.inflight()} in flight)",
        ]
        best = self.best()
        lines.append(
            f"Best leaderboard score so far: {best.lb_score} "
            f"(candidate `{best.candidate_id}` by {best.author})"
            if best else
            "Best leaderboard score so far: nothing has scored yet."
        )
        scored = [a for a in attempts if a.lb_score is not None]
        if scored:
            lines.append("")
            lines.append("| candidate | author | leaderboard | idea |")
            lines.append("|---|---|---|---|")
            for attempt in sorted(scored, key=lambda a: a.lb_score, reverse=True)[:limit]:
                idea = attempt.idea.replace("|", "/")[:70] or "-"
                lines.append(
                    f"| {attempt.candidate_id} | {attempt.author} | "
                    f"{attempt.lb_score} | {idea} |"
                )
        pending = [a for a in attempts if a.status == "reserved"]
        if pending:
            lines.append("")
            lines.append("In flight right now: " + ", ".join(
                f"{a.candidate_id} ({a.author})" for a in pending))
        failed = [a for a in attempts if a.status in _TERMINAL_FAILURES]
        if failed:
            lines.append("")
            lines.append("Rejected before scoring (do not repeat these mistakes):")
            for attempt in failed[-6:]:
                lines.append(f"  - {attempt.candidate_id}: {attempt.detail[:160]}")
        waiting = [c for c in self.ready_candidates()
                   if not any(a.candidate_id == c["candidate_id"]
                              for a in attempts)]
        if waiting:
            lines.append("")
            lines.append("Queued, not yet submitted: " + ", ".join(
                f"{c['candidate_id']} ({c['author']})" for c in waiting[:12]))
        return "\n".join(lines)
