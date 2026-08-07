"""Budgets that survive a crash, and the GPU quota pool shared across a day.

Three separate scarce resources, deliberately not conflated:

1. **Wall clock** — 6 hours per competition day, shared by three problems.
2. **Submissions** — 50 per problem. Plentiful; not the binding constraint.
3. **Kaggle GPU quota** — 30 h/week on the account, and *all three problems of
   a day share one account* (``docs/QA-NOTES.md``). This is the real budget.
   CPU kernels cost zero GPU quota, which is what makes a cheap probe lane
   possible.

Everything is persisted on every mutation. The pre-existing ``agent.Budget``
resets its clock and counters whenever the process restarts, which silently
hands a resumed run a second full budget; that bug is not reproduced here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .bus import _atomic_write, _locked


@dataclass
class QuotaState:
    gpu_seconds_used: float = 0.0
    gpu_seconds_limit: float = 30 * 3600.0  # 30 h/week per Kaggle account
    reservations: dict[str, float] = field(default_factory=dict)  # sub_id -> seconds


class QuotaPool:
    """GPU-hours shared by every pod running on one Kaggle account.

    Reserve before pushing a kernel, settle when it finishes. Reserving up
    front is what stops three pods from simultaneously believing they have the
    last GPU-hour.
    """

    def __init__(self, path: Path, limit_hours: float = 30.0):
        self.path = Path(path)
        self.limit_s = limit_hours * 3600.0
        if not self.path.exists():
            self._save(QuotaState(gpu_seconds_limit=self.limit_s))

    def _load(self) -> QuotaState:
        if not self.path.exists():
            return QuotaState(gpu_seconds_limit=self.limit_s)
        d = json.loads(self.path.read_text())
        return QuotaState(
            gpu_seconds_used=d.get("gpu_seconds_used", 0.0),
            gpu_seconds_limit=d.get("gpu_seconds_limit", self.limit_s),
            reservations=d.get("reservations", {}),
        )

    def _save(self, st: QuotaState) -> None:
        _atomic_write(self.path, json.dumps(asdict(st), indent=2))

    def committed_seconds(self, st: QuotaState | None = None) -> float:
        st = st or self._load()
        return st.gpu_seconds_used + sum(st.reservations.values())

    def remaining_seconds(self) -> float:
        st = self._load()
        return max(0.0, st.gpu_seconds_limit - self.committed_seconds(st))

    def try_reserve(self, sub_id: str, seconds: float) -> bool:
        """Reserve GPU seconds. Returns False when the pool cannot cover it."""
        with _locked(self.path):
            st = self._load()
            if self.committed_seconds(st) + seconds > st.gpu_seconds_limit:
                return False
            st.reservations[sub_id] = seconds
            self._save(st)
            return True

    def settle(self, sub_id: str, actual_seconds: float) -> None:
        """Convert a reservation into actual usage."""
        with _locked(self.path):
            st = self._load()
            st.reservations.pop(sub_id, None)
            st.gpu_seconds_used += max(0.0, actual_seconds)
            self._save(st)

    def release(self, sub_id: str) -> None:
        """Drop a reservation that never ran (push failed, kernel cancelled)."""
        with _locked(self.path):
            st = self._load()
            if st.reservations.pop(sub_id, None) is not None:
                self._save(st)

    def status(self) -> dict:
        st = self._load()
        return {
            "gpu_hours_used": round(st.gpu_seconds_used / 3600.0, 2),
            "gpu_hours_reserved": round(sum(st.reservations.values()) / 3600.0, 2),
            "gpu_hours_limit": round(st.gpu_seconds_limit / 3600.0, 2),
            "gpu_hours_remaining": round(self.remaining_seconds() / 3600.0, 2),
        }


@dataclass
class BudgetState:
    t0: float = 0.0
    deadline_s: float = 6 * 3600.0
    submissions: int = 0
    max_submissions: int = 50
    cost_usd: float = 0.0
    max_cost_usd: float = 1000.0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0


class PodBudget:
    """Per-problem budget. Persisted on every mutation so resume is honest."""

    def __init__(
        self,
        path: Path,
        deadline_s: float = 6 * 3600.0,
        max_submissions: int = 50,
        max_cost_usd: float = 1000.0,
        quota: QuotaPool | None = None,
    ):
        self.path = Path(path)
        self.quota = quota
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.st = BudgetState(**d)
            # Deadline and caps may be raised between runs; never silently
            # reset t0 -- that is what makes a resumed run honest.
            self.st.deadline_s = deadline_s
            self.st.max_submissions = max_submissions
            self.st.max_cost_usd = max_cost_usd
        else:
            self.st = BudgetState(
                t0=time.time(),
                deadline_s=deadline_s,
                max_submissions=max_submissions,
                max_cost_usd=max_cost_usd,
            )
        self._save()

    def _save(self) -> None:
        _atomic_write(self.path, json.dumps(asdict(self.st), indent=2))

    # ------------------------------------------------------------ clocks
    def elapsed(self) -> float:
        return time.time() - self.st.t0

    def remaining(self) -> float:
        return max(0.0, self.st.deadline_s - self.elapsed())

    def fraction_elapsed(self) -> float:
        return min(1.0, self.elapsed() / self.st.deadline_s) if self.st.deadline_s else 1.0

    # ------------------------------------------------------- submissions
    def can_submit(self) -> bool:
        return self.st.submissions < self.st.max_submissions

    def submissions_left(self) -> int:
        return max(0, self.st.max_submissions - self.st.submissions)

    def note_submit(self) -> None:
        self.st.submissions += 1
        self._save()

    # --------------------------------------------------------------- llm
    def note_llm(self, usage: dict, cost: float) -> None:
        self.st.tokens_in += int(usage.get("input_tokens", 0) or 0)
        self.st.tokens_out += int(usage.get("output_tokens", 0) or 0)
        self.st.cost_usd += float(cost or 0.0)
        self.st.llm_calls += 1
        self._save()

    # ------------------------------------------------------------- halt
    def exhausted(self) -> str | None:
        if self.remaining() <= 0:
            return "wall-clock deadline reached"
        if self.st.cost_usd >= self.st.max_cost_usd:
            return "LLM cost budget reached"
        return None

    def status(self) -> dict:
        d = {
            "elapsed_min": round(self.elapsed() / 60, 1),
            "remaining_min": round(self.remaining() / 60, 1),
            "fraction_elapsed": round(self.fraction_elapsed(), 3),
            "submissions": self.st.submissions,
            "submissions_left": self.submissions_left(),
            "cost_usd": round(self.st.cost_usd, 2),
            "tokens_in": self.st.tokens_in,
            "tokens_out": self.st.tokens_out,
            "llm_calls": self.st.llm_calls,
        }
        if self.quota:
            d.update(self.quota.status())
        return d

    def status_line(self) -> str:
        s = self.status()
        line = (
            f"[budget] elapsed={s['elapsed_min']}min remaining={s['remaining_min']}min "
            f"submissions={s['submissions']}/{self.st.max_submissions} "
            f"llm=${s['cost_usd']} (in={s['tokens_in']} out={s['tokens_out']})"
        )
        if self.quota:
            line += f" gpu_quota={s['gpu_hours_remaining']}h left"
        return line
