"""Tuner: improve a working candidate's training configuration, one change at a time.

The acceptance rule is enforced here rather than trusted to the prompt. A model
asked "did this help?" will answer yes to a +0.001 mean move on a fold spread of
0.02; that is the exact mechanism by which a run drifts into noise-chasing, and
we have watched it make later iterations strictly worse.
"""

from __future__ import annotations

from typing import Any

from ..schemas import CandidateState, Experiment
from .base import Role, RoleResult

_STATUSES = ("ready", "broken", "exhausted")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TunerRole(Role):
    """Changes training config and resource envelope; never architecture."""

    role_name = "tuner"
    prompt_name = "tuner"
    tools = (
        "run_python",
        "run_bash",
        "write_file",
        "read_file",
        "list_dir",
        "memory_recall",
    )
    max_steps = 32

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_tuning(report: dict, candidate_id: str = "") -> dict:
        """Normalise the tuning report, recomputing acceptance from the folds.

        ``accepted`` is never taken on trust when the fold scores are present:
        the gain must clear one cross-fold standard deviation and improve the
        majority of folds. Both conditions are checkable from the numbers, so
        they are checked.
        """
        rep = report if isinstance(report, dict) else {}
        folds = [f for f in (_as_float(x) for x in (rep.get("folds") or [])) if f is not None]
        score = _as_float(rep.get("local_score"))
        std = _as_float(rep.get("local_std"))
        if score is None and folds:
            score = sum(folds) / len(folds)
        if std is None and len(folds) > 1:
            mean = sum(folds) / len(folds)
            std = (sum((f - mean) ** 2 for f in folds) / (len(folds) - 1)) ** 0.5

        changes = [c for c in (rep.get("changes") or []) if isinstance(c, dict)]
        status = str(rep.get("status", "") or "").strip().lower()
        if status not in _STATUSES:
            status = "ready" if score is not None else "broken"

        accepted = bool(rep.get("accepted"))
        problems: list[str] = []
        if not rep:
            problems.append("tuner produced no JSON report")
        if accepted and score is None:
            accepted = False
            problems.append("accepted a change with no measured score")
        if accepted and not folds:
            problems.append("accepted a change without per-fold scores")
        if accepted and len(changes) > 1 and sum(1 for c in changes if c.get("accepted")) > 1:
            problems.append(
                "more than one change accepted in a single run; the gain is unattributable"
            )

        return {
            "candidate_id": str(rep.get("candidate_id", "") or candidate_id),
            "changes": changes,
            "local_score": score,
            "local_std": std,
            "folds": folds,
            "runtime_min": _as_float(rep.get("runtime_min")) or 0.0,
            "accepted": accepted,
            "status": status,
            "reason": str(rep.get("reason", "") or ""),
            "problems": problems,
        }

    @staticmethod
    def is_significant(
        new_score: float | None,
        baseline: float | None,
        std: float | None,
        *,
        direction: str = "maximize",
        sigma: float = 1.0,
        folds_improved: int | None = None,
        n_folds: int | None = None,
    ) -> bool:
        """The gate from ``memory/lessons/local-cv-optimism.md``.

        Significant means: the gain exceeds ``sigma`` x cross-fold std AND, when
        per-fold information exists, it improves a majority of folds. A mean
        lifted by one lucky split is not an improvement.
        """
        if new_score is None:
            return False
        if baseline is None:
            return True  # nothing to beat yet; the first measurement is the baseline
        gain = (new_score - baseline) if direction == "maximize" else (baseline - new_score)
        band = abs(std or 0.0) * float(sigma)
        if gain <= band:
            return False
        if folds_improved is not None and n_folds:
            return folds_improved * 2 > n_folds
        return True

    # ---------------------------------------------------------------- run
    def tune(
        self,
        candidate: CandidateState | str,
        instructions: str = "",
        context: dict | None = None,
        *,
        noise_band: float = 0.0,
    ) -> dict:
        """Run one tuning pass on ``candidate`` and record it in the ledger."""
        cand = candidate if isinstance(candidate, CandidateState) else self.bb.get_candidate(str(candidate))
        cand_id = cand.candidate_id if cand else str(candidate or "")

        instructions = instructions or (
            "Improve this candidate's training configuration. Change one thing at a time, "
            "measure on the full fold scheme, and accept only a gain larger than the noise "
            f"band ({noise_band:g}) that also improves most folds."
        )
        ctx = dict(context or {})
        ctx["candidate"] = cand.to_dict() if cand else {"candidate_id": cand_id}
        ctx["noise_band"] = noise_band

        res: RoleResult = self.run(instructions, ctx)
        out = self.parse_tuning(res.report, candidate_id=cand_id)
        if not res.ok and not res.report:
            out["status"] = "broken"
            out["problems"].append(f"tuner step failed: {res.error or 'no report'}")

        baseline = cand.local_score if cand else None
        if out["accepted"] and not self.is_significant(
            out["local_score"], baseline, out["local_std"] or noise_band
        ):
            out["accepted"] = False
            out["problems"].append(
                "gain does not clear the noise band; recorded as rejected"
            )

        self.bb.add_experiment(
            Experiment(
                candidate_id=cand_id,
                role=self.name,
                description=out["reason"][:500] or "tuning pass",
                local_score=out["local_score"],
                local_std=out["local_std"],
                folds=out["folds"],
                runtime_s=out["runtime_min"] * 60.0,
                accepted=out["accepted"],
                reject_reason="; ".join(out["problems"])[:500] if not out["accepted"] else "",
            )
        )
        if cand is not None and out["accepted"] and out["local_score"] is not None:
            cand.local_score = out["local_score"]
            cand.local_std = out["local_std"]
            cand.folds = out["folds"]
            cand.status = "ready" if out["status"] == "ready" else cand.status
            self.bb.put_candidate(cand)
        return out
