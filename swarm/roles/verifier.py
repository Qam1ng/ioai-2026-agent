"""Verifier: decides whether a candidate's number is real.

This role exists because of one measured failure: a single 25% stratified
holdout read 0.9156 while the leaderboard read 0.78095, and the phantom 0.0
accuracies from 1-sample classes in that holdout led to "fixes" that made later
iterations worse. So the Verifier re-measures with k-fold, reports a standard
deviation next to every mean, and is adversarial by construction — it sees the
candidate's artifacts, never the Coder's or Tuner's account of them.

Normalisation here is fail-closed on purpose: a report that lists problems but
claims ``pass``, or that carries an unrecognised verdict, becomes ``suspect``.
The Manager is allowed to take a risk knowingly; it is not allowed to take one
because a verdict string was ambiguous.
"""

from __future__ import annotations

from typing import Any

from ..schemas import CandidateState, Experiment
from .base import Role, RoleResult

VERDICTS = ("pass", "suspect", "fail")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class VerifierRole(Role):
    """Independent re-measurement of a candidate: score, variance, format, leakage."""

    role_name = "verifier"
    prompt_name = "verifier"
    # Execution plus writes for its own verification scripts, but nothing that
    # can reach Kaggle: a verifier that can submit could "verify" by submitting.
    tools = ("run_python", "write_file", "read_file", "list_dir", "memory_recall")
    max_steps = 26

    def __init__(self, name: str | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(name or self.role_name, *args, **kwargs)

    # ------------------------------------------------------------- parsing
    @staticmethod
    def parse_verification(report: dict, candidate_id: str = "") -> dict:
        """Normalise a verification report; never upgrade a verdict, only downgrade."""
        rep = report if isinstance(report, dict) else {}

        folds = [f for f in (_as_float(x) for x in (rep.get("folds") or [])) if f is not None]
        score = _as_float(rep.get("local_score"))
        std = _as_float(rep.get("local_std"))
        if score is None and folds:
            score = sum(folds) / len(folds)
        if std is None and len(folds) > 1:
            mean = sum(folds) / len(folds)
            std = (sum((f - mean) ** 2 for f in folds) / (len(folds) - 1)) ** 0.5

        problems = [str(p) for p in (rep.get("problems") or []) if str(p).strip()]
        verdict = str(rep.get("verdict", "") or "").strip().lower()

        if not rep:
            problems.append("verifier produced no JSON report")
            verdict = "fail"
        elif verdict not in VERDICTS:
            problems.append(f"unrecognised verdict {rep.get('verdict')!r}; treated as suspect")
            verdict = "suspect"

        selftest = rep.get("metric_selftest")
        selftest = selftest if isinstance(selftest, dict) else {}
        if not selftest.get("passed"):
            problems.append("metric self-test did not pass or was not run")
        if rep.get("format_valid") is not True:
            problems.append("submission format was not confirmed valid")
        if len(folds) < 2:
            problems.append(
                "fewer than 2 folds reported: a single split reads optimistically "
                "(0.9156 local vs 0.78095 leaderboard on a past task)"
            )
        if score is not None and std is not None and std > abs(score) * 0.25 and score:
            problems.append(f"cross-fold std {std:.4g} is large relative to the mean {score:.4g}")

        # Fail-closed: problems and a clean verdict cannot coexist.
        if problems and verdict == "pass":
            verdict = "suspect"

        return {
            "candidate_id": str(rep.get("candidate_id", "") or candidate_id),
            "local_score": score,
            "local_std": std,
            "folds": folds,
            "fold_scheme": str(rep.get("fold_scheme", "") or ""),
            "metric_selftest": selftest,
            "format_valid": bool(rep.get("format_valid")),
            "problems": problems,
            "verdict": verdict,
        }

    # ---------------------------------------------------------------- run
    def verify(
        self,
        candidate: CandidateState | str,
        instructions: str = "",
        context: dict | None = None,
    ) -> dict:
        """Re-measure ``candidate`` and write the verdict back to the blackboard.

        The context deliberately carries only artifact locations and the task
        card — never the Coder's or Tuner's narrative about what the candidate
        should score.
        """
        cand = candidate if isinstance(candidate, CandidateState) else self.bb.get_candidate(str(candidate))
        cand_id = cand.candidate_id if cand else str(candidate or "")
        card = self.bb.get_task_card()

        instructions = instructions or (
            "Independently re-measure this candidate: k-fold (grouped or stratified as the "
            "task card requires), mean and cross-fold std with every fold score, metric "
            "self-test, submission format validation, and an explicit leakage hunt."
        )
        ctx = dict(context or {})
        ctx["candidate_id"] = cand_id
        ctx["kernel_dir"] = cand.kernel_dir if cand else ""
        ctx["grouping_variable"] = card.grouping_variable if card else ""
        ctx["metric"] = {
            "name": card.metric_name if card else "",
            "description": card.metric_description if card else "",
            "direction": card.metric_direction if card else "maximize",
        }
        ctx["submission_format"] = card.submission_format if card else ""

        res: RoleResult = self.run(instructions, ctx)
        out = self.parse_verification(res.report, candidate_id=cand_id)
        if not res.ok and not res.report:
            out["verdict"] = "fail"
            out["problems"].append(f"verifier step failed: {res.error or 'no report'}")

        self.bb.add_experiment(
            Experiment(
                candidate_id=cand_id,
                role=self.name,
                description=f"verification: {out['fold_scheme'] or 'k-fold'} -> {out['verdict']}",
                local_score=out["local_score"],
                local_std=out["local_std"],
                folds=out["folds"],
                accepted=out["verdict"] == "pass",
                reject_reason="; ".join(out["problems"])[:500],
            )
        )
        if cand is not None:
            if out["verdict"] == "fail":
                cand.status = "failed"
                cand.last_error = "; ".join(out["problems"])[:500]
            else:
                cand.local_score = out["local_score"]
                cand.local_std = out["local_std"]
                cand.folds = out["folds"]
                cand.status = "ready"
            self.bb.put_candidate(cand)
        self._emit("verified", candidate_id=cand_id, verdict=out["verdict"],
                   local_score=out["local_score"], n_problems=len(out["problems"]))
        return out
