"""The submission broker: one door, four lanes, and a winner's-curse-aware exit.

Every real Kaggle submission in the system goes through :meth:`SubmissionBroker.submit`.
That is the point — a single choke point is the only way the budget, the GPU
quota and the anti-noise-chasing gate can actually be enforced rather than
merely documented.

The four lanes and what each is *for*:

``floor``      insurance. One valid submission as early as possible so that
               "we have something scoring" is never in question. It outranks
               the final reserve: a run with no floor submission and a crash at
               T+3h scores nothing at all.
``probe``      information, not score. CPU kernels only, therefore zero GPU
               quota (``docs/QA-NOTES.md``: 30 GPU-h/week shared by all three
               problems of a day). See :mod:`swarm.submit.probes`.
``milestone``  a genuine candidate. Gated on the local gain exceeding the
               measured leaderboard noise band — this is the gate that stops
               the swarm from spending its day submitting noise.
``final``      end-of-window consolidation, inside the freeze window only.

And the exit rule: the competition takes our best submission on private test A
and re-scores it on private test B. Picking the raw argmax on A therefore picks
whichever submission had the luckiest A-noise, and that luck does not transfer.
:meth:`final_selection` shrinks each leaderboard score toward what our local CV
predicted before ranking — see its docstring for the arithmetic.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..budget import PodBudget, QuotaPool
from ..bus import Blackboard
from ..config import SwarmConfig
from ..schemas import SubmissionRecord
from agent.tools.registry import _kaggle_bin
from .kernel import (
    estimate_gpu_seconds,
    make_metadata,
    poll_kernel,
    push_kernel,
    unique_kernel_id,
    write_kernel,
)

LANES = ("floor", "probe", "milestone", "final")

#: Statuses that mean a real submission slot was consumed. A record that failed
#: before ``competition_submit_code`` never reached Kaggle and must not be
#: counted against the 50-per-problem budget.
_CONSUMING = ("queued", "running", "scored")


class SubmissionBroker:
    """Budget state machine + the only path to a real Kaggle submission."""

    def __init__(
        self,
        bb: Blackboard,
        budget: PodBudget,
        cfg: SwarmConfig,
        quota: QuotaPool,
    ):
        self.bb = bb
        self.budget = budget
        self.cfg = cfg
        self.quota = quota

    # --------------------------------------------------------------- counts
    def _records(self) -> list[SubmissionRecord]:
        return self.bb.get_submissions()

    def _consumed(self, recs: list[SubmissionRecord] | None = None) -> dict[str, int]:
        recs = self._records() if recs is None else recs
        counts = dict.fromkeys(LANES, 0)
        for r in recs:
            if r.status in _CONSUMING and r.lane in counts:
                counts[r.lane] += 1
        return counts

    def lane_allowance(self) -> dict[str, int]:
        """Submissions still available to each lane.

        These are per-lane *caps* checked at request time, not a partition —
        the numbers deliberately overlap, because the scarce thing is the pool
        and the caps only shape how it is spent. ``reserve_for_final`` is held
        back from probe and milestone but not from floor (insurance first) nor
        from final (that is what the reserve is).
        """
        s = self.cfg.submit
        used = self._consumed()
        used_total = sum(used.values())
        remaining = max(0, s.max_submissions - used_total)

        final_reserve = max(0, s.reserve_for_final - used["final"])
        floor_left = max(0, s.floor_lane - used["floor"])
        probe_left = max(0, s.probe_lane - used["probe"])

        explorable = max(0, remaining - final_reserve)
        allow_floor = min(floor_left, remaining)
        allow_probe = min(probe_left, explorable)
        return {
            "floor": allow_floor,
            "probe": allow_probe,
            "milestone": max(0, explorable - allow_floor - allow_probe),
            "final": remaining,
        }

    # ------------------------------------------------------------- the gate
    def best_submitted_local(self) -> float | None:
        """Best local score among submissions that actually reached Kaggle.

        Probes are excluded on purpose: a probe is built to be weak, so
        counting it as "the bar to beat" would let any real candidate through.
        """
        vals = [
            r.local_score
            for r in self._records()
            if r.status in _CONSUMING
            and r.lane in ("floor", "milestone", "final")
            and r.local_score is not None
        ]
        return max(vals) if vals else None

    def noise_band(self) -> float:
        return float(self.bb.get_calibration().noise_band or 0.0)

    def in_freeze_window(self) -> tuple[bool, str]:
        """Is now inside [freeze, deadline - buffer]? The final lane's window."""
        g = self.cfg.gates
        elapsed_min = self.budget.elapsed() / 60.0
        last_min = self.cfg.deadline_min - g.final_submit_buffer_min
        if elapsed_min < g.freeze_min:
            return False, (
                f"final lane opens at T+{g.freeze_min:.0f}min; now T+{elapsed_min:.0f}min"
            )
        if elapsed_min > last_min:
            return False, (
                f"final lane closed at T+{last_min:.0f}min (deadline buffer); "
                f"now T+{elapsed_min:.0f}min"
            )
        return True, ""

    def may_submit(
        self,
        lane: str,
        local_score: float | None = None,
        candidate_id: str | None = None,
        accelerator: str | None = None,
    ) -> tuple[bool, str]:
        """Decide whether ``lane`` may spend a submission right now.

        Returns ``(allowed, reason)``; ``reason`` is always populated so the
        refusal is auditable and the Manager can act on it rather than guess.
        """
        if lane not in LANES:
            return False, f"unknown lane {lane!r}; use one of {list(LANES)}"

        # No lane outlives the window. Run 3's best score was submitted at
        # T+33.3 of a 30-minute window because only the final lane checked the
        # clock — in a live competition that submission does not exist.
        if self.budget.remaining() <= 0:
            return False, "competition window closed; no lane accepts submissions"

        allowance = self.lane_allowance()
        used = self._consumed()

        # Floor first: insurance beats every other consideration except an
        # empty pool. A run without a floor submission scores zero if it dies.
        if lane == "floor":
            if used["floor"] > 0:
                return False, "floor submission already exists"
            if allowance["floor"] <= 0 or not self.budget.can_submit():
                return False, "submission budget exhausted"
            return True, "floor lane: insurance submission, always allowed first"

        if not self.budget.can_submit():
            return False, "submission budget exhausted"
        if allowance[lane] <= 0:
            return False, (
                f"{lane} lane allowance exhausted "
                f"(used {used[lane]}, reserve_for_final={self.cfg.submit.reserve_for_final})"
            )

        if lane == "probe":
            if accelerator is not None and accelerator.lower() != "cpu":
                return False, (
                    f"probe lane is CPU-only (got {accelerator!r}); probes exist "
                    "because they cost zero GPU quota"
                )
            return True, "probe lane: CPU kernel, no GPU quota consumed"

        if lane == "final":
            ok, why = self.in_freeze_window()
            if not ok:
                return False, why

        note = f"{lane} lane: allowed"
        if lane == "milestone":
            if local_score is None:
                return False, "milestone lane requires a local CV score to gate on"
            best = self.best_submitted_local()
            band = self.noise_band()
            threshold = band * self.cfg.submit.milestone_min_gain_sigma
            if best is None:
                note = "milestone lane: first scored candidate, nothing to beat"
            else:
                gain = float(local_score) - float(best)
                if gain <= threshold:
                    return False, (
                        f"local gain {gain:+.5f} over best submitted {best:.5f} does not "
                        f"clear {self.cfg.submit.milestone_min_gain_sigma:g} x noise band "
                        f"{band:.5f} = {threshold:.5f}; submitting this would be chasing "
                        "noise"
                    )
            # NB: no early return here — every non-probe lane still has to pass
            # the GPU quota check below.

        # GPU accelerators must be paid for out of the shared weekly quota.
        if accelerator is not None and accelerator.lower() != "cpu":
            need = self.cfg.submit.default_gpu_seconds
            have = self.quota.remaining_seconds()
            if have < need:
                return False, (
                    f"GPU quota too low: need {need/3600:.2f}h, have {have/3600:.2f}h "
                    "(30h/week is shared by all three problems of the day)"
                )

        return True, note

    # ---------------------------------------------------------------- submit
    def submit(
        self,
        code: str,
        candidate_id: str,
        lane: str,
        purpose: str,
        accelerator: str,
        local_score: float | None,
    ) -> SubmissionRecord:
        """Package, push, wait, submit, record. The only real submission path."""
        accel = (accelerator or "cpu").lower()
        rec = SubmissionRecord(
            candidate_id=candidate_id,
            lane=lane,
            purpose=purpose,
            accelerator=accel,
            local_score=local_score,
            status="error",
        )

        ok, why = self.may_submit(lane, local_score, candidate_id, accelerator=accel)
        if not ok:
            rec.error = f"blocked: {why}"
            self.bb.event("submit_blocked", lane=lane, candidate_id=candidate_id, reason=why)
            return rec

        if lane == "probe" and accel != "cpu":
            rec.error = "blocked: probe lane is CPU-only"
            self.bb.event("submit_blocked", lane=lane, reason=rec.error)
            return rec

        # The submission id keeps concurrent pushes of the same candidate
        # (floor lane and milestone lane) on separate Kaggle kernels.
        kernel_ref = unique_kernel_id(
            self.cfg.kaggle_user, self.cfg.slug, candidate_id, rec.sub_id
        )
        rec.kernel_ref = kernel_ref
        kdir = Path(self.bb.candidate_dir(candidate_id)) / f"kernel_{rec.sub_id}"
        code_file = kernel_ref.split("/", 1)[-1].replace("-", "_") + ".py"
        meta = make_metadata(kernel_ref, code_file, self.cfg.slug, accel)
        write_kernel(kdir, code, meta)
        self.bb.event(
            "kernel_written",
            sub_id=rec.sub_id,
            lane=lane,
            kernel_ref=kernel_ref,
            accelerator=accel,
            dir=str(kdir),
        )

        # Reserve GPU quota BEFORE pushing: two pods that both check "is there
        # an hour left" and then both push have each spent the same hour.
        reserved = 0.0
        if accel != "cpu":
            reserved = self.cfg.submit.default_gpu_seconds
            if not self.quota.try_reserve(rec.sub_id, reserved):
                rec.error = "blocked: GPU quota reservation failed"
                self.bb.event("submit_blocked", lane=lane, reason=rec.error)
                return rec

        if self.cfg.dry_run:
            # Local rehearsal: everything above really happened (the kernel is
            # on disk and can be inspected), nothing below touches Kaggle.
            if accel != "cpu":
                self.quota.release(rec.sub_id)
            rec.status = "scored"
            rec.lb_score = None
            rec.kernel_version = 0
            rec.error = ""
            rec.gpu_seconds = 0.0
            rec.scored_at = time.time()
            self.budget.note_submit()
            self.bb.add_submission(rec)
            self.bb.event("submit_dry_run", sub_id=rec.sub_id, lane=lane, dir=str(kdir))
            return rec

        t_push = time.time()
        pushed, version, raw = push_kernel(kdir)
        rec.kernel_version = version
        if not pushed:
            if accel != "cpu":
                self.quota.release(rec.sub_id)
            rec.error = f"kernel push failed: {raw[-1500:]}"
            self.bb.event("kernel_push_failed", sub_id=rec.sub_id, output=raw[-800:])
            self.bb.add_submission(rec)
            return rec

        status, log = poll_kernel(
            kernel_ref,
            timeout_s=self.cfg.submit.poll_max_min * 60.0,
            interval_s=self.cfg.submit.poll_interval_s,
        )
        measured_s = time.time() - t_push
        # The measured wall clock is the honest floor; the parsed duration only
        # refines it when Kaggle happens to report one.
        actual_s = max(measured_s, estimate_gpu_seconds(log, fallback=0.0))
        rec.gpu_seconds = actual_s if accel != "cpu" else 0.0
        if accel != "cpu":
            self.quota.settle(rec.sub_id, actual_s)

        if status != "complete":
            rec.error = f"kernel {status}: {log[-1500:]}"
            self.bb.event("kernel_failed", sub_id=rec.sub_id, status=status)
            self.bb.add_submission(rec)
            return rec

        message = f"[{rec.sub_id}] {lane}: {purpose}"[:480]
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.competition_submit_code(
                "submission.csv",
                message,
                self.cfg.slug,
                kernel=kernel_ref,
                kernel_version=version,
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            body = getattr(getattr(exc, "response", None), "text", "")
            rec.error = f"submit failed: {type(exc).__name__}: {exc}{body[:300]}"
            self.bb.event("submit_failed", sub_id=rec.sub_id, error=rec.error[:400])
            self.bb.add_submission(rec)
            return rec

        rec.status = "queued"
        rec.error = ""
        self.budget.note_submit()
        self.bb.add_submission(rec)
        # Scores can stay PENDING 30+ minutes (memory: kaggle-code-competition-
        # submission). One non-blocking look now; the pod re-polls later.
        self.poll_scores()
        refreshed = self._by_id(rec.sub_id)
        return refreshed or rec

    # ----------------------------------------------------------- score poll
    def _by_id(self, sub_id: str) -> SubmissionRecord | None:
        for r in self._records():
            if r.sub_id == sub_id:
                return r
        return None

    def poll_scores(self) -> int:
        """Refresh submissions whose leaderboard score has not landed yet.

        Deliberately one CLI round trip regardless of how many submissions are
        pending, and never a blocking wait: a Kaggle score can take half an
        hour and the pod has better things to do than sit on it. Returns the
        number of records updated, so a caller can log progress.
        """
        pending = [
            r
            for r in self._records()
            if r.status in ("queued", "running") and r.lb_score is None
        ]
        if not pending or self.cfg.dry_run:
            return 0

        rows = self._fetch_submission_rows()
        if not rows:
            return 0

        updated = 0
        for rec in pending:
            row = self._match_row(rows, rec)
            if row is None:
                continue
            status = (row.get("status") or "").strip().lower()
            score = _to_float(row.get("publicScore"))
            if score is not None:
                self.bb.update_submission(
                    rec.sub_id, lb_score=score, status="scored", scored_at=time.time()
                )
                self.bb.event("submission_scored", sub_id=rec.sub_id, lb_score=score)
                updated += 1
            elif status in ("error", "failed") and rec.status != "error":
                self.bb.update_submission(
                    rec.sub_id, status="error", error="Kaggle reported submission error"
                )
                updated += 1
            elif status == "running" and rec.status != "running":
                self.bb.update_submission(rec.sub_id, status="running")
                updated += 1
        return updated

    def _fetch_submission_rows(self) -> list[dict]:
        """``kaggle competitions submissions --csv`` parsed into dicts.

        CSV rather than the table so the description (which carries our
        ``sub_id``) survives without being truncated by column padding.
        """
        import csv
        import io

        from .kernel import _run

        rc, out = _run(
            [_kaggle_bin(), "competitions", "submissions", "-c", self.cfg.slug, "--csv"],
            timeout=300,
        )
        if rc != 0 or not out:
            return []
        # The CLI prints chatter ("Using competition: ...") before the header.
        lines = out.splitlines()
        start = next(
            (i for i, ln in enumerate(lines) if ln.lower().startswith("ref,")), None
        )
        if start is None:
            return []
        try:
            return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
        except csv.Error:
            return []

    @staticmethod
    def _match_row(rows: list[dict], rec: SubmissionRecord) -> dict | None:
        """Match by the sub_id we embedded in the description.

        Matching on date or position is what makes score polling attach the
        wrong score under concurrency; the description is ours and unique.
        """
        for row in rows:
            if rec.sub_id in (row.get("description") or ""):
                return row
        return None

    # ------------------------------------------------------ final selection
    def final_selection(self) -> SubmissionRecord | None:
        """Choose which submission we want counted as our best on private A.

        **Why not argmax(lb_score).** The final ranking re-scores the
        best-on-A submission on a *different* private set B. Among candidates
        of similar true quality, the one that tops A is disproportionately the
        one whose A-noise was favourable, and that noise does not repeat on B.
        Taking the raw maximum therefore systematically over-selects lucky runs
        — the winner's curse — and the expected drop from A to B grows with how
        hard we selected.

        **What we do instead.** Each submission's leaderboard score is treated
        as a noisy observation of its true quality, with noise sd = the
        measured ``calibration.noise_band``. Our prior for that true quality is
        what local CV predicted (``lb ~ intercept + slope * local`` from the
        calibration probe), or, absent a regression, the mean leaderboard score
        of all scored submissions. The posterior mean is the classic shrinkage
        estimate::

            w      = tau^2 / (tau^2 + sigma^2)
            shrunk = prediction + w * (lb - prediction)

        where ``tau`` is the spread of predictions across candidates (how much
        candidates genuinely differ) and ``sigma`` is the noise band. When
        candidates genuinely differ a lot, ``w -> 1`` and we trust the
        leaderboard; when the leaderboard is noisy relative to real differences,
        ``w -> 0`` and we fall back on local CV. Ranking on ``shrunk`` picks the
        candidate most likely to still be best on B.

        With ``noise_band == 0`` (no noise-floor probe was ever run) ``w = 1``
        and this degrades to raw argmax — correctly, since we then have no
        evidence that the leaderboard is noisy at all.
        """
        ranked = self.final_ranking()
        return ranked[0][0] if ranked else None

    def final_ranking(self) -> list[tuple[SubmissionRecord, float]]:
        """``(record, shrunk_estimate)`` best first. See :meth:`final_selection`."""
        scored = [
            r
            for r in self._records()
            if r.lb_score is not None and r.status in _CONSUMING and r.lane != "probe"
        ]
        if not scored:
            # No milestone/final scored yet — a floor or probe submission is
            # still better than nothing, so fall back to anything that scored.
            scored = [r for r in self._records() if r.lb_score is not None]
        if not scored:
            return []

        cal = self.bb.get_calibration()
        sigma = float(cal.noise_band or 0.0)
        mean_lb = sum(float(r.lb_score) for r in scored) / len(scored)

        def predict(r: SubmissionRecord) -> float:
            if cal.slope is not None and cal.intercept is not None and r.local_score is not None:
                return float(cal.intercept) + float(cal.slope) * float(r.local_score)
            return mean_lb

        preds = [predict(r) for r in scored]
        tau = _std(preds) if len(preds) > 1 else 0.0
        if sigma <= 0:
            w = 1.0  # no measured noise -> nothing to shrink toward
        elif tau <= 0:
            w = 0.5  # predictions identical: split the difference
        else:
            w = (tau**2) / (tau**2 + sigma**2)

        out = [
            (r, p + w * (float(r.lb_score) - p)) for r, p in zip(scored, preds)
        ]
        # Ties broken by local score: the candidate our own CV liked more is
        # the one more likely to hold up on B.
        out.sort(key=lambda t: (t[1], t[0].local_score or float("-inf")), reverse=True)
        return out

    # ------------------------------------------------------------- summary
    def status(self) -> dict:
        used = self._consumed()
        return {
            "used_by_lane": used,
            "allowance": self.lane_allowance(),
            "best_submitted_local": self.best_submitted_local(),
            "noise_band": self.noise_band(),
            "gpu_quota": self.quota.status(),
            "submissions_left": self.budget.submissions_left(),
        }


def _to_float(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in ("none", "pending", "nan", "n/a", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
