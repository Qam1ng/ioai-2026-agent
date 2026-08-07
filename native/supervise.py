"""Supervision — deterministic, and deliberately not an agent.

Everything here is a comparison of an observed number against an expected one:
the harness's submission counter against Kaggle's, our processes against the
GPUs we were allotted, this round's output against last round's. An LLM asked to
do this would still need someone to run `nvidia-smi` and hand it the output, and
would then charge a round trip to return a `!=`.

The distinction that makes this work: the harness is poor at *prohibition* —
Bash is a universal escape hatch, and every behavioural fence we built was
walked around within one run — but it is very good at *observation*. A solver can
bypass the submission gate; it cannot stop us noticing afterwards that the
counter disagrees with Kaggle.

So nothing here blocks anything. It watches, reports loudly, and puts what it
finds on the facts board where the solvers will see it too.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import facts

# Last time each agent did anything observable, updated from the message drain.
ACTIVITY: dict[str, float] = {}
# Facts flagged urgent by whoever found them, awaiting delivery.
URGENT: list[dict] = []


@dataclass
class Drift:
    kind: str
    detail: str
    fatal: bool = False


@dataclass
class Reconciler:
    """Compares what the harness believes with what the machine actually shows."""

    ws: Path
    slug: str
    allowed_gpus: set[str]
    seen: set[str] = field(default_factory=set)

    # ---------------------------------------------------------------- checks
    def submissions(self, counted: int) -> Drift | None:
        """Kaggle's count is the truth; ours is a belief.

        Run 1 shelled out to the CLI three times while the harness sat at 0/3,
        and nobody noticed until the next day's log reading.
        """
        try:
            from datetime import datetime, timezone

            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            today = datetime.now(timezone.utc).date()
            real = sum(1 for s in (api.competition_submissions(self.slug) or [])
                       if getattr(s, "date", None) and s.date.date() == today)
        except Exception:  # noqa: BLE001 — a failed check is not a drift
            return None
        if real > counted:
            return Drift("submission-bypass",
                         f"Kaggle shows {real} submissions today but the harness "
                         f"counted {counted}. Something submitted without going "
                         f"through the gate, so the budget is no longer ours to "
                         f"spend.", fatal=True)
        return None

    def gpus(self) -> Drift | None:
        """CUDA_VISIBLE_DEVICES is not hierarchical, so a child can walk onto a
        card belonging to somebody else's job. In run 2 one did."""
        if not self.allowed_gpus:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout
            idx = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout
        except Exception:  # noqa: BLE001
            return None
        uuid_to_idx = {}
        for line in idx.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                uuid_to_idx[parts[1]] = parts[0]
        trespass = set()
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            gpu = uuid_to_idx.get(parts[0])
            if gpu and gpu not in self.allowed_gpus and self._ours(parts[1]):
                trespass.add(gpu)
        if trespass:
            return Drift("gpu-trespass",
                         f"our processes are on GPU {sorted(trespass)}, outside "
                         f"the allotted {sorted(self.allowed_gpus)}. This machine "
                         f"is shared.")
        return None

    def orphans(self) -> Drift | None:
        """Training and watcher scripts backgrounded with nohup outlive the
        session that started them — run 1's watcher submitted twice more after
        the harness had printed its summary."""
        n = sum(1 for _ in self._workspace_pids())
        if n > 12:  # a few are normal; a pile is a leak
            return Drift("process-pile",
                         f"{n} processes are running out of the workspace. Some "
                         f"were backgrounded and will outlive their solver.")
        return None

    def cost(self, spent: float, cap: float) -> Drift | None:
        # cap 0 means the run is deliberately uncapped; every dollar would
        # otherwise read as an overrun.
        if cap and spent > cap * 1.05:
            return Drift("cost-overrun",
                         f"${spent:.2f} spent against a ${cap:.0f} cap.")
        return None

    # --------------------------------------------------------------- helpers
    def _ours(self, pid: str) -> bool:
        try:
            return os.stat(f"/proc/{pid}").st_uid == os.getuid()
        except OSError:
            return False

    def _workspace_pids(self):
        root = str(self.ws.resolve())
        if not Path("/proc").is_dir():
            return
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                if os.path.realpath(f"/proc/{pid}/cwd").startswith(root):
                    yield int(pid)
            except OSError:
                continue

    # ------------------------------------------------------------------ main
    def sweep(self, counted: int, spent: float, cap: float) -> list[Drift]:
        found = [d for d in (self.submissions(counted), self.gpus(),
                             self.orphans(), self.cost(spent, cap)) if d]
        fresh = [d for d in found if d.kind not in self.seen]
        self.seen.update(d.kind for d in fresh)
        return fresh


def report(drifts: list[Drift], trace) -> bool:
    """Announce drift everywhere at once. Returns True if any was fatal."""
    fatal = False
    for d in drifts:
        print(f"\n!! DRIFT [{d.kind}] {d.detail}\n", flush=True)
        trace.log("drift", kind=d.kind, detail=d.detail, fatal=d.fatal)
        facts.board().post("env", f"[{d.kind}] {d.detail}"[:300], src="harness")
        fatal = fatal or d.fatal
    return fatal


def predict_next_cost(usage, costs: list[float]) -> float:
    """Estimate the next round from context growth rather than from the last bill.

    Cost tracks context, and context only grows: run 1 went 0.85 -> 1.74 -> 2.65
    -> 3.51 -> 4.07 and walked through a $15 cap while each individual round
    still looked affordable. `percentage` is readable at any time, so scale the
    last round by how much fuller the window has become rather than assuming the
    next one costs what the last one did.
    """
    if not costs:
        return 0.0
    last = costs[-1]
    if len(costs) >= 2 and costs[-2] > 0:
        growth = max(1.0, last / costs[-2])
    else:
        growth = 1.3
    pct = float(getattr(usage, "percentage", 0) or
                (usage or {}).get("percentage", 0) if usage else 0)
    if pct > 70:            # close to auto-compaction: expect a jump
        growth = max(growth, 1.6)
    return last * growth


def stalled_for(agent: str) -> float:
    """Seconds since this agent last did anything observable."""
    t = ACTIVITY.get(agent)
    return 0.0 if t is None else time.time() - t


def flag_urgent(kind: str, text: str, src: str) -> None:
    URGENT.append({"kind": kind, "text": text, "src": src, "ts": time.time()})


def drain_urgent(reader: str) -> str:
    """Urgent facts this reader has not been handed yet.

    Normal facts ride along on tool results, which costs nothing but only
    reaches an agent that is calling tools — precisely not the one sitting on a
    ten-minute background job. Urgent facts are pushed at the next round
    boundary instead, which guarantees delivery without cutting into a turn
    that is already in flight.
    """
    out = [u for u in URGENT if u["src"] != reader
           and reader not in u.setdefault("delivered", set())]
    for u in out:
        u["delivered"].add(reader)
    if not out:
        return ""
    lines = "\n".join(f"  {u['kind']} ({u['src']}) {u['text']}" for u in out)
    return ("\n\n!! URGENT from the facts board — another solver flagged this as "
            "something that changes what everyone should be doing:\n" + lines + "\n")
