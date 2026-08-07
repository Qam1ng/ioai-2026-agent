#!/usr/bin/env python
"""End-to-end rehearsal of everything except the agents' judgement.

Exercises the full mechanical path on real competition data with a scripted
baseline standing in for the Coder: build a kernel, dry-run it against a staged
`/kaggle/input` tree, validate the submission format, push it through the
broker's gates in dry-run mode, refresh the calibration, and write the report.

Run it before every rehearsal and on competition morning. If this passes, the
only thing that can still fail is the models' thinking; if it fails, no amount
of model quality will save the run.

    python scripts/rehearse_mechanical.py --task task2_robot
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.budget import PodBudget, QuotaPool  # noqa: E402
from swarm.bus import Blackboard  # noqa: E402
from swarm.config import SwarmConfig  # noqa: E402
from swarm.ledger import ExperimentLedger, fit_calibration  # noqa: E402
from swarm.mirror.dryrun import dry_run, stage_input, validate_submission  # noqa: E402
from swarm.mirror.packages import check_kernel  # noqa: E402
from swarm.report import write_report  # noqa: E402
from swarm.schemas import (  # noqa: E402
    CandidateState,
    Constraint,
    Experiment,
    TaskCard,
)
from swarm.submit.broker import SubmissionBroker  # noqa: E402

TASKS = {
    "task2_robot": {
        "slug": "ioai-2026-ai-models-track-at-home-practice-task-2",
        "task_type": "imitation",
        "metric": "episode success rate",
    },
    "task3_radar": {
        "slug": "radar-ioai-2025",
        "task_type": "supervised",
        "metric": "RadarMetric (assumed macro-F1 over foreground classes)",
    },
    "task1_audio": {
        "slug": "ioai-2026-ai-models-track-practice-task-1",
        "task_type": "supervised",
        "metric": "0.5*acc_old + 0.5*acc_new",
    },
}

def floor_kernel() -> str:
    """Use the pod's own last-resort kernel, so the rehearsal tests the real thing."""
    from swarm.pod import TaskPod

    return TaskPod._fallback_kernel(None)  # type: ignore[arg-type]


#: Competitions that ship no sample-submission file, so the generic "echo the
#: sample" floor has nothing to echo. Measured, not assumed: only task 1 of the
#: three ships one. This is why plan 1 is always a task-specific minimal valid
#: submission rather than a fallback the harness can hardcode.
NO_SAMPLE_TASKS = {"task2_robot", "task3_radar"}


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task2_robot", choices=sorted(TASKS))
    ap.add_argument("--workspace", default="")
    ap.add_argument("--keep", action="store_true", help="keep the workspace afterwards")
    args = ap.parse_args()

    meta = TASKS[args.task]
    slug = meta["slug"]
    data_dir = ROOT / "data" / slug
    if not data_dir.exists():
        print(f"FAIL: no staged data at {data_dir}")
        return 1

    ws = Path(args.workspace or (ROOT / "workspace" / f"rehearsal-{args.task}"))
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    t_start = time.time()
    failures: list[str] = []

    step(1, "config + blackboard + budgets")
    cfg = SwarmConfig.load(ROOT / "configs" / "default.yaml", slug=slug, dry_run=True)
    cfg.slug, cfg.dry_run = slug, True
    bb = Blackboard(ws)
    quota = QuotaPool(ws / "gpu_quota.json", limit_hours=cfg.gpu_quota_hours)
    budget = PodBudget(ws / "budget.json", deadline_s=cfg.deadline_min * 60, quota=quota)
    bb.put_task_card(
        TaskCard(
            slug=slug,
            task_type=meta["task_type"],
            metric_name=meta["metric"],
            constraints=[Constraint("must", "the solution must train inside the Kaggle kernel")],
        )
    )
    print(f"    workspace={ws}")

    step(2, "static checks on the kernel (packages + no-network)")
    kernel = floor_kernel()
    problems = check_kernel(kernel)
    print(f"    violations: {problems or 'none'}")
    if problems:
        failures.append(f"static check flagged the floor kernel: {problems}")

    step(3, "stage a fake /kaggle/input tree from real competition data")
    staged = stage_input(ws / "kaggle_input", {slug: data_dir})
    n_staged = sum(1 for _ in staged.rglob("*"))
    print(f"    staged {n_staged} entries under {staged}")

    step(4, "dry-run the kernel with a wall-clock cap")
    res = dry_run(
        kernel,
        slug=slug,
        input_sources={slug: data_dir},
        workdir=ws / "dryrun",
        timeout_s=300,
        python_exe=sys.executable,
    )
    print(f"    ok={res.ok} exit={res.exit_code} wall={res.wall_s:.1f}s rows={res.submission_rows}")
    if not res.ok:
        print(f"    stderr: {res.stderr_tail[-300:]}")
        if args.task in NO_SAMPLE_TASKS and "no sample submission" in res.stderr_tail:
            print(
                "    EXPECTED: this competition ships no sample-submission file, so the\n"
                "    generic sample-echo has nothing to echo. The floor must come from plan 1,\n"
                "    built against the task's own output spec. The dry-run machinery itself is\n"
                "    working -- it staged the tree and the kernel ran and reported correctly."
            )
        else:
            failures.append("dry-run failed")

    step(5, "validate the produced submission against the sample")
    sample = next(
        (p for p in data_dir.rglob("*.csv") if p.name.lower().startswith(("submission", "sample"))),
        None,
    )
    if sample and res.submission_path:
        issues = validate_submission(Path(res.submission_path), sample)
        print(f"    issues: {issues or 'none'}")
        if issues:
            failures.append(f"format validation: {issues}")
    else:
        print("    skipped: this competition has no CSV sample submission")

    step(6, "broker gates + dry-run submission")
    broker = SubmissionBroker(bb, budget, cfg, quota)
    ok, why = broker.may_submit("floor")
    print(f"    may_submit(floor) -> {ok}: {why}")
    rec = broker.submit(
        code=kernel,
        candidate_id="rehearsal-floor",
        lane="floor",
        purpose="mechanical rehearsal",
        accelerator="cpu",
        local_score=None,
    )
    print(f"    submitted {rec.sub_id} lane={rec.lane} status={rec.status}")
    ok2, why2 = broker.may_submit("floor")
    print(f"    second floor blocked -> {not ok2}: {why2}")
    if ok2:
        failures.append("broker allowed a second floor submission")

    step(7, "ledger + calibration")
    cand = CandidateState(family="floor", status="ready", local_score=0.10, folds=[0.09, 0.11])
    bb.put_candidate(cand)
    led = ExperimentLedger(bb)
    led.add(
        Experiment(
            candidate_id=cand.candidate_id,
            role="rehearsal",
            description="floor kernel",
            local_score=0.10,
            folds=[0.09, 0.11],
            accepted=True,
        )
    )
    cal = fit_calibration(bb)
    print(f"    calibration: n={cal.n_points} noise_band={cal.noise_band:.4f} warn={cal.warning or '-'}")

    step(8, "technical report")
    path = write_report(bb, budget, quota, cfg, bb.get_task_card())
    text = Path(path).read_text()
    print(f"    wrote {path} ({len(text)} chars)")
    for required in ("Declaration", "Efficiency", "multi-agent"):
        if required.lower() not in text.lower():
            failures.append(f"report is missing a required section: {required}")

    print("\n" + "=" * 68)
    if failures:
        print(f"REHEARSAL FAILED ({len(failures)} problems) in {time.time() - t_start:.1f}s")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"REHEARSAL PASSED in {time.time() - t_start:.1f}s")
    print(json.dumps(bb.snapshot(), indent=2, default=str)[:900])

    if not args.keep:
        shutil.rmtree(ws, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
