"""Command line entry point for the swarm.

The human's only competition-day action is ``swarm run --slug <slug>``. The
organizers hand each model a task-agnostic start prompt and a continuation
prompt, so ``run`` must be safe to invoke twice: state lives on disk and a
second invocation resumes rather than restarting (in particular the budget
clock does not reset, so a resumed run cannot silently claim a second budget).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.budget import PodBudget, QuotaPool  # noqa: E402
from swarm.bus import Blackboard  # noqa: E402
from swarm.config import SwarmConfig  # noqa: E402


def _make_broker(slug, bb, budget, cfg, quota):
    from swarm.submit.broker import SubmissionBroker

    return SubmissionBroker(bb, budget, cfg, quota)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = SwarmConfig.load(args.config, slug=args.slug, dry_run=args.dry_run)
    cfg.slug = args.slug
    if args.deadline_min:
        cfg.deadline_min = args.deadline_min
    if args.max_candidates:
        cfg.parallel.max_candidates = args.max_candidates
    if args.gpu_quota_hours:
        cfg.gpu_quota_hours = args.gpu_quota_hours

    ws = Path(args.workspace or (ROOT / "workspace" / args.slug))
    ws.mkdir(parents=True, exist_ok=True)
    cfg.workspace = str(ws)

    bb = Blackboard(ws)
    quota = QuotaPool(ws.parent / "gpu_quota.json", limit_hours=cfg.gpu_quota_hours)
    budget = PodBudget(
        ws / "budget.json",
        deadline_s=cfg.deadline_min * 60,
        max_submissions=cfg.submit.max_submissions,
        max_cost_usd=cfg.max_cost_usd,
        quota=quota,
    )

    from swarm.pod import TaskPod

    pod = TaskPod(cfg, bb, budget, quota, _make_broker(args.slug, bb, budget, cfg, quota))
    result = pod.run()
    print(json.dumps(result.__dict__, indent=2, default=str))
    return 0 if result.ok else 1


def cmd_day(args: argparse.Namespace) -> int:
    from swarm.portfolio import DayPlan, Portfolio

    slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
    root = Path(args.workspace or (ROOT / "workspace"))

    def cfg_for(slug: str) -> SwarmConfig:
        cfg = SwarmConfig.load(args.config, slug=slug, dry_run=args.dry_run)
        cfg.slug = slug
        cfg.deadline_min = args.window_min
        if args.max_candidates:
            cfg.parallel.max_candidates = args.max_candidates
        return cfg

    plan = DayPlan(
        slugs=slugs,
        window_min=args.window_min,
        gpu_quota_hours=args.gpu_quota_hours,
        workspace_root=root,
    )
    res = Portfolio(plan, cfg_for, _make_broker).run()
    print(
        json.dumps(
            {
                "results": {k: v.__dict__ for k, v in res.results.items()},
                "reallocations": res.reallocations,
                "errors": res.errors,
            },
            indent=2,
            default=str,
        )
    )
    return 0 if not res.errors else 1


def cmd_status(args: argparse.Namespace) -> int:
    ws = Path(args.workspace or (ROOT / "workspace" / args.slug))
    if not ws.exists():
        print(f"no workspace at {ws}")
        return 1
    bb = Blackboard(ws)
    snap = bb.snapshot()
    budget_path = ws / "budget.json"
    if budget_path.exists():
        snap["budget"] = json.loads(budget_path.read_text())
    print(json.dumps(snap, indent=2, default=str))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Pre-flight checks. Run this before the window opens, not during it."""
    import os

    checks: list[tuple[str, bool, str]] = []

    checks.append(("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    checks.append(("claude CLI", shutil.which("claude") is not None, shutil.which("claude") or "-"))
    checks.append(("kaggle CLI", shutil.which("kaggle") is not None, shutil.which("kaggle") or "-"))

    # The expensive check, and the one that actually bites: a spawned `claude -p`
    # does not inherit an interactive login, so an unauthenticated CLI turns
    # every coding role into a silent no-op.
    if not args.quick:
        from swarm.runners import check_auth

        ok, detail = check_auth()
        checks.append(("claude CLI authenticated", ok, detail))

    kag = Path.home() / ".kaggle"
    checks.append(
        (
            "kaggle credentials",
            (kag / "kaggle.json").exists() or (kag / "access_token").exists(),
            str(kag),
        )
    )
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SWARM_BASE_URL", "SWARM_MODEL"):
        checks.append((f"env {var}", bool(os.environ.get(var)), "set" if os.environ.get(var) else "unset"))

    for mod in ("numpy", "pandas", "sklearn", "yaml"):
        try:
            __import__(mod)
            checks.append((f"import {mod}", True, "ok"))
        except ImportError as exc:
            checks.append((f"import {mod}", False, str(exc)))

    for mod in ("swarm.bus", "swarm.budget", "swarm.pod", "swarm.roles", "swarm.submit.broker"):
        try:
            __import__(mod)
            checks.append((f"import {mod}", True, "ok"))
        except Exception as exc:
            checks.append((f"import {mod}", False, f"{type(exc).__name__}: {exc}"))

    width = max(len(c[0]) for c in checks)
    failed = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{mark}] {name.ljust(width)}  {detail}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("swarm", description="IOAI 2026 multi-agent competition system")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one problem end to end")
    r.add_argument("--slug", required=True)
    r.add_argument("--workspace")
    r.add_argument("--deadline-min", type=float)
    r.add_argument("--max-candidates", type=int)
    r.add_argument("--gpu-quota-hours", type=float)
    r.add_argument("--dry-run", action="store_true", help="never touch Kaggle")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("day", help="run all problems of a competition day")
    d.add_argument("--slugs", required=True, help="comma-separated competition slugs")
    d.add_argument("--workspace")
    d.add_argument("--window-min", type=float, default=360.0)
    d.add_argument("--gpu-quota-hours", type=float, default=30.0)
    d.add_argument("--max-candidates", type=int)
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(func=cmd_day)

    s = sub.add_parser("status", help="print the blackboard snapshot for a run")
    s.add_argument("--slug", required=True)
    s.add_argument("--workspace")
    s.set_defaults(func=cmd_status)

    doc = sub.add_parser("doctor", help="pre-flight environment checks")
    doc.add_argument("--quick", action="store_true", help="skip the live model-auth probe")
    doc.set_defaults(func=cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
