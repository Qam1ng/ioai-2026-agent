#!/usr/bin/env python3
"""Launcher — the human's ONLY action.

    python -m agent.main --slug <kaggle-competition> [--backend claude]
        [--deadline-min 120] [--max-submissions 10] [--max-cost-usd 20]
        [--phase INGEST]            # run a single phase (dev/debug)
        [--resume]                  # continue from state.json

Everything after launch is autonomous: read task -> analyze -> plan -> scaffold
-> floor submission -> experiment loop -> finalize -> report. See DESIGN.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .orchestrator import Orchestrator, Budget
from .providers.base import get_provider

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="kaggle competition slug")
    ap.add_argument("--backend", default="claude", help="claude | openai")
    ap.add_argument("--model", default=None, help="override backend model id")
    ap.add_argument("--deadline-min", type=float, default=120)
    ap.add_argument("--max-submissions", type=int, default=10)
    ap.add_argument("--max-cost-usd", type=float, default=20.0)
    ap.add_argument("--phase", default=None, help="run only this phase")
    ap.add_argument("--start-phase", default=None, help="start from this phase")
    ap.add_argument("--brief", default=None,
                    help="path to the official task statement; copied to "
                         "workspace/BRIEF.md as launch input (part of Phase 0)")
    args = ap.parse_args()

    ws = ROOT / "workspace" / args.slug
    ws.mkdir(parents=True, exist_ok=True)
    if args.brief:
        (ws / "BRIEF.md").write_text(Path(args.brief).read_text())

    provider = get_provider(args.backend, args.model)
    budget = Budget(deadline_s=args.deadline_min * 60,
                    max_submissions=args.max_submissions,
                    max_cost_usd=args.max_cost_usd)

    print(f"launching autonomous agent | slug={args.slug} backend={provider.name}"
          f" model={provider.model}\nworkspace={ws}")
    orch = Orchestrator(provider, args.slug, ws, budget,
                        start_phase=args.start_phase)
    orch.run(only_phase=args.phase)


if __name__ == "__main__":
    main()
