from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

from .config import SystemConfig
from .controller import FinalController, ROOT


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="IOAI final three-lane agent system")
    sub = value.add_subparsers(dest="command", required=True)
    for name in ("doctor", "run"):
        command = sub.add_parser(name)
        command.add_argument(
            "--config", type=Path, default=ROOT / "configs/final_agent_system.toml"
        )
        command.add_argument("--assets-dir", type=Path, required=True)
    run = sub.choices["run"]
    run.add_argument("--slug", required=True)
    run.add_argument("--duration-minutes", type=float, required=True)
    run.add_argument("--competition-mode", choices=["practice", "formal"], required=True)
    run.add_argument("--run-id", default="")
    run.add_argument("--kaggle-user", default="")
    run.add_argument(
        "--resource-pool-id", default="",
        help="Kaggle-account quota pool id; keep it identical across both days",
    )
    run.add_argument(
        "--floor-group-id", default="",
        help="three-task simultaneous floor cohort, for example day1 or day2",
    )
    run.add_argument(
        "--day-slugs", default="",
        help="comma-separated three competition slugs sharing the account",
    )
    run.add_argument(
        "--live", action="store_true",
        help="enable real Kaggle submissions; without this flag the Broker is dry-run",
    )
    return value


def doctor(config: SystemConfig, assets: Path) -> int:
    errors: list[str] = []
    checks = {
        "assets": assets.expanduser().is_dir(),
        "claude_binary": config.claude.binary.is_file(),
        "claude_integrated_profile": (
            config.claude.integrated_profile_dir.is_dir()
        ),
        "claude_direct_profile": config.claude.direct_profile_dir.is_dir(),
        "codex_binary": config.codex.binary.is_file(),
        "search_prompt": config.search.prompt_spec.is_file(),
        "openrouter_key_env": bool(os.environ.get(config.codex.api_key_env)),
        "agent_os_sandbox": Path("/usr/bin/sandbox-exec").is_file(),
    }
    if checks["claude_binary"]:
        for label, profile in (
            ("integrated", config.claude.integrated_profile_dir),
            ("direct", config.claude.direct_profile_dir),
        ):
            if not profile.is_dir():
                continue
            env = os.environ.copy()
            env["CLAUDE_CONFIG_DIR"] = str(profile)
            try:
                result = subprocess.run(
                    [str(config.claude.binary), "auth", "status"],
                    capture_output=True, text=True, timeout=15, env=env,
                )
                payload = json.loads(result.stdout or "{}")
                checks[f"claude_{label}_max_login"] = bool(
                    payload.get("loggedIn")
                )
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
                checks[f"claude_{label}_max_login"] = False
    for name, ok in checks.items():
        print(f"{'OK' if ok else 'FAIL'}  {name}")
        if not ok:
            errors.append(name)
    print("NOTE  doctor only checks local wiring; it does not spend model tokens or submit.")
    return 2 if errors else 0


def main() -> int:
    args = parser().parse_args()
    config = SystemConfig.load(args.config, repo_root=ROOT)
    if args.command == "doctor":
        return doctor(config, args.assets_dir)
    controller = FinalController(
        config=config, slug=args.slug, assets_dir=args.assets_dir,
        duration_minutes=args.duration_minutes,
        competition_mode=args.competition_mode, kaggle_user=args.kaggle_user,
        live=args.live, run_id=args.run_id,
        resource_pool_id=args.resource_pool_id,
        floor_group_id=args.floor_group_id,
        day_slugs=tuple(
            item.strip() for item in args.day_slugs.split(",") if item.strip()
        ),
    )
    session = asyncio.run(controller.run())
    print(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
