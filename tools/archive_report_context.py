#!/usr/bin/env python3
"""Continuously preserve the small text artifacts needed for solution reports.

The competition workspaces contain multi-gigabyte assets and caches.  This
watcher deliberately copies only conversations, trajectories, source files,
candidate metadata, evidence, evaluation state, and logs.  Each growing file is
copied through a temporary file and atomically replaced in the archive.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".toml",
    ".txt",
}

EXCLUDED_PARTS = {
    ".agent_home",
    ".cache",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "mcache",
    "models",
    "OFFICIAL_ASSETS",
    "ORIGINAL_ASSETS",
}

CONTROL_TEXT_FILES = {
    "BROKER_STATUS.json",
    "broker_events.jsonl",
    "broker_state.json",
    "controller_events.jsonl",
    "kaggle_gateway_events.jsonl",
}


def _included(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if relative == Path("RUN_STATUS.json"):
        return True
    if any(part in EXCLUDED_PARTS for part in parts):
        return False

    if parts[0] == "control":
        if len(parts) == 2 and parts[1] in CONTROL_TEXT_FILES:
            return True
        return len(parts) >= 2 and parts[1] in {
            "calibration",
            "candidates",
            "feedback",
            "kernel_outputs",
            "selection_manager",
            "trajectories",
        }

    if parts[0] == "lanes":
        # Codex's durable rollout is the authoritative copy of a temporary
        # conversation.  Other CODEX_HOME state databases are unnecessary and
        # may be large, so only sessions are retained from that directory.
        if ".codex_home" in parts:
            index = parts.index(".codex_home")
            return len(parts) > index + 1 and parts[index + 1] == "sessions"
        return True

    if parts[0] == "search":
        return True

    return False


def _copy_atomic(source: Path, destination: Path) -> bool:
    try:
        source_stat = source.stat()
    except FileNotFoundError:
        return False
    try:
        destination_stat = destination.stat()
        if (
            destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        ):
            return False
    except FileNotFoundError:
        pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except FileNotFoundError:
        temporary.unlink(missing_ok=True)
        return False
    return True


def snapshot(run_root: Path, destination: Path) -> tuple[int, int]:
    copied = 0
    observed = 0
    allowed_control_dirs = {
        "calibration",
        "candidates",
        "feedback",
        "kernel_outputs",
        "selection_manager",
        "trajectories",
    }
    for directory, dirnames, filenames in os.walk(run_root, topdown=True):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(run_root)
        parts = relative_directory.parts

        if not parts:
            dirnames[:] = [name for name in dirnames if name in {"control", "lanes", "search"}]
        elif any(part in EXCLUDED_PARTS for part in parts):
            dirnames[:] = []
            continue
        elif relative_directory == Path("control"):
            dirnames[:] = [name for name in dirnames if name in allowed_control_dirs]
        elif ".codex_home" in parts:
            index = parts.index(".codex_home")
            if len(parts) == index + 1:
                dirnames[:] = [name for name in dirnames if name == "sessions"]
            elif parts[index + 1] != "sessions":
                dirnames[:] = []
                continue

        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_PARTS]
        for filename in filenames:
            source = directory_path / filename
            if source.is_symlink():
                continue
            relative = source.relative_to(run_root)
            if source.suffix.lower() not in TEXT_SUFFIXES or not _included(relative):
                continue
            observed += 1
            if _copy_atomic(source, destination / relative):
                copied += 1
    return observed, copied


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_status(destination: Path, payload: dict[str, object]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "ARCHIVE_STATUS.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    destination = args.archive_root.resolve() / run_root.name
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {run_root}")

    total_copied = 0
    while True:
        observed, copied = snapshot(run_root, destination)
        total_copied += copied
        running = _pid_is_running(args.watch_pid)
        _write_status(
            destination,
            {
                "archive_root": str(destination),
                "controller_pid": args.watch_pid,
                "controller_running": running,
                "files_copied_this_pass": copied,
                "files_observed": observed,
                "run_root": str(run_root),
                "status": "watching" if running and not args.once else "complete",
                "total_copy_operations": total_copied,
                "updated_at": time.time(),
            },
        )
        if args.once or not running:
            break
        time.sleep(max(2.0, args.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
