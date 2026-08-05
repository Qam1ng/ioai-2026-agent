from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backup_system.config import BackupConfig  # noqa: E402
from backup_system.run import BackupRun  # noqa: E402


def _parse_task(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--task must look like <slug>=<absolute assets dir>"
        )
    slug, _, assets = value.partition("=")
    path = Path(assets).expanduser()
    if not slug.strip():
        raise argparse.ArgumentTypeError("--task slug must not be empty")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"assets dir does not exist: {path}")
    return slug.strip(), path.resolve()


def _probe(url: str, payload: dict, key: str) -> str:
    """Send one real request and report exactly what came back."""
    import urllib.error
    import urllib.request

    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "user-agent": "ioai-backup-system/1.0",
    }
    if key:
        # Both shapes are accepted by the Azure resource; sending both means one
        # probe covers whichever the real client will use.
        headers["api-key"] = key
        headers["x-api-key"] = key
        headers["authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
        text = ""
        if isinstance(body.get("content"), list) and body["content"]:
            text = str(body["content"][0].get("text", ""))[:40]
        elif body.get("status"):
            text = f"status={body['status']}"
        return f"HTTP {response.status} OK {text}".strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:180]
        note = {
            401: "key missing or rejected",
            404: "ROUTE DOES NOT EXIST — wrong base URL",
        }.get(exc.code, "")
        return f"HTTP {exc.code} {note} :: {detail}"
    except Exception as exc:  # noqa: BLE001
        return f"unreachable: {type(exc).__name__}: {exc}"


def doctor(config: BackupConfig) -> int:
    print("=== binaries ===")
    problems = 0
    for name, binary in (
        ("codex", config.codex.binary),
        ("claude", config.claude.binary),
        ("manager (claude)", config.manager.binary),
    ):
        found = Path(binary).is_file() or shutil.which(str(binary))
        print(f"  {name:<18} {binary} -> {'ok' if found else 'MISSING'}")
        problems += 0 if found else 1
    kaggle_bin = shutil.which("kaggle")
    print(f"  {'kaggle':<18} {kaggle_bin or '(not on PATH)'} -> "
          f"{'ok' if kaggle_bin else 'MISSING (only needed with --live)'}")

    print("\n=== API key ===")
    key = os.environ.get(config.run.api_key_env, "")
    print(f"  {config.run.api_key_env} = "
          f"{'set (' + str(len(key)) + ' chars, from shell or .env)' if key else 'EMPTY — required before --live'}")
    problems += 0 if key else 1

    print("\n=== endpoints (one real request each) ===")
    for label, agent, suffix, payload in (
        ("codex", config.codex, "/responses",
         {"model": config.codex.model, "input": "Reply with exactly: pong",
          "max_output_tokens": 32}),
        ("claude", config.claude, "/v1/messages",
         {"model": config.claude.model, "max_tokens": 32,
          "messages": [{"role": "user", "content": "Reply with exactly: pong"}]}),
        ("manager", config.manager, "/v1/messages",
         {"model": config.manager.model, "max_tokens": 32,
          "messages": [{"role": "user", "content": "Reply with exactly: pong"}]}),
    ):
        # The suffix each client appends is a property of the client, so probe
        # exactly what that client will actually request.
        shape = "/responses" if getattr(agent, "runner", "claude_code") == "codex" \
            else "/v1/messages"
        url = f"{agent.base_url}{shape if label != 'manager' else suffix}"
        result = _probe(url, payload, key)
        print(f"  {label:<8} {url}")
        print(f"           {result}")
        problems += 0 if result.startswith("HTTP 2") else 1

    print("\n=== verdict ===")
    print("  ready to run" if problems == 0 else
          f"  {problems} blocking problem(s) above")
    return 0 if problems == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser("backup_system")
    parser.add_argument("command", choices=["doctor", "run"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "backup_system.toml"))
    parser.add_argument("--task", action="append", type=_parse_task, default=[],
                        metavar="SLUG=ASSETS_DIR",
                        help="repeat once per problem, up to three")
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--kaggle-user", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--live", action="store_true",
                        help="actually submit to Kaggle; without it nothing is spent")
    args = parser.parse_args()

    config = BackupConfig.load(Path(args.config), repo_root=ROOT)
    if args.command == "doctor":
        return doctor(config)

    if not args.task:
        parser.error("run requires at least one --task SLUG=ASSETS_DIR")
    if len(args.task) > 3:
        parser.error("a competition day has three problems; --task given more than three times")
    if args.live and not os.environ.get(config.run.api_key_env):
        parser.error(
            f"{config.run.api_key_env} is empty; export your OpenRouter key before --live"
        )

    run = BackupRun(
        config=config, tasks=args.task, minutes=args.minutes, live=args.live,
        kaggle_user=args.kaggle_user, run_id=args.run_id,
    )
    status = asyncio.run(run.execute())
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
