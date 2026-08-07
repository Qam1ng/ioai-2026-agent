from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
import tempfile
from pathlib import Path

from .claude import ClaudeRunner, SecretBroker
from .codex import CodexRunner
from .search_orchestrator import SearchOrchestrator, SearchRunConfig


def _research_backends(value: str) -> tuple[str, str, str, str]:
    values = tuple(item.strip().lower() for item in value.split(","))
    if len(values) != 4 or any(item not in {"claude", "codex"} for item in values):
        raise argparse.ArgumentTypeError(
            "must be four comma-separated values for R1,R2,R3,R4: claude or codex"
        )
    return values  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the decoupled IOAI task-analysis and web-research system"
    )
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--prior-lessons-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--duration-seconds",
        type=int,
        required=True,
        help="caller-assigned Search module budget; no built-in 30-minute default",
    )
    parser.add_argument(
        "--competition-mode", choices=["practice", "formal"], required=True
    )
    parser.add_argument(
        "--prompt-spec",
        type=Path,
        default=package_root / "SEARCH_PROMPTS_ZH.md",
    )
    parser.add_argument(
        "--analyst-backend", choices=["claude", "codex"], default="claude"
    )
    parser.add_argument(
        "--research-backends",
        type=_research_backends,
        default=("claude", "claude", "claude", "claude"),
        metavar="R1,R2,R3,R4",
    )
    parser.add_argument("--claude-bin", type=Path, default=Path("/opt/homebrew/bin/claude"))
    parser.add_argument("--codex-bin", type=Path, default=Path("/opt/homebrew/bin/codex"))
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--claude-model", default="claude-opus-5")
    parser.add_argument("--codex-model", default="gpt-5.6-codex")
    parser.add_argument("--claude-effort", default="high")
    parser.add_argument("--codex-effort", default="high")
    parser.add_argument("--max-budget-usd-per-claude-agent", type=float, default=20.0)
    parser.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="read one Anthropic key line from stdin; never persist it",
    )
    return parser


def _read_anthropic_key(from_stdin: bool) -> str:
    if from_stdin:
        value = (
            getpass.getpass("")
            if sys.stdin.isatty()
            else sys.stdin.readline().rstrip("\r\n")
        )
    else:
        value = getpass.getpass("Anthropic API key: ")
    if not value:
        raise ValueError("an Anthropic API key is required by the selected backends")
    return value


async def async_main(args: argparse.Namespace) -> int:
    research_backends = tuple(args.research_backends)
    selected = {args.analyst_backend, *research_backends}
    config = SearchRunConfig(
        assets_dir=args.assets_dir.resolve(),
        prior_lessons_dir=(
            args.prior_lessons_dir.resolve() if args.prior_lessons_dir else None
        ),
        output_root=args.output_root.resolve(),
        prompt_spec=args.prompt_spec.resolve(),
        duration_s=args.duration_seconds,
        competition_mode=args.competition_mode,
        analyst_backend=args.analyst_backend,
        research_backends=research_backends,
    )

    runners: dict[str, object] = {}
    if "codex" in selected:
        runners["codex"] = CodexRunner(
            codex_bin=args.codex_bin.expanduser().absolute(),
            model=args.codex_model,
            effort=args.codex_effort,
        )

    if "claude" not in selected:
        orchestrator = SearchOrchestrator(config, runners=runners)  # type: ignore[arg-type]
        session = await orchestrator.execute()
    else:
        api_key = _read_anthropic_key(args.api_key_stdin)
        with tempfile.TemporaryDirectory(prefix="ioai-search-secret-") as raw_temp:
            socket_path = Path(raw_temp) / "anthropic.sock"
            runners["claude"] = ClaudeRunner(
                claude_bin=args.claude_bin.expanduser().absolute(),
                python_bin=args.python_bin.expanduser().absolute(),
                package_root=Path(__file__).resolve().parents[1],
                model=args.claude_model,
                effort=args.claude_effort,
                secret_socket=socket_path,
                max_budget_usd=args.max_budget_usd_per_claude_agent,
            )
            async with SecretBroker(api_key, socket_path):
                orchestrator = SearchOrchestrator(
                    config, runners=runners  # type: ignore[arg-type]
                )
                session = await orchestrator.execute()
        api_key = ""

    print(session)
    status = json.loads((session / "RUN_STATUS.json").read_text(encoding="utf-8"))
    return 0 if status["status"] in {"complete", "degraded"} else 2


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
