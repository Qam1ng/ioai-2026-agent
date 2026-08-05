"""Configuration for the backup system.

Deliberately flat. Everything the operator needs to change on competition day is
one TOML file and three CLI flags; nothing important is computed from anything
else.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a gitignored .env, without overriding the shell.

    The API key lives here and nowhere else: not in the TOML, not in any file
    git tracks, and never in a prompt or a trace.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

#: Both wire formats come off one Azure resource and one key. Verified live:
#:   POST <host>/openai/v1/responses     + Authorization: Bearer  -> 200
#:   POST <host>/anthropic/v1/messages   + x-api-key              -> 200
#:
#: The two bases differ in how much of the path they carry, because the two
#: clients append different things:
#:   * Claude Code's SDK appends "/v1/messages", so its base stops at
#:     ".../anthropic". Writing ".../anthropic/v1" yields ".../anthropic/v1/v1/
#:     messages" and a 404.
#:   * Codex appends "/responses", so its base carries the full ".../openai/v1".
#: Getting these two confused is the single most common way to make every agent
#: silently fail.
AZURE_HOST = "https://prosgrow-aoai-prod.services.ai.azure.com"
DEFAULT_ANTHROPIC_BASE = f"{AZURE_HOST}/anthropic"
DEFAULT_OPENAI_BASE = f"{AZURE_HOST}/openai/v1"

#: Kept for reference: OpenRouter also serves both formats, but codex-cli
#: 0.144.3 does not attach an Authorization header to a custom provider over
#: wire_api="responses" there (401 "Missing Authentication header", while
#: `codex doctor` reports the key as present). Azure does not have that problem.
OPENROUTER_ANTHROPIC_BASE = "https://openrouter.ai/api"
OPENROUTER_OPENAI_BASE = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class AgentConfig:
    binary: Path
    model: str
    effort: str
    base_url: str
    turn_minutes: float
    max_rounds: int
    #: Which CLI drives this lane: "codex" or "claude_code". It is a separate
    #: knob from the model on purpose. Codex 0.144.3 does not attach the
    #: Authorization header when talking to a custom OpenRouter provider over
    #: wire_api="responses" -- `codex doctor` reports the key as present and the
    #: request still comes back 401 "Missing Authentication header". If that is
    #: still true on the day, flip this lane to "claude_code" and keep the same
    #: OpenRouter model slug: the lane keeps its model diversity and only
    #: changes which client speaks to OpenRouter.
    runner: str


@dataclass(frozen=True)
class ManagerConfig:
    binary: Path
    model: str
    effort: str
    base_url: str
    timeout_seconds: float
    interval_seconds: float


@dataclass(frozen=True)
class RunConfig:
    workspace_root: Path
    api_key_env: str
    llm_host: str
    max_submissions: int
    kaggle_concurrency: int
    floor_after_minutes: float
    submission_only_minutes_before_end: float
    poll_seconds: float
    kernel_timeout_minutes: float
    retry_attempts: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class BackupConfig:
    run: RunConfig
    codex: AgentConfig
    claude: AgentConfig
    manager: ManagerConfig

    @classmethod
    def load(cls, path: Path, *, repo_root: Path) -> "BackupConfig":
        raw = tomllib.loads(Path(path).expanduser().read_text(encoding="utf-8"))

        def section(name: str) -> dict[str, Any]:
            value = raw.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"[{name}] must be a TOML table")
            return value

        _load_dotenv(repo_root / ".env")
        run = section("run")
        workspace = Path(run.get("workspace_root", "workspace/backup_system"))
        if not workspace.is_absolute():
            workspace = repo_root / workspace

        host = str(run.get("llm_host", AZURE_HOST)).rstrip("/")

        def base_for(runner: str) -> str:
            # The endpoint shape follows the CLIENT, not the model. Deriving it
            # rather than configuring it is what makes the escape hatch a真 one-
            # liner: flipping `runner` moves the lane to the matching endpoint.
            return f"{host}/openai/v1" if runner == "codex" else f"{host}/anthropic"

        def agent(name: str, default_model: str, default_runner: str) -> AgentConfig:
            value = section(name)
            runner = str(value.get("runner", default_runner))
            binary_default = (
                "/opt/homebrew/bin/codex" if runner == "codex"
                else "/opt/homebrew/bin/claude"
            )
            base_default = base_for(runner)
            return AgentConfig(
                binary=Path(value.get("binary", binary_default)),
                model=str(value.get("model", default_model)),
                effort=str(value.get("effort", "high")),
                base_url=str(value.get("base_url", base_default)).rstrip("/"),
                turn_minutes=float(value.get("turn_minutes", 12)),
                max_rounds=int(value.get("max_rounds", 40)),
                runner=runner,
            )

        manager = section("manager")
        config = cls(
            run=RunConfig(
                workspace_root=workspace.resolve(),
                api_key_env=str(run.get("api_key_env", "IOAI_LLM_API_KEY")),
                llm_host=host,
                max_submissions=int(run.get("max_submissions", 50)),
                kaggle_concurrency=int(run.get("kaggle_concurrency", 2)),
                floor_after_minutes=float(run.get("floor_after_minutes", 8)),
                submission_only_minutes_before_end=float(
                    run.get("submission_only_minutes_before_end", 25)
                ),
                poll_seconds=float(run.get("poll_seconds", 10)),
                kernel_timeout_minutes=float(run.get("kernel_timeout_minutes", 45)),
                retry_attempts=int(run.get("retry_attempts", 3)),
                retry_backoff_seconds=float(run.get("retry_backoff_seconds", 45)),
            ),
            codex=agent("codex", "gpt-5.6-sol", "codex"),
            claude=agent("claude", "claude-fable-5", "claude_code"),
            manager=ManagerConfig(
                binary=Path(manager.get("binary", "/opt/homebrew/bin/claude")),
                model=str(manager.get("model", "claude-fable-5")),
                effort=str(manager.get("effort", "high")),
                base_url=str(
                    manager.get("base_url", base_for("claude_code"))
                ).rstrip("/"),
                timeout_seconds=float(manager.get("timeout_seconds", 180)),
                interval_seconds=float(manager.get("interval_seconds", 60)),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 1 <= self.run.max_submissions <= 50:
            raise ValueError("run.max_submissions must be in 1..50")
        if not 1 <= self.run.kaggle_concurrency <= 2:
            raise ValueError("run.kaggle_concurrency must be 1 or 2 (Kaggle's limit)")
        if self.run.poll_seconds <= 0:
            raise ValueError("run.poll_seconds must be positive")
        if self.run.retry_attempts < 1:
            raise ValueError("run.retry_attempts must be at least 1")
        for name, agent in (("codex", self.codex), ("claude", self.claude)):
            if agent.turn_minutes <= 0:
                raise ValueError(f"{name}.turn_minutes must be positive")
            if agent.max_rounds < 1:
                raise ValueError(f"{name}.max_rounds must be at least 1")
            if agent.runner not in {"codex", "claude_code"}:
                raise ValueError(f"{name}.runner must be codex or claude_code")
            # A base_url that does not match the client's shape produces a 404
            # on every single call, which is expensive to notice at runtime.
            if agent.runner == "codex" and not agent.base_url.endswith("/v1"):
                raise ValueError(
                    f"{name}.base_url must end in /v1 for the codex client "
                    f"(it appends /responses); got {agent.base_url!r}"
                )
            if agent.runner == "claude_code" and agent.base_url.endswith("/v1"):
                raise ValueError(
                    f"{name}.base_url must NOT end in /v1 for the claude_code "
                    f"client (its SDK appends /v1/messages); got {agent.base_url!r}"
                )
        if self.manager.base_url.endswith("/v1"):
            raise ValueError(
                "manager.base_url must not end in /v1; Claude Code appends "
                "/v1/messages itself"
            )
        if self.manager.timeout_seconds <= 0 or self.manager.interval_seconds <= 0:
            raise ValueError("manager timeout/interval must be positive")
