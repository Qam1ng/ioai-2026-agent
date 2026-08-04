from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentSandbox:
    prefix: tuple[str, ...]
    denied_paths: tuple[Path, ...]
    protected_write_roots: tuple[Path, ...] = ()
    writable_roots: tuple[Path, ...] = ()
    read_only_roots: tuple[Path, ...] = ()
    protected_read_roots: tuple[Path, ...] = ()
    readable_roots: tuple[Path, ...] = ()

    def wrap(self, command: list[str]) -> list[str]:
        return [*self.prefix, *command]


def _seatbelt_string(path: Path) -> str:
    return json.dumps(str(path.expanduser().resolve()))


def build_agent_sandbox(
    *,
    probe_secret: Path,
    protected_write_roots: tuple[Path, ...] = (),
    writable_roots: tuple[Path, ...] = (),
    read_only_roots: tuple[Path, ...] = (),
    protected_read_roots: tuple[Path, ...] = (),
    readable_roots: tuple[Path, ...] = (),
) -> AgentSandbox:
    """Build a macOS Seatbelt boundary that hides operator Kaggle credentials."""
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError(
            "formal live mode requires an OS Agent sandbox; this build currently "
            "supports macOS /usr/bin/sandbox-exec"
        )
    operator_home = Path.home().resolve()
    denied = {
        operator_home / ".kaggle",
        operator_home / ".config" / "kaggle",
        operator_home / "Library" / "Application Support" / "kaggle",
        Path(probe_secret).resolve(),
    }
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    if configured:
        denied.add(Path(configured).expanduser().resolve())
    # Claude Code 需要读取自身进程信息；只开放 target self，子进程不能枚举
    # Controller/Codex 父进程环境中的 Kaggle 或 OpenRouter 凭证。
    rules = [
        "(version 1)", "(allow default)", "(deny process-info*)",
        "(allow process-info* (target self))",
    ]
    protected = tuple(sorted(
        {Path(item).resolve() for item in protected_write_roots}, key=str
    ))
    writable = tuple(sorted(
        {Path(item).resolve() for item in writable_roots}, key=str
    ))
    read_only = tuple(sorted(
        {Path(item).resolve() for item in read_only_roots}, key=str
    ))
    protected_read = tuple(sorted(
        {Path(item).resolve() for item in protected_read_roots}, key=str
    ))
    readable = tuple(sorted(
        {Path(item).resolve() for item in readable_roots}, key=str
    ))
    for path in protected:
        rules.append(f"(deny file-write* (subpath {_seatbelt_string(path)}))")
    # Seatbelt 同等匹配时后写规则生效：先封整个 session，再只开放本 Agent 工作区。
    for path in writable:
        rules.append(f"(allow file-write* (subpath {_seatbelt_string(path)}))")
    for path in read_only:
        rules.append(f"(deny file-write* (subpath {_seatbelt_string(path)}))")
    for path in protected_read:
        rules.append(f"(deny file-read* (subpath {_seatbelt_string(path)}))")
    for path in readable:
        rules.append(f"(allow file-read* (subpath {_seatbelt_string(path)}))")
    for path in sorted(denied, key=str):
        rules.append(f"(deny file-read* (subpath {_seatbelt_string(path)}))")
        rules.append(f"(deny file-write* (subpath {_seatbelt_string(path)}))")
    # 禁止 Agent 直接执行本机 Kaggle CLI；它们仍可使用浏览器进行公开研究。
    kaggle_paths = {
        Path(value).resolve()
        for value in (
            shutil.which("kaggle"),
            str(Path(sys.executable).parent / "kaggle"),
        )
        if value and Path(value).exists()
    }
    for path in sorted(kaggle_paths, key=str):
        rules.append(f"(deny process-exec (literal {_seatbelt_string(path)}))")
    profile = " ".join(rules)
    return AgentSandbox(
        prefix=("/usr/bin/sandbox-exec", "-p", profile),
        denied_paths=tuple(sorted(denied, key=str)),
        protected_write_roots=protected,
        writable_roots=writable,
        read_only_roots=read_only,
        protected_read_roots=protected_read,
        readable_roots=readable,
    )


def verify_agent_sandbox(sandbox: AgentSandbox, *, probe_secret: Path) -> None:
    probe_secret = Path(probe_secret).resolve()
    probe_secret.parent.mkdir(parents=True, exist_ok=True)
    probe_secret.write_text("sandbox-probe-secret\n", encoding="utf-8")
    script = (
        "if /bin/cat \"$1\" >/dev/null 2>&1; then exit 21; fi; "
        "if /bin/ps -p \"$PPID\" >/dev/null 2>&1; then exit 22; fi; "
        "exit 0"
    )
    result = subprocess.run(
        sandbox.wrap(["/bin/sh", "-c", script, "sh", str(probe_secret)]),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Agent OS sandbox verification failed: "
            f"exit={result.returncode}, stderr={result.stderr[-500:]}"
        )
