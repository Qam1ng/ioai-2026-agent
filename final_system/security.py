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
    """Build an OS boundary that hides operator Kaggle credentials.

    macOS uses Seatbelt (/usr/bin/sandbox-exec). Linux uses bubblewrap with the
    same contract: the agent cannot read the operator's Kaggle credentials,
    cannot execute the local Kaggle CLI, cannot write outside its own roots,
    and cannot inspect processes outside its own tree.
    """
    if sys.platform == "linux" and Path("/usr/bin/bwrap").is_file():
        return _build_bwrap_sandbox(
            probe_secret=probe_secret,
            protected_write_roots=protected_write_roots,
            writable_roots=writable_roots,
            read_only_roots=read_only_roots,
            protected_read_roots=protected_read_roots,
            readable_roots=readable_roots,
        )
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError(
            "formal live mode requires an OS Agent sandbox; this build supports "
            "macOS /usr/bin/sandbox-exec and Linux /usr/bin/bwrap"
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


def _build_bwrap_sandbox(
    *,
    probe_secret: Path,
    protected_write_roots: tuple[Path, ...] = (),
    writable_roots: tuple[Path, ...] = (),
    read_only_roots: tuple[Path, ...] = (),
    protected_read_roots: tuple[Path, ...] = (),
    readable_roots: tuple[Path, ...] = (),
) -> AgentSandbox:
    """The Seatbelt contract, said in bubblewrap.

    The whole filesystem is re-mounted read-only, so "may not write" is the
    default rather than a rule; the agent's own roots are then bound back
    writable. Credential directories are shadowed with empty tmpfs, the Kaggle
    CLI binaries are shadowed with /dev/null, and --unshare-pid gives the agent
    a private PID namespace so the controller — whose environment holds the
    real keys — does not exist as far as /proc is concerned.
    """
    operator_home = Path.home().resolve()
    denied = {
        operator_home / ".kaggle",
        operator_home / ".config" / "kaggle",
        Path(probe_secret).resolve(),
    }
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    if configured:
        denied.add(Path(configured).expanduser().resolve())

    args: list[str] = [
        "/usr/bin/bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--unshare-pid",
        "--die-with-parent",
    ]
    # A private /tmp replaces the shared one; the operator TMPDIR, if it points
    # elsewhere (ours lives on /data), must stay writable for the tools.
    tmpdir = os.environ.get("TMPDIR", "").strip()
    if tmpdir and Path(tmpdir).is_dir():
        args += ["--bind", tmpdir, tmpdir]
    writable = tuple(sorted({Path(x).resolve() for x in writable_roots}, key=str))
    for path in writable:
        if path.exists():
            args += ["--bind", str(path), str(path)]
    # Shadow the credentials AFTER the writable binds above: bwrap applies
    # mounts in order, last wins, and a secret may live under a writable root
    # (the probe does). A file is shadowed with /dev/null (an empty dir tmpfs
    # cannot cover a single file), a directory with an empty tmpfs.
    for path in sorted(denied, key=str):
        if path.is_file():
            args += ["--ro-bind", "/dev/null", str(path)]
        elif path.is_dir():
            args += ["--tmpfs", str(path)]
    protected_read = tuple(sorted(
        {Path(x).resolve() for x in protected_read_roots}, key=str))
    for path in protected_read:
        if path.is_dir():
            args += ["--tmpfs", str(path)]
    # The agents may not run the operator Kaggle CLI even though the network
    # stays open: submission authority belongs to the broker alone.
    kaggle_paths = {
        Path(value).resolve()
        for value in (
            shutil.which("kaggle"),
            str(Path(sys.executable).parent / "kaggle"),
        )
        if value and Path(value).exists()
    }
    for path in sorted(kaggle_paths, key=str):
        args += ["--ro-bind", "/dev/null", str(path)]

    protected = tuple(sorted(
        {Path(x).resolve() for x in protected_write_roots}, key=str))
    read_only = tuple(sorted(
        {Path(x).resolve() for x in read_only_roots}, key=str))
    readable = tuple(sorted(
        {Path(x).resolve() for x in readable_roots}, key=str))
    return AgentSandbox(
        prefix=tuple(args),
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
    if sys.platform == "linux":
        # In a private PID namespace the controller must simply not exist; its
        # pid is the strongest probe, because that process's environment block
        # is where the real credentials live.
        script = (
            "if /bin/cat \"$1\" >/dev/null 2>&1; then exit 21; fi; "
            f"if [ -e /proc/{os.getpid()} ]; then exit 22; fi; "
            "exit 0"
        )
    else:
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
