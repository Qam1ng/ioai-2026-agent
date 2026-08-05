"""A read-only Kaggle gateway the sandboxed agents are allowed to call.

IOAI requires the AI system to download the competition data itself, but the
agent sandbox denies `~/.kaggle` and the real Kaggle CLI so that nothing except
the Broker can spend the 50-submission budget. Those two facts collide: reading
data and spending quota are different operations that were being blocked by one
rule.

So the credential never enters the sandbox at all. The controller runs this
gateway *outside* it, holding the operator credentials; each agent gets a tiny
`kaggle` shim on its PATH that forwards argv over a unix socket. The gateway
accepts a fixed whitelist of read-only subcommands and refuses everything else,
so an agent can pull the data and inspect the competition, and still cannot
submit, push a kernel, or read the token — there is no code path that returns
it.

The whitelist is on the *server* side deliberately. A shim that validated its
own argv would be one `chmod`/rewrite away from being bypassed, and the agents
can write to their own workspace.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from .io import append_jsonl

#: (subcommand, action) pairs an agent may run. Everything that consumes quota
#: or mutates remote state is absent by construction: no `submit`, no
#: `kernels push`, no `datasets create`. `competitions list` is included so an
#: agent can confirm the slug it was given actually exists.
ALLOWED: frozenset[tuple[str, str]] = frozenset({
    ("competitions", "download"),
    ("competitions", "files"),
    ("competitions", "list"),
    ("competitions", "leaderboard"),
})

#: Flags that would turn an allowed read into something else.
FORBIDDEN_FLAGS: frozenset[str] = frozenset({"--submit", "-f", "--file"})

_SHIM = """#!/bin/sh
# Read-only Kaggle shim. The real CLI and ~/.kaggle are outside this sandbox;
# this forwards argv to the controller-side gateway, which enforces the
# whitelist. Submitting is not available here by design — the Broker owns the
# competition's submission budget.
exec {python} {client} "$@"
"""

_CLIENT = '''import json, socket, sys
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    sock.connect({socket_path!r})
except OSError as exc:
    sys.stderr.write("kaggle gateway unavailable: %s\\n" % exc)
    raise SystemExit(70)
sock.sendall(json.dumps({{"argv": sys.argv[1:], "cwd": __import__("os").getcwd()}}).encode() + b"\\n")
sock.shutdown(socket.SHUT_WR)
chunks = []
while True:
    piece = sock.recv(65536)
    if not piece:
        break
    chunks.append(piece)
reply = json.loads(b"".join(chunks) or b'{{"rc":70,"stderr":"empty reply"}}')
sys.stdout.write(reply.get("stdout", ""))
sys.stderr.write(reply.get("stderr", ""))
raise SystemExit(int(reply.get("rc", 70)))
'''


def classify(argv: list[str]) -> tuple[bool, str]:
    """Whitelist check. Returns (allowed, reason)."""
    positional = [item for item in argv if not item.startswith("-")]
    if len(positional) < 2:
        return False, "expected '<group> <action>', e.g. competitions download"
    pair = (positional[0], positional[1])
    if pair not in ALLOWED:
        allowed = ", ".join(sorted(f"{a} {b}" for a, b in ALLOWED))
        return False, (
            f"'{pair[0]} {pair[1]}' is not available to agents. This gateway is "
            f"read-only; allowed: {allowed}. Submissions and kernel pushes are "
            "made by the Broker, which owns the competition's submission budget."
        )
    bad = sorted(set(argv) & FORBIDDEN_FLAGS)
    if bad:
        return False, f"flag(s) not permitted on a read-only call: {bad}"
    return True, ""


class KaggleReadOnlyGateway:
    """Serves whitelisted Kaggle reads to sandboxed agents over a unix socket."""

    def __init__(
        self, root: Path, *, kaggle_bin: str, python: str,
        events_path: Path | None = None, timeout_s: float = 900.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # sockaddr_un.sun_path is ~104 bytes on macOS and the session lives
        # under a workspace path that alone can exceed it, so the socket goes
        # to a short temp path keyed by the session rather than beside the shim.
        digest = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:10]
        self.socket_path = Path(tempfile.gettempdir()) / f"ioai-kg-{digest}.sock"
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.kaggle_bin = kaggle_bin
        self.python = python
        self.events_path = events_path
        self.timeout_s = timeout_s
        self._server: asyncio.AbstractServer | None = None

    def _write_shim(self) -> None:
        client = self.bin_dir / "_kaggle_gateway_client.py"
        client.write_text(
            _CLIENT.format(socket_path=str(self.socket_path)), encoding="utf-8"
        )
        shim = self.bin_dir / "kaggle"
        shim.write_text(
            _SHIM.format(python=shlex.quote(self.python), client=shlex.quote(str(client))),
            encoding="utf-8",
        )
        shim.chmod(0o755)

    def _log(self, **payload) -> None:
        if self.events_path is not None:
            append_jsonl(self.events_path, {"timestamp": time.time(), **payload})

    async def _handle(self, reader, writer) -> None:
        reply: dict = {"rc": 70, "stdout": "", "stderr": "gateway error\n"}
        try:
            raw = await asyncio.wait_for(reader.read(65536), timeout=30)
            request = json.loads(raw or b"{}")
            argv = [str(item) for item in request.get("argv", [])]
            allowed, reason = classify(argv)
            if not allowed:
                self._log(event="kaggle_gateway_denied", argv=argv, reason=reason)
                reply = {"rc": 77, "stdout": "", "stderr": reason + "\n"}
            else:
                cwd = str(request.get("cwd") or self.root)
                result = await asyncio.to_thread(
                    subprocess.run, [self.kaggle_bin, *argv],
                    capture_output=True, text=True, timeout=self.timeout_s,
                    cwd=cwd if Path(cwd).is_dir() else str(self.root),
                )
                self._log(
                    event="kaggle_gateway_call", argv=argv,
                    returncode=result.returncode, cwd=cwd,
                )
                reply = {
                    "rc": result.returncode,
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                }
        except Exception as exc:  # noqa: BLE001
            self._log(event="kaggle_gateway_error", error=f"{type(exc).__name__}: {exc}")
            reply = {"rc": 70, "stdout": "", "stderr": f"gateway failure: {exc}\n"}
        finally:
            try:
                writer.write(json.dumps(reply).encode())
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                pass
            writer.close()

    async def start(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._write_shim()
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self.socket_path)
        )
        os.chmod(self.socket_path, 0o600)
        self._log(event="kaggle_gateway_started", socket=str(self.socket_path))

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._server = None
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)
